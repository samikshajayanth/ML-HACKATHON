
import numpy as np


def viterbi_decode(candidate_layers, transition_model, timestamps=None):
    """
    Find the most likely sequence of road arcs using Viterbi.
    """

    if not candidate_layers:
        return []

    if any(not layer for layer in candidate_layers):
        raise ValueError("Every observation must have candidates.")

    if timestamps is not None:
        if len(timestamps) != len(candidate_layers):
            raise ValueError("Timestamp count must match candidate layers.")

        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("Timestamps must be strictly increasing.")

    scores = []
    backpointers = []

    # Initialization
    first_scores = {}

    for candidate in candidate_layers[0]:
        arc_id = int(candidate["arc_id"])
        probability = float(candidate["probability"])

        if not np.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("Invalid emission probability.")

        first_scores[arc_id] = np.log(probability)

    scores.append(first_scores)
    backpointers.append({})

    # Dynamic programming
    for t in range(1, len(candidate_layers)):
        current_scores = {}
        current_backpointers = {}

        for candidate in candidate_layers[t]:
            current_arc = int(candidate["arc_id"])
            emission_probability = float(candidate["probability"])

            if (
                not np.isfinite(emission_probability)
                or not 0 < emission_probability <= 1
            ):
                raise ValueError("Invalid emission probability.")

            emission_log = np.log(emission_probability)

            best_score = -np.inf
            best_previous_arc = None

            for previous_arc, previous_score in scores[t - 1].items():

                dt = 1.0

                if timestamps is not None:
                    dt = timestamps[t] - timestamps[t - 1]

                transition_score = transition_model.score(
                    previous_arc,
                    current_arc,
                    dt=dt
                )

                total_score = (
                    previous_score
                    + transition_score
                    + emission_log
                )

                if total_score > best_score:
                    best_score = total_score
                    best_previous_arc = previous_arc

            current_scores[current_arc] = best_score
            current_backpointers[current_arc] = best_previous_arc

        scores.append(current_scores)
        backpointers.append(current_backpointers)

    # Termination
    best_final_arc = max(scores[-1], key=scores[-1].get)

    # Backtracking
    best_path = [best_final_arc]

    for t in range(len(candidate_layers) - 1, 0, -1):
        previous_arc = backpointers[t][best_path[-1]]
        best_path.append(previous_arc)

    best_path.reverse()

    return best_path
