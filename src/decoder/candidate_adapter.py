
import numpy as np
import pandas as pd


def dataframe_to_candidate_layers(
    candidates: pd.DataFrame,
    probability_col: str = "emission_probability",
):
    required = {
        "observation_id",
        "timestamp",
        "candidate_arc_id",
        probability_col,
    }

    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if candidates.empty:
        raise ValueError("Candidate DataFrame is empty.")

    candidates = candidates.copy()

    if candidates[list(required)].isnull().any().any():
        raise ValueError("Missing values in candidate DataFrame.")

    layers = []
    timestamps = []

    for observation_id, group in candidates.groupby(
        "observation_id", sort=True
    ):
        if group["timestamp"].nunique() != 1:
            raise ValueError(
                f"Inconsistent timestamps for observation {observation_id}"
            )

        probabilities = group[probability_col].to_numpy(dtype=float)

        if (
            not np.isfinite(probabilities).all()
            or (probabilities <= 0).any()
        ):
            raise ValueError("Emission probabilities must be positive.")

        probabilities = probabilities / probabilities.sum()

        layer = []

        for (_, row), probability in zip(
            group.iterrows(), probabilities
        ):
            layer.append({
                "arc_id": int(row["candidate_arc_id"]),
                "probability": float(probability),
            })

        layers.append(layer)
        timestamps.append(float(group["timestamp"].iloc[0]))

    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Timestamps must be strictly increasing.")

    return layers, timestamps

