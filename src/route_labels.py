"""Per-observation target labels from the ordered reference route.

The dataset gives, per trip, an ordered list of route arcs (.route) but not which route arc each
GPS observation belongs to. This module aligns observations to route positions:

  * only candidates whose arc is on the route can be labelled (others are never correct),
  * the chosen route position may not go backwards between consecutive observations
    (small cost per skipped arc, restart penalty if the sequence is broken),
  * among the allowed sequences, the one with the smallest total GPS-to-arc distance wins,
  * the same arc may appear several times in the route (loops); each occurrence is a separate state.

Observations with no on-route candidate get all-zero labels: they are candidate-generation misses
and are counted in the recall report.

THIS IS A DRAFT LABEL DEFINITION. The team (Member 1 owns labels) must confirm it matches the
organisers' notion of "correctly aligned reference segment".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def assign_route_labels(cands: pd.DataFrame, route, *, max_jump: int = 30, jump_cost: float = 0.01,
                        restart_penalty: float = 100.0) -> pd.DataFrame:
    """cands: ONE trip, contract column names (obs_id, candidate_id = arc index, dist_m).
    route: ordered arc indices of the reference route. Returns a copy with is_correct (0/1)."""
    route = np.atleast_1d(np.asarray(route, dtype=int))
    positions: dict[int, list[int]] = {}
    for i, a in enumerate(route):
        positions.setdefault(int(a), []).append(i)

    df = cands.reset_index(drop=True).copy()
    labels = np.zeros(len(df), dtype=int)
    layers: list[list[tuple[int, int, float, int | None]]] = []   # (row, route_pos, cost, back_ptr)

    for _, g in df.groupby(C.OBS_COL, sort=True):
        states = [(ridx, p, float(dist))
                  for ridx, arc, dist in zip(g.index, g[C.CAND_COL], g[C.DIST_COL])
                  for p in positions.get(int(arc), ())]
        if not states:
            continue
        layer = []
        for ridx, p, unary in states:
            if not layers:
                layer.append((ridx, p, unary, None))
                continue
            best_cost, best_k = np.inf, None
            for k, (_, prev_p, prev_cost, _) in enumerate(layers[-1]):
                step = p - prev_p
                trans = jump_cost * step if 0 <= step <= max_jump else restart_penalty
                c = prev_cost + trans + unary
                if c < best_cost:
                    best_cost, best_k = c, k
            layer.append((ridx, p, best_cost, best_k))
        layers.append(layer)

    if layers:
        k = int(np.argmin([s[2] for s in layers[-1]]))
        for layer in reversed(layers):
            ridx, _, _, back = layer[k]
            labels[ridx] = 1
            if back is None:
                break
            k = back
    df[C.LABEL_COL] = labels
    return df
