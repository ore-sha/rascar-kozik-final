"""Метрика кейса 2 по формуле из TASK.md: 0.2*AP(occupied) + 0.8*mean_k max(0, 1-WAPE_k)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

COUNTS = ['people', 'hands', 'cups', 'dishes', 'items']


def breakdown(labels: pd.DataFrame, pred: pd.DataFrame) -> dict:
    pred = pred.set_index('row_id').loc[labels.row_id]
    q = {}
    for c in COUNTS:
        y, p = labels[c].to_numpy(float), pred[c].to_numpy(float)
        denom = y.sum() if y.sum() > 0 else len(y)
        q[c] = max(0.0, 1 - np.abs(y - p).sum() / denom)
    occ = (labels.people > 0).to_numpy(int)
    prob = pred.occ_prob.to_numpy(float)
    if occ.sum() == 0:
        ap = 1 - prob.mean()
    elif occ.sum() == len(occ):
        ap = 1.0
    else:
        ap = average_precision_score(occ, prob)
    return dict(score=0.2 * ap + 0.8 * np.mean(list(q.values())), ap=ap, **q)


def fmt(r: dict) -> str:
    return '  '.join(f'{k}={v:.3f}' for k, v in r.items())
