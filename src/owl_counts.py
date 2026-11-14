"""Счётчики из боксов OWLv2: запросы -> классы, NMS внутри класса, центр бокса в рамке стола,
порог score на каждый запрос (координатный спуск по L1 на train одного дня)."""
import numpy as np
import pandas as pd

GROUPS = {
    'dishes': ['a plate', 'a bowl', 'a food tray'],
    'cups': ['a cup', 'a drinking glass', 'a paper coffee cup'],
    'items': ['a mobile phone', 'a handbag', 'a wallet', 'headphones', 'a sheet of paper'],
    'hands': ['a human hand', 'a hand on a table'],
}
GRID = np.r_[np.round(np.arange(0.06, 0.6, 0.02), 2), 1.01]   # 1.01 = запрос выключен


def iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def nms(g, thr=0.5):
    g = g.sort_values('score', ascending=False)
    b = g[['x1', 'y1', 'x2', 'y2']].to_numpy()
    keep, alive = [], np.ones(len(b), bool)
    for i in range(len(b)):
        if alive[i]:
            keep.append(i)
            alive &= iou(b[i], b) < thr
    return g.iloc[keep]


def prepare(owl, tables, margin=0.1, max_area=0.5):
    q2c = {q: c for c, qs in GROUPS.items() for q in qs}
    o = owl[owl['query'].isin(q2c)].copy()
    o['cls'] = o['query'].map(q2c)
    o = o.merge(tables[['row_id', 'x1', 'y1', 'x2', 'y2']].rename(
        columns=lambda c: 't' + c if c != 'row_id' else c), on='row_id')
    tw, th = o.tx2 - o.tx1, o.ty2 - o.ty1
    cx, cy = (o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2
    inside = (cx >= o.tx1 - margin * tw) & (cx <= o.tx2 + margin * tw) & \
             (cy >= o.ty1 - margin * th) & (cy <= o.ty2 + margin * th)
    small = (o.x2 - o.x1) * (o.y2 - o.y1) <= max_area * tw * th
    o = o[inside & small]
    return pd.concat([nms(g) for _, g in o.groupby(['row_id', 'cls'])]) if len(o) else o


def counts(cand, row_ids, qthr: dict, cls_list=GROUPS):
    """qthr: порог на каждый запрос."""
    t = cand['query'].map(qthr).fillna(9)
    keep = cand[cand.score >= t]
    out = pd.DataFrame(index=pd.Index(row_ids, name='row_id'))
    for c in cls_list:
        s = keep[keep.cls == c].groupby('row_id').size()
        out[c] = s.reindex(out.index).fillna(0).to_numpy()
    return out


def fit_qthr(cand, labels, cls, rounds=3):
    """Координатный спуск: по очереди подбираем порог каждого запроса класса, минимизируя sum|y - pred|."""
    y = labels.set_index('row_id')[cls]
    qs = GROUPS[cls]
    thr = {q: 1.01 for q in qs}
    err = lambda th: np.abs(y - counts(cand, y.index, th, [cls])[cls]).sum()
    best = err(thr)
    for _ in range(rounds):
        for q in qs:
            for t in GRID:
                trial = {**thr, q: t}
                e = err(trial)
                if e < best - 1e-9:
                    best, thr = e, trial
    return thr, best
