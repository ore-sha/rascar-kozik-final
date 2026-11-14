"""Комбинация: привязка людей Влада (exp/people_ctx) + items из OWLv2-телефонов.
Варианты items сравниваются на обоих днях."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'exp' / 'people_ctx')]
from data import load
from metric import breakdown
from people import P, build, predict
from owl_counts import prepare, counts

PHONE_THR = 0.25
VLAD = P(walk=2.0)


ITEM_QUERIES = ['a mobile phone']   # 'best2' добавляет 'a wallet' (ID25 Влада)


def phones(d, owl_csv, ids, thr=PHONE_THR, queries=None):
    cand = prepare(pd.read_csv(owl_csv), d['tables'])
    return counts(cand, ids, {q: thr for q in (queries or ITEM_QUERIES)}, ['items'])['items'].to_numpy()


CUP_THR = 0.25
DISH_MULT = 2.0   # прод видит только подносы; поднос с тарелками = несколько предметов (ID21)


def extra_cups(d, owl_csv, ids, thr=CUP_THR):
    """Чашки OWLv2, не совпавшие (IoU<0.3) с прод-чашками того же кадра."""
    from owl_counts import iou
    cand = prepare(pd.read_csv(owl_csv), d['tables'])
    cand = cand[(cand.cls == 'cups') & (cand.score >= thr)]
    cand = cand.merge(d['tables'][['row_id', 'frame_id']], on='row_id', suffixes=('_o', ''))
    dg = dict(tuple(d['detections'][d['detections'].class_name == 'cup'].groupby('frame_id')))
    def unmatched(r):
        g = dg.get(r.frame_id)
        return g is None or iou(np.array([r.x1, r.y1, r.x2, r.y2]), g[['x1', 'y1', 'x2', 'y2']].to_numpy()).max() < 0.3
    cand = cand[[unmatched(r) for r in cand.itertuples()]]
    return cand.groupby('row_id').size().reindex(ids).fillna(0).to_numpy()


def make(d, owl_csv, ids, mode='max1', cups_union=False):
    p = predict(build(d, VLAD), {}, 0.0).set_index('row_id').loc[ids]
    ph = phones(d, owl_csv, ids)
    occ = (p.people > 0).to_numpy()
    if cups_union:   # пропуски чашек добираем OWLv2, только на занятых столах
        p['cups'] = p['cups'].to_numpy() + extra_cups(d, owl_csv, ids) * occ
    if mode == 'phones':
        p['items'] = ph
    elif mode == 'occ1':
        p['items'] = occ * 1.0
    elif mode == 'max1':
        p['items'] = np.where(occ, np.maximum(ph, 1), ph)
    elif mode == 'occ_phones':
        p['items'] = np.where(occ, ph, 0)
    return p.reset_index()


if __name__ == '__main__':
    d = load(); L = d['train_labels']
    if len(sys.argv) > 2:   # python src/combo.py OUT.csv phones|phones+cups|best
        ids = d['sample_submission'].row_id
        sub = make(d, 'out/owl_test.csv', ids, 'phones', cups_union=sys.argv[2] == 'phones+cups')
        if sys.argv[2] in ('best', 'best2', 'best3'):
            sub['dishes'] = sub['dishes'] * DISH_MULT
        if sys.argv[2] in ('best2', 'best3', 'best4'):
            sub['items'] = phones(d, 'out/owl_test.csv', ids, queries=['a mobile phone', 'a wallet'])
        if sys.argv[2] == 'best3':   # ID29: посуда = max(прод x2, YOLO на aux >= 0.4); нужен out/yolo_test.csv
            from yolo_counts import table_counts
            tabs = d['tables'][d['tables'].row_id.isin(ids)]
            yd = table_counts(pd.read_csv('out/yolo_test.csv'), tabs, {'dish': 0.4}).loc[ids, 'dish'].to_numpy()
            sub['dishes'] = np.maximum(sub['dishes'].to_numpy(), yd)
        if sys.argv[2] == 'best4':   # ID38: v8 + чашки = max(v8, YOLO со слабой разметкой >= 0.4) на занятых; нужен out/yolo_weak_ALL.csv
            from yolo_counts import table_counts
            tabs = d['tables'][d['tables'].row_id.isin(ids)]
            sub['dishes'] = np.maximum(sub['dishes'].to_numpy() * DISH_MULT,
                                       table_counts(pd.read_csv('out/yolo_test.csv'), tabs, {'dish': 0.4}).loc[ids, 'dish'].to_numpy())
            wc = table_counts(pd.read_csv('out/yolo_weak_ALL.csv'), tabs, {'cup': 0.4}).loc[ids, 'cup'].to_numpy()
            sub['cups'] = np.where(sub['people'] > 0, np.maximum(sub['cups'].to_numpy(), wc), sub['cups'].to_numpy())
        assert len(sub) == len(ids) and sub.row_id.is_unique and not sub.isna().any().any()
        sub[['row_id', 'people', 'hands', 'cups', 'dishes', 'items', 'occ_prob']].to_csv(sys.argv[1], index=False)
        print('saved', sys.argv[1], len(sub)); sys.exit()
    for mode, cu in [('occ1', False), ('phones', False), ('max1', False), ('phones', True)]:
        p = make(d, 'out/owl_train.csv', L.row_id, mode, cu)
        r = [breakdown(L[L.date_utc == day], p) for day in ['2026-10-26', '2026-10-25']]
        m = {k: np.mean([x[k] for x in r]) for k in r[0]}
        print(f'{mode + ("+cups" if cu else ""):11s} avg={m["score"]:.3f} (26: {r[0]["score"]:.3f} / 25: {r[1]["score"]:.3f})  '
              + ' '.join(f'{k}={m[k]:.2f}' for k in ['ap', 'people', 'hands', 'cups', 'dishes', 'items']))
