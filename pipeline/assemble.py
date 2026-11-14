"""Финальная сборка сабмитов команды kozik из кэшей детекций (кейс 2, Table State).

    python pipeline/assemble.py --cache CACHE_DIR --out OUT_DIR

CACHE_DIR должен содержать (их создаёт pipeline/run.py):
    owl_test.csv          OWLv2-base по кропам столов (src/owl.py)
    yolo_test.csv         YOLO11s, дообученный на aux (src/yolo_aux.py, веса yolo_aux_s.pt)
    yolo_test_tta.csv     тот же YOLO11s с TTA (augment=True)
    yolo_weak_ALL.csv     YOLO11s со слабой разметкой из train-счётчиков (src/yolo_weak.py, веса yolo_weak_ALL.pt)
    plates_test.csv       тарелки D-FINE Objects365 (exp/det_closed/detect.py + plates.py)

Сборка (каждый шаг проверен на train по дням, см. таблицу гипотез):
    люди, руки, чашки, подносы, occ_prob  - прод-детекции с привязкой людей ID5 (exp/people_ctx, P(walk=2.0))
    items                                 - телефоны + кошельки OWLv2, score >= 0.25 (ID2, ID25)
    cups  (v11+)                          - max(прод, YOLO weak cup >= 0.4) на занятых столах (ID38)
    dishes v12                            - max(подносы x2, медиана 3 YOLO dish >= 0.4) (ID21, ID41)
    dishes v13                            - max(dishes v12, подносы + тарелки D-FINE) (ID26)
    cups  v14 (если есть sam3_test.csv)   - max(cups v13, чашки SAM 3 score >= 0.6) на занятых столах (ID46)
    hands v16 (поверх v14)                - на занятом столе без рук: руки прод с рамкой стола, расширенной вверх на 0.2 высоты (ID73)
    hands v17 (поверх v16)                - если рук всё ещё нет: руки SAM 3 score >= 0.6 в рамке стола (ID76)
Пишет OUT_DIR/submission_v12.csv, submission_v13.csv (основной) и submission_v14.csv (если есть кэш SAM 3)."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'exp' / 'people_ctx')]
from data import load  # noqa: E402
from people import P, build, predict  # noqa: E402
from owl_counts import prepare, counts  # noqa: E402
from yolo_counts import table_counts  # noqa: E402

COLS = ['row_id', 'people', 'hands', 'cups', 'dishes', 'items', 'occ_prob']


def sam3_count(path, d, ids, cls, thr):
    """Экземпляр SAM 3 засчитывается столу своего кропа, если центр бокса внутри рамки стола (+10%)."""
    s = pd.read_csv(path)
    s = s[(s.cls == cls) & (s.score >= thr)]
    s = s.join(d['tables'].set_index('row_id')[['x1', 'y1', 'x2', 'y2']].add_prefix('t'), on='row_id')
    tw, th = s.tx2 - s.tx1, s.ty2 - s.ty1
    cx, cy = (s.x1 + s.x2) / 2, (s.y1 + s.y2) / 2
    s = s[(cx >= s.tx1 - .1 * tw) & (cx <= s.tx2 + .1 * tw) & (cy >= s.ty1 - .1 * th) & (cy <= s.ty2 + .1 * th)]
    return s.groupby('row_id').size().reindex(ids).fillna(0).to_numpy()


def with_sam3_cups(v13, path, d, ids, occ, thr=0.6):
    """ID46: чашка SAM 3 засчитывается столу своего кропа, если центр бокса внутри рамки стола (+10%)."""
    s = pd.read_csv(path)
    s = s[(s.cls == 'cup') & (s.score >= thr)]
    s = s.join(d['tables'].set_index('row_id')[['x1', 'y1', 'x2', 'y2']].add_prefix('t'), on='row_id')
    tw, th = s.tx2 - s.tx1, s.ty2 - s.ty1
    cx, cy = (s.x1 + s.x2) / 2, (s.y1 + s.y2) / 2
    s = s[(cx >= s.tx1 - .1 * tw) & (cx <= s.tx2 + .1 * tw) & (cy >= s.ty1 - .1 * th) & (cy <= s.ty2 + .1 * th)]
    n = s.groupby('row_id').size().reindex(ids).fillna(0).to_numpy()
    v = v13.copy()
    v['cups'] = np.where(occ, np.maximum(v13.cups.to_numpy(), n), v13.cups.to_numpy())
    return v


def assemble(cache: Path):
    d = load()
    ids = d['sample_submission'].row_id
    tabs = d['tables'][d['tables'].row_id.isin(ids)]
    base = predict(build(d, P(walk=2.0)), {}, 0.0).set_index('row_id').loc[ids]
    trays = base.dishes.to_numpy().copy()
    occ = (base.people > 0).to_numpy()
    cand = prepare(pd.read_csv(cache / 'owl_test.csv'), d['tables'])
    base['items'] = counts(cand, ids, {'a mobile phone': .25, 'a wallet': .25}, ['items'])['items'].to_numpy()
    weak = table_counts(pd.read_csv(cache / 'yolo_weak_ALL.csv'), tabs, {'cup': .4}).loc[ids, 'cup'].to_numpy()
    base['cups'] = np.where(occ, np.maximum(base.cups.to_numpy(), weak), base.cups.to_numpy())
    med = np.median(np.stack([table_counts(pd.read_csv(cache / f), tabs, {'dish': .4}).loc[ids, 'dish'].to_numpy()
                              for f in ['yolo_test.csv', 'yolo_test_tta.csv', 'yolo_weak_ALL.csv']]), 0)
    v12 = base.copy()
    v12['dishes'] = np.maximum(trays * 2, med)
    plates = pd.read_csv(cache / 'plates_test.csv').set_index('row_id').plates.reindex(ids).fillna(0).to_numpy()
    v13 = v12.copy()
    v13['dishes'] = np.maximum(v12.dishes.to_numpy(), trays + plates)
    vs = [('v12', v12), ('v13', v13)]
    if (cache / 'sam3_test.csv').exists():
        v14 = with_sam3_cups(v13, cache / 'sam3_test.csv', d, ids, occ)
        vs.append(('v14', v14))
        hu = predict(build(d, P(walk=2.0, obj_up=0.2)), {}, 0.0).set_index('row_id').loc[ids].hands.to_numpy()
        v16 = v14.copy()
        v16['hands'] = np.where(occ & (v14.hands.to_numpy() == 0), hu, v14.hands.to_numpy())
        vs.append(('v16', v16))
        v17 = v16.copy()   # ID76: занятый стол всё ещё без рук -> руки SAM 3 >= 0.6 в рамке стола
        v17['hands'] = np.where(occ & (v16.hands.to_numpy() == 0), sam3_count(cache / 'sam3_test.csv', d, ids, 'hand', .6), v16.hands.to_numpy())
        vs.append(('v17', v17))
    out = {}
    for name, v in vs:
        s = v.reset_index()[COLS]
        assert len(s) == len(ids) and s.row_id.is_unique and not s.isna().any().any()
        assert (s[COLS[1:6]] >= 0).all().all() and s[COLS[1:6]].max().max() <= 50 and s.occ_prob.between(0, 1).all()
        out[name] = s
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name, s in assemble(a.cache).items():
        s.to_csv(a.out / f'submission_{name}.csv', index=False)
        print('saved', a.out / f'submission_{name}.csv', len(s), s[COLS[1:6]].sum().round(0).to_dict())
