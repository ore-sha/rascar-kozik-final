"""ID38: YOLO11s на aux + train-кадрах со слабой разметкой из ручных счётчиков train.
Для стола и класса (hand/cup/dish) с истинным счётчиком k>0 берём k лучших боксов YOLO-s (ID29, score>=0.1);
если боксов меньше k - закрашиваем расширенную рамку стола (не учим на неизвестном). k=0 - все боксы класса
на столе выкидываем (стол становится негативом). Классы phone/item/person - боксы YOLO-s со score>=0.5.
Фолды: A = псевдоразметка 25.10 (проверка на 26.10), B = 26.10 (проверка на 25.10), ALL = оба дня (для test).
python src/yolo_weak.py spec -> out/yolo_weak_spec.json (используется ядром Kaggle rascar-yolo-weak)"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from data import load

NAMES = ['hand', 'cup', 'dish', 'phone', 'item', 'person']
CNT = {'hand': 'hands', 'cup': 'cups', 'dish': 'dishes'}
PAD = (0.35, 0.6, 0.35, 0.25)


def spec():
    d = load(); L = d['train_labels'].set_index('row_id')
    det = pd.read_csv(ROOT / 'out' / 'yolo_train.csv')
    tabs = d['tables'][d['tables'].row_id.isin(L.index)]
    out = {}
    for fid, t in tabs.groupby('frame_id'):
        g = det[det.frame_id == fid].copy()
        tb = t[['x1', 'y1', 'x2', 'y2']].to_numpy(); tw, th = tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]
        cx, cy = ((g.x1 + g.x2) / 2).to_numpy(), ((g.y1 + g.y2) / 2).to_numpy()
        inside = (cx[:, None] >= tb[None, :, 0] - .1 * tw) & (cx[:, None] <= tb[None, :, 2] + .1 * tw) & \
                 (cy[:, None] >= tb[None, :, 1] - .1 * th) & (cy[:, None] <= tb[None, :, 3] + .1 * th)
        dist = np.where(inside, (cx[:, None] - (tb[None, :, 0] + tb[None, :, 2]) / 2) ** 2 +
                        (cy[:, None] - (tb[None, :, 1] + tb[None, :, 3]) / 2) ** 2, np.inf)
        g['tab'] = np.where(inside.any(1), dist.argmin(1), -1)
        labels, masks = [], []
        for j, r in enumerate(t.itertuples()):
            for cls, col in CNT.items():
                k = int(L.loc[r.row_id, col])
                cand = g[(g.tab == j) & (g.cls == cls) & (g.score >= 0.1)].sort_values('score', ascending=False)
                if k == 0:
                    continue
                if len(cand) >= k:
                    labels += [(cls, *b) for b in cand.head(k)[['x1', 'y1', 'x2', 'y2']].to_numpy().tolist()]
                else:
                    w, h = r.x2 - r.x1, r.y2 - r.y1
                    masks.append([max(0, r.x1 - PAD[0] * w), max(0, r.y1 - PAD[1] * h),
                                  min(1280, r.x2 + PAD[2] * w), min(720, r.y2 + PAD[3] * h)])
        conf = g[g.cls.isin(['phone', 'item', 'person']) & (g.score >= 0.5)]
        labels += [(c, *b) for c, b in zip(conf.cls, conf[['x1', 'y1', 'x2', 'y2']].to_numpy().tolist())]
        day = str(L[L.frame_id == fid].date_utc.iloc[0])
        out[fid] = dict(day=day, labels=labels, masks=masks)
    (ROOT / 'out' / 'yolo_weak_spec.json').write_text(json.dumps(out))
    n_lab = sum(len(v['labels']) for v in out.values()); n_mask = sum(len(v['masks']) for v in out.values())
    print('frames', len(out), 'labels', n_lab, 'masked tables', n_mask)


if __name__ == '__main__':
    spec()
