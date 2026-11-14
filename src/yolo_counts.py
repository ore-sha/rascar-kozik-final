"""ID29: инференс дообученного на aux YOLO по целевым кадрам и проверка счётчиков на train-днях.
python src/yolo_counts.py detect train|test   -> out/yolo_<split>.csv (все боксы conf>=0.05)
python src/yolo_counts.py eval                -> сравнение с v7 по классам на обоих днях"""
import sys
from pathlib import Path
import os
DEVICE = os.environ.get('YOLO_DEVICE', 'mps')
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'exp' / 'people_ctx')]
from data import load, RAW

import os
SIZE = os.environ.get('YOLO_SIZE', 's')
SPL = '_split' if os.environ.get('YOLO_SPLIT_TRAY') == '1' else ''
W = Path(os.environ['YOLO_WEIGHTS']) if os.environ.get('YOLO_WEIGHTS') else ROOT / 'out' / 'yolo_runs' / f'aux_{SIZE}{SPL}' / 'weights' / 'best.pt'
TTA = os.environ.get('YOLO_TTA') == '1'
CLAHE = os.environ.get('YOLO_CLAHE') == '1'   # ID43: выравнивание яркости (пересвет у окна)
SUF = ('' if SIZE == 's' else f'_{SIZE}') + SPL + ('_tta' if TTA else '') + ('_clahe' if CLAHE else '')
NAMES = ['hand', 'cup', 'tray', 'phone', 'item', 'person', 'plate'] if SPL else ['hand', 'cup', 'dish', 'phone', 'item', 'person']


def load_img(path):
    if not CLAHE:
        return str(path)
    import cv2
    im = cv2.imread(str(path)); lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def detect(split):
    from ultralytics import YOLO
    d = load(); m = YOLO(str(W))
    fids = d['targets'].query('split == @split').frame_id.tolist()
    rows = []
    for i in range(0, len(fids), 8):
        chunk = fids[i:i + 8]
        res = m.predict([load_img(RAW / f'{f}.jpg') for f in chunk], imgsz=1280, conf=0.05, device=DEVICE, verbose=False, augment=TTA)
        for f, r in zip(chunk, res):
            for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                rows.append(dict(frame_id=f, cls=NAMES[int(c)], score=round(s, 4), x1=b[0], y1=b[1], x2=b[2], y2=b[3]))
        if i % 80 == 0:
            print(i, len(fids), flush=True)
    pd.DataFrame(rows).to_csv(os.environ.get('YOLO_OUT') or ROOT / 'out' / f'yolo_{split}{SUF}.csv', index=False)
    print('saved', len(rows))


PAD = (0.35, 0.6, 0.35, 0.25)   # как у OWLv2: слева, сверху, справа, снизу


def detect_crops(split, imgsz=960):
    """ID31: тот же YOLO, но по увеличенному кропу каждого стола. Бокс засчитывается своему столу,
    если центр внутри рамки стола (+10%). Пишет out/yolo_crops_<split>_<imgsz>.csv с row_id."""
    from ultralytics import YOLO
    from PIL import Image
    d = load(); m = YOLO(str(W))
    fids = set(d['targets'].query('split == @split').frame_id)
    tabs = d['tables'][d['tables'].frame_id.isin(fids)]
    rows = []
    for k, (fid, t) in enumerate(tabs.groupby('frame_id')):
        im = Image.open(RAW / f'{fid}.jpg').convert('RGB')
        crops, meta = [], []
        for r in t.itertuples():
            w, h = r.x2 - r.x1, r.y2 - r.y1
            b = (max(0, r.x1 - PAD[0] * w), max(0, r.y1 - PAD[1] * h), min(1280, r.x2 + PAD[2] * w), min(720, r.y2 + PAD[3] * h))
            crops.append(im.crop(tuple(map(int, b)))); meta.append((r, b))
        for (r, b), res in zip(meta, m.predict(crops, imgsz=imgsz, conf=0.05, device=DEVICE, verbose=False)):
            for bb, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                x1, y1, x2, y2 = bb[0] + int(b[0]), bb[1] + int(b[1]), bb[2] + int(b[0]), bb[3] + int(b[1])
                cx, cy, tw, th = (x1 + x2) / 2, (y1 + y2) / 2, r.x2 - r.x1, r.y2 - r.y1
                if r.x1 - .1 * tw <= cx <= r.x2 + .1 * tw and r.y1 - .1 * th <= cy <= r.y2 + .1 * th:
                    rows.append(dict(row_id=r.row_id, frame_id=fid, cls=NAMES[int(c)], score=round(s, 4), x1=x1, y1=y1, x2=x2, y2=y2))
        if k % 100 == 0:
            print(k, flush=True)
    out = ROOT / 'out' / f'yolo_crops_{split}_{imgsz}{SUF}.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out, len(rows))


def crop_counts(boxes, ids, thr: dict):
    b = boxes[boxes.score >= boxes.cls.map(thr).fillna(9)]
    return b.groupby(['row_id', 'cls']).size().unstack(fill_value=0).reindex(index=ids, columns=list(thr), fill_value=0)


def table_counts(boxes, tables, thr: dict, margin=0.1):
    """Предмет относится к столу, если центр бокса внутри рамки стола (с запасом margin); при нескольких - ближайший центр."""
    out = pd.DataFrame(0.0, index=pd.Index(tables.row_id, name='row_id'), columns=list(thr))
    b = boxes[boxes.score >= boxes.cls.map(thr).fillna(9)]
    tg = dict(tuple(tables.groupby('frame_id')))
    for fid, g in b.groupby('frame_id'):
        t = tg.get(fid)
        if t is None:
            continue
        tb = t[['x1', 'y1', 'x2', 'y2']].to_numpy(); tw, th = tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]
        cx, cy = ((g.x1 + g.x2) / 2).to_numpy(), ((g.y1 + g.y2) / 2).to_numpy()
        inside = (cx[:, None] >= tb[None, :, 0] - margin * tw) & (cx[:, None] <= tb[None, :, 2] + margin * tw) & \
                 (cy[:, None] >= tb[None, :, 1] - margin * th) & (cy[:, None] <= tb[None, :, 3] + margin * th)
        dist = (cx[:, None] - (tb[None, :, 0] + tb[None, :, 2]) / 2) ** 2 + (cy[:, None] - (tb[None, :, 1] + tb[None, :, 3]) / 2) ** 2
        dist = np.where(inside, dist, np.inf)
        own = np.where(inside.any(1), dist.argmin(1), -1)
        for k, c in enumerate(g.cls.to_numpy()):
            if own[k] >= 0 and c in thr:
                out.loc[t.row_id.iloc[own[k]], c] += 1
    return out


def evaluate():
    from metric import breakdown
    from combo import make, phones
    d = load(); L = d['train_labels']
    base = make(d, 'out/owl_train.csv', L.row_id, 'phones').set_index('row_id')
    base['dishes'] *= 2
    base['items'] = phones(d, 'out/owl_train.csv', L.row_id, queries=['a mobile phone', 'a wallet'])
    y = pd.read_csv(ROOT / 'out' / f'yolo_train{SUF}.csv')
    tabs = d['tables'][d['tables'].row_id.isin(L.row_id)]
    occ = (base.people > 0).to_numpy()

    def ev(p):
        r = [breakdown(L[L.date_utc == day], p.reset_index()) for day in ['2026-10-26', '2026-10-25']]
        return np.mean([x['score'] for x in r]), r
    m, r = ev(base)
    print(f'v7 avg={m:.3f} ({r[0]["score"]:.3f}/{r[1]["score"]:.3f})')
    for col, classes in [('hands', ['hand']), ('cups', ['cup']), ('dishes', ['dish']), ('items', ['phone', 'item'])]:
        for t in [0.15, 0.25, 0.35, 0.5]:
            c = table_counts(y, tabs, {k: t for k in classes}).loc[base.index].sum(1).to_numpy()
            for mode in ['repl', 'max', 'repl@occ']:
                p = base.copy()
                if mode == 'repl':
                    p[col] = c
                elif mode == 'max':
                    p[col] = np.maximum(p[col], c)
                else:
                    p[col] = np.where(occ, c, p[col])
                m, r = ev(p)
                print(f'  {col:6s} thr={t} {mode:8s} avg={m:.3f} ({r[0]["score"]:.3f}/{r[1]["score"]:.3f}) '
                      f'{col}={r[0][col]:.2f}/{r[1][col]:.2f} pred={c.sum():.0f} true={L[col].sum()}')


if __name__ == '__main__':
    {'detect': lambda: detect(sys.argv[2]), 'crops': lambda: detect_crops(sys.argv[2], int(sys.argv[3])),
     'eval': evaluate}[sys.argv[1]]()
