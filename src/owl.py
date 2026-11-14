"""OWLv2 (google/owlv2-base-patch16-ensemble, Apache 2.0) по расширенным кропам столов.
Для каждой строки стола сохраняет все боксы со score >= MIN_SCORE в координатах кадра.
Запуск: python src/owl.py --split train|test [--frames context] --out out/owl_train.csv
"""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from data import load, RAW

import os
MODEL = os.environ.get('OWL_MODEL', 'google/owlv2-base-patch16-ensemble')
QUERIES = ['a plate', 'a bowl', 'a food tray', 'a cup', 'a drinking glass', 'a paper coffee cup',
           'a mobile phone', 'a handbag', 'a backpack', 'a wallet', 'headphones', 'a sheet of paper',
           'a human hand', 'a person', 'a bottle', 'a napkin',
           # отрицательные запросы: по условию не считаются, нужны чтобы гасить ложные срабатывания
           'a laptop', 'a vase with flowers', 'a computer keyboard', 'a tablet computer', 'a hand on a table',
           'a menu', 'a spoon', 'a cushion']
MIN_SCORE = 0.05
COLS = ['row_id', 'frame_id', 'img', 'key', 'query', 'score', 'x1', 'y1', 'x2', 'y2', 'cx1', 'cy1', 'cx2', 'cy2']
PAD = (0.35, 0.6, 0.35, 0.25)   # расширение рамки стола: слева, сверху, справа, снизу (доли ширины/высоты)


def crop_box(r, W=1280, H=720):
    w, h = r.x2 - r.x1, r.y2 - r.y1
    return (max(0, r.x1 - PAD[0] * w), max(0, r.y1 - PAD[1] * h),
            min(W, r.x2 + PAD[2] * w), min(H, r.y2 + PAD[3] * h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--context', action='store_true', help='рамки целевого кадра на соседних кадрах ±30 с')
    a = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    rev = os.environ.get('OWL_REV')   # pipeline фиксирует commit весов
    proc = Owlv2Processor.from_pretrained(MODEL, revision=rev)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL, revision=rev).to(dev).eval()
    text = proc(text=[QUERIES], return_tensors='pt').to(dev)
    d = load()
    fids = d['targets'].query('split == @a.split').frame_id
    tabs = d['tables'][d['tables'].frame_id.isin(fids)].copy()
    tabs['img'] = tabs.frame_id
    if a.context:
        # камера неподвижна: рамки целевого кадра применяем к его соседям
        ctx = d['targets'].set_index('frame_id').context_frames
        parts = []
        for fid, t in tabs.groupby('frame_id'):
            for n in [x for x in ctx.get(fid, '').split(';') if x]:
                parts.append(t.assign(img=n))
        tabs = pd.concat(parts)
    tabs['key'] = tabs.frame_id + '|' + tabs.img
    if a.limit:
        tabs = tabs.head(a.limit)
    out_p = Path(a.out); out_p.parent.mkdir(exist_ok=True, parents=True)
    part = out_p.with_suffix('.part.csv')
    done = set()
    if part.exists() and part.stat().st_size > 0:
        old = pd.read_csv(part)
        if 'key' not in old:   # файл старого формата (без контекста): key = frame|frame
            old.insert(2, 'img', old.frame_id); old.insert(3, 'key', old.frame_id + '|' + old.frame_id)
            old.to_csv(part, index=False)
        done = set(old.key)
    rows, t0 = [], time.time()
    groups = [(k, t) for k, t in tabs.groupby('key') if k not in done]
    print(f'resume: {len(done)} frames done, {len(groups)} left', flush=True)
    for i, (key, tab) in enumerate(groups):
        fid, img = key.split('|')
        im = Image.open(RAW / f'{img}.jpg').convert('RGB')
        crops, boxes = [], []
        for r in tab.itertuples():
            b = crop_box(r)
            crops.append(im.crop(tuple(map(int, b))))
            boxes.append((r.row_id, b))
        inp = proc(images=crops, return_tensors='pt').to(dev)
        with torch.no_grad():
            out = model(input_ids=text['input_ids'].repeat(len(crops), 1),
                        attention_mask=text['attention_mask'].repeat(len(crops), 1),
                        pixel_values=inp['pixel_values'])
        # OWLv2 дополняет картинку до квадрата, поэтому целевой размер - max(w, h)
        sizes = torch.tensor([[max(c.size)] * 2 for c in crops])
        res = proc.post_process_grounded_object_detection(out, threshold=MIN_SCORE, target_sizes=sizes)
        for (row_id, b), rr in zip(boxes, res):
            for s, l, bb in zip(rr['scores'].tolist(), rr['labels'].tolist(), rr['boxes'].tolist()):
                rows.append(dict(row_id=row_id, frame_id=fid, img=img, key=key, query=QUERIES[l], score=round(s, 4),
                                 x1=bb[0] + b[0], y1=bb[1] + b[1], x2=bb[2] + b[0], y2=bb[3] + b[1],
                                 cx1=b[0], cy1=b[1], cx2=b[2], cy2=b[3]))
        if (i + 1) % 25 == 0 or i == len(groups) - 1:
            # дописываем кусок, чтобы после падения/зависания продолжить с этого места
            pd.DataFrame(rows, columns=COLS).to_csv(part, mode='a', header=not part.exists() or part.stat().st_size == 0, index=False)
            rows = []
            print(f'{len(done) + i + 1} frames, {time.time() - t0:.0f}s', flush=True)
    part.rename(out_p)
    print('saved', a.out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
