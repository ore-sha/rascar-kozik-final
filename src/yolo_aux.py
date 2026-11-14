"""ID29: дообучение YOLO11 (Ultralytics, AGPL-3.0, веса yolo11s.pt с github.com/ultralytics/assets) на 45 aux-кадрах той же камеры.
Классы: hand, cup, dish (tray+empty_plate+full_plate), phone, item (personal_items+headphones), person.
Отсутствие бокса в aux не надёжный негатив (TASK.md), поэтому пороги подбираем потом на train-днях.
Запуск: python src/yolo_aux.py prepare && python src/yolo_aux.py train"""
import sys, shutil, random
from pathlib import Path
import os
DEVICE = os.environ.get('YOLO_DEVICE', 'mps')
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
DS = ROOT / 'out' / ('yolo_aux_split' if os.environ.get('YOLO_SPLIT_TRAY') == '1' else 'yolo_aux')
SPLIT_TRAY = os.environ.get('YOLO_SPLIT_TRAY') == '1'   # ID30: поднос и тарелка разными классами
MAP = {'nude_hands': 0, 'cup': 1, 'tray': 2, 'empty_plate': 2, 'full_plate': 2, 'phone': 3,
       'personal_items': 4, 'headphones': 4, 'person': 5}
NAMES = ['hand', 'cup', 'dish', 'phone', 'item', 'person']
if SPLIT_TRAY:
    MAP = {**MAP, 'empty_plate': 6, 'full_plate': 6}
    NAMES = ['hand', 'cup', 'tray', 'phone', 'item', 'person', 'plate']


def prepare():
    a = pd.read_csv(RAW / 'aux_boxes.csv'); f = pd.read_csv(RAW / 'aux_frames.csv')
    ids = sorted(f.frame_id); random.Random(0).shuffle(ids)
    split = {i: ('val' if k < 5 else 'train') for k, i in enumerate(ids)}
    shutil.rmtree(DS, ignore_errors=True)
    for s in ['train', 'val']:
        (DS / 'images' / s).mkdir(parents=True); (DS / 'labels' / s).mkdir(parents=True)
    for fid in ids:
        s = split[fid]
        shutil.copy(RAW / f'{fid}.jpg', DS / 'images' / s / f'{fid}.jpg')
        g = a[(a.frame_id == fid) & a.class_name.isin(MAP)]
        (DS / 'labels' / s / f'{fid}.txt').write_text(''.join(
            f'{MAP[r.class_name]} {r.cx:.6f} {r.cy:.6f} {r.w:.6f} {r.h:.6f}\n' for r in g.itertuples()))
    (DS / 'data.yaml').write_text(f'path: {DS}\ntrain: images/train\nval: images/val\nnames: {NAMES}\n')
    print('prepared', len(ids), 'frames')


def train(size='s'):
    from ultralytics import YOLO
    m = YOLO(f'yolo11{size}.pt')
    m.train(data=str(DS / 'data.yaml'), imgsz=1280, epochs=80, batch=int(os.environ.get('YOLO_BATCH', 4)), device=DEVICE, workers=2,
            project=str(ROOT / 'out' / 'yolo_runs'), name=f'aux_{size}' + ('_split' if SPLIT_TRAY else ''), exist_ok=True, seed=0, plots=False,
            fliplr=0.5, mosaic=1.0, close_mosaic=10, patience=0)


if __name__ == '__main__':
    {'prepare': prepare, 'train': lambda: train(*(sys.argv[2:] or ['s']))}[sys.argv[1]]()
