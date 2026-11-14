"""Загрузка таблиц и привязка детекций к столам."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / 'data' / 'raw'


def load(raw: Path = RAW) -> dict:
    d = {n: pd.read_csv(raw / f'{n}.csv') for n in
         ['frames', 'targets', 'tables', 'detections', 'train_labels', 'sample_submission']}
    d['targets']['context_frames'] = d['targets']['context_frames'].fillna('')
    d['frames']['ts'] = pd.to_datetime(d['frames'].timestamp)
    return d


def assign(boxes: np.ndarray, is_person: np.ndarray, tables: np.ndarray,
           person_thr=.15, obj_thr=.5) -> np.ndarray:
    """Для каждого бокса - индекс стола или -1. Логика как в бейзлайне организаторов:
    люди - по доле площади стола под боксом, предметы - по доле площади предмета внутри стола."""
    if len(boxes) == 0 or len(tables) == 0:
        return np.full(len(boxes), -1)
    inter = np.maximum(0, np.minimum(boxes[:, None, 2:], tables[None, :, 2:]) -
                       np.maximum(boxes[:, None, :2], tables[None, :, :2])).prod(-1)
    area = np.maximum((boxes[:, 2:] - boxes[:, :2]).prod(1), 1)
    tarea = np.maximum((tables[:, 2:] - tables[:, :2]).prod(1), 1)
    share = inter / area[:, None]
    share[is_person] = inter[is_person] / tarea[None, :]
    owner = share.argmax(1)
    ok = share.max(1) >= np.where(is_person, person_thr, obj_thr)
    return np.where(ok, owner, -1)
