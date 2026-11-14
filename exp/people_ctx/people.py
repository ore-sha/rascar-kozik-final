"""Гипотезы №4 и №5: привязка людей к столам и сглаживание по соседним кадрам.

Привязка людей (№5). В бейзлайне человек относится к столу, если его бокс накрывает >=15% площади стола.
Так теряются люди за дальней стороной (сидят у стены, бокс почти целиком выше столешницы)
и засчитываются прохожие (бокс стоящего в проходе накрывает стол, ноги далеко ниже него).
Здесь для пары человек-стол считаем:
  * xov  - доля горизонтального перекрытия (от меньшей из ширин);
  * cov  - доля бокса человека внутри расширенной рамки стола (вверх на up*h, вниз на down*h);
  * feet - насколько низ бокса ниже низа стола, в высотах стола (прохожий, если > walk).
Человек уходит к столу с максимальным xov*cov среди подходящих.
Опционально: если у стола нет людей, но есть >= hands_min рук, считаем одного человека
(руки на столе = кто-то сидит, часто за дальней стороной, где детектор людей промахивается).

Соседние кадры (№4). Камера неподвижна, рамки столов целевого кадра применяем к детекциям соседей
(±30 с). Агрегат по кадру и соседям: медиана или среднее.
"""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd

OBJ = {'nude_hands': 'hands', 'cup': 'cups', 'tray': 'dishes'}


@dataclass(frozen=True)
class P:
    conf: float = 0.0     # порог confidence для людей
    up: float = 0.9       # расширение рамки стола вверх, в высотах стола
    down: float = 0.6     # вниз
    side: float = 0.1     # по бокам, в ширинах стола
    min_xov: float = 0.3
    min_cov: float = 0.3
    walk: float = 1.6     # низ бокса ниже стола больше чем на walk*h -> прохожий
    hands_min: int = 0    # 0 = правило «руки => человек» выключено
    legacy: bool = False  # True = привязка как в бейзлайне
    obj_up: float = 0.0   # объекты: расширение рамки стола вверх (в высотах стола)
    obj_side: float = 0.0  # объекты: расширение по бокам (в ширинах стола)
    obj_thr: float = 0.5  # доля бокса объекта внутри рамки
    via_person: str = ''  # классы через запятую, которые привязываем через бокс человека


def _legacy(pb, tb, thr=.15):
    inter = np.maximum(0, np.minimum(pb[:, None, 2:], tb[None, :, 2:]) -
                       np.maximum(pb[:, None, :2], tb[None, :, :2])).prod(-1)
    tarea = np.maximum((tb[:, 2:] - tb[:, :2]).prod(1), 1)
    share = inter / tarea[None]
    return np.where(share.max(1) >= thr, share.argmax(1), -1)


def assign_people(pb: np.ndarray, tb: np.ndarray, p: P) -> np.ndarray:
    """pb (n,4), tb (m,4) -> индекс стола для каждого человека или -1."""
    if len(pb) == 0 or len(tb) == 0:
        return np.full(len(pb), -1)
    if p.legacy:
        return _legacy(pb, tb)
    tw, th = tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]
    ex = np.stack([tb[:, 0] - p.side * tw, tb[:, 1] - p.up * th,
                   tb[:, 2] + p.side * tw, tb[:, 3] + p.down * th], 1)
    pw, ph = pb[:, 2] - pb[:, 0], pb[:, 3] - pb[:, 1]
    ox = np.maximum(0, np.minimum(pb[:, None, 2], tb[None, :, 2]) - np.maximum(pb[:, None, 0], tb[None, :, 0]))
    xov = ox / np.maximum(np.minimum(pw[:, None], tw[None]), 1)
    inter = np.maximum(0, np.minimum(pb[:, None, 2:], ex[None, :, 2:]) -
                       np.maximum(pb[:, None, :2], ex[None, :, :2])).prod(-1)
    cov = inter / np.maximum(pw * ph, 1)[:, None]
    feet = (pb[:, None, 3] - tb[None, :, 3]) / np.maximum(th[None], 1)
    ok = (xov >= p.min_xov) & (cov >= p.min_cov) & (feet <= p.walk)
    score = np.where(ok, xov * cov, -1)
    return np.where(ok.any(1), score.argmax(1), -1)


def frame_counts(det: pd.DataFrame, tb: np.ndarray, p: P) -> np.ndarray:
    """Счётчики (m, 4): people, hands, cups, dishes для одного кадра."""
    out = np.zeros((len(tb), 4))
    if len(tb) == 0 or det.empty:
        return out
    pp = det[(det.class_name == 'person') & (det.confidence >= p.conf)]
    own = assign_people(pp[['x1', 'y1', 'x2', 'y2']].to_numpy(float), tb, p)
    np.add.at(out[:, 0], own[own >= 0], 1)
    ob = det[det.class_name.isin(OBJ)]
    if len(ob):
        b = ob[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
        tw, th = tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1]
        ex = np.stack([tb[:, 0] - p.obj_side * tw, tb[:, 1] - p.obj_up * th, tb[:, 2] + p.obj_side * tw, tb[:, 3]], 1)
        inter = np.maximum(0, np.minimum(b[:, None, 2:], ex[None, :, 2:]) -
                           np.maximum(b[:, None, :2], ex[None, :, :2])).prod(-1)
        share = inter / np.maximum((b[:, 2:] - b[:, :2]).prod(1), 1)[:, None]
        o = np.where(share.max(1) >= p.obj_thr, share.argmax(1), -1)
        if p.via_person and (own >= 0).any():
            # объект внутри бокса привязанного человека -> стол этого человека (если по рамке не привязан)
            pb = pp[['x1', 'y1', 'x2', 'y2']].to_numpy(float)[own >= 0]
            po = own[own >= 0]
            pi = np.maximum(0, np.minimum(b[:, None, 2:], pb[None, :, 2:]) -
                            np.maximum(b[:, None, :2], pb[None, :, :2])).prod(-1)
            psh = pi / np.maximum((b[:, 2:] - b[:, :2]).prod(1), 1)[:, None]
            cls_ok = ob.class_name.isin(p.via_person.split(',')).to_numpy()
            take = (o < 0) & cls_ok & (psh.max(1) >= .5)
            o = np.where(take, po[psh.argmax(1)], o)
        for k, cls in enumerate(OBJ, start=1):
            m = (o >= 0) & (ob.class_name.to_numpy() == cls)
            np.add.at(out[:, k], o[m], 1)
    if p.hands_min:
        out[:, 0] = np.where((out[:, 0] == 0) & (out[:, 1] >= p.hands_min), 1, out[:, 0])
    return out


def build(d: dict, p: P) -> pd.DataFrame:
    """Для каждой строки tables: счётчики на кадре и на каждом соседе."""
    det = dict(tuple(d['detections'].groupby('frame_id')))
    empty = d['detections'].iloc[:0]
    ctx = d['targets'].set_index('frame_id').context_frames
    rows = []
    for fid, tab in d['tables'].groupby('frame_id'):
        tab = tab.sort_values('table_id')
        tb = tab[['x1', 'y1', 'x2', 'y2']].to_numpy(float)
        cur = frame_counts(det.get(fid, empty), tb, p)
        nb = [n for n in ctx.get(fid, '').split(';') if n]
        ctxc = [frame_counts(det.get(n, empty), tb, p) for n in nb]
        for j, r in enumerate(tab.itertuples()):
            row = dict(row_id=r.row_id)
            for k, name in enumerate(['people', 'hands', 'cups', 'dishes']):
                row['cur_' + name] = cur[j, k]
                row['nb_' + name] = [c[j, k] for c in ctxc]
            rows.append(row)
    return pd.DataFrame(rows)


def predict(f: pd.DataFrame, agg: dict, items_per_occ: float = 0.0) -> pd.DataFrame:
    """agg: класс -> 'cur' | 'median' | 'mean' | 'max'.
    items_per_occ: столько вещей ставим каждому столу, где предсказан человек (медиана items
    по занятым столам train = 1). Прод-детектор вещи не ищет, это априорная оценка."""
    out = pd.DataFrame({'row_id': f.row_id})
    for name in ['people', 'hands', 'cups', 'dishes']:
        allv = [[c] + n for c, n in zip(f['cur_' + name], f['nb_' + name])]
        how = agg.get(name, 'cur')
        fn = {'cur': lambda v: v[0], 'median': np.median, 'mean': np.mean, 'max': np.max}[how]
        out[name] = [float(fn(v)) for v in allv]
    out['items'] = np.where(out.people > 0, items_per_occ, 0.0)
    pm = np.array([np.mean([c] + n) for c, n in zip(f.cur_people, f.nb_people)])
    hm = np.array([np.mean([c] + n) for c, n in zip(f.cur_hands, f.nb_hands)])
    out['occ_prob'] = np.clip(f.cur_people.to_numpy() + 0.5 * pm + 0.1 * hm, 0, 5) / 5
    return out
