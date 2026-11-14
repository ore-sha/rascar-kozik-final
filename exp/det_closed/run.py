"""ID 26: оценка детекторов Objects365/COCO на train, оба дня.

Для каждого класса (cups, dishes, items) перебираем модель, область (crop/frame), набор меток, порог и способ
объединения с текущим предсказанием:
  new   - только новый детектор;
  max   - максимум из текущего и нового;
  sum   - текущее + новое (для dishes: прод-подносы + тарелки и миски нового детектора).
Текущее: cups/dishes - прод-детектор с привязкой ID5 (для dishes ещё вариант x2 из лучшего сабмита),
items - телефоны OWLv2 >= 0.25 (если есть out/owl_train.csv).
Качество класса = 1 - WAPE на каждом дне. Честная оценка - кросс-выбор: конфиг выбран на одном дне, оценён на другом.
Запуск: python exp/det_closed/run.py   (после detect.py --split train)
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'exp/people_ctx')]
from data import load  # noqa: E402
from people import P, build, predict  # noqa: E402
from owl_counts import prepare, counts, nms  # noqa: E402

DAYS = ['2026-10-26', '2026-10-25']
# наборы меток (нижний регистр); метка модели подходит, если совпадает с одним из вариантов
SETS = {
    'cups': {'cup': ['cup'], 'cup+glass': ['cup', 'wine glass'], 'cup+glass+mug': ['cup', 'wine glass', 'mug', 'glass']},
    'dishes': {'plate': ['plate'], 'plate+bowl': ['plate', 'bowl', 'bowl/basin'],
               'plate+bowl+tray': ['plate', 'bowl', 'bowl/basin', 'tray'], 'bowl': ['bowl', 'bowl/basin']},
    'items': {'phone': ['cell phone'], 'phone+bag': ['cell phone', 'handbag', 'handbag/satchel', 'backpack'],
              'phone+bag+wallet': ['cell phone', 'handbag', 'handbag/satchel', 'backpack', 'wallet/purse', 'wallet'],
              'personal': ['cell phone', 'handbag', 'handbag/satchel', 'backpack', 'wallet/purse', 'wallet', 'book',
                           'head phone', 'earphone', 'glasses', 'key', 'pen/pencil', 'camera', 'briefcase']},
}
THRS = [.2, .3, .4, .5, .6]


def quality(y, p):
    return max(0.0, 1 - np.abs(y - p).sum() / max(y.sum(), 1))


def assign(c, tables):
    """Бокс относится к столу, если центр внутри рамки (+10%) и площадь <= половины стола.
    crop: только свой стол; frame: стол с ближайшим центром среди подходящих."""
    t = tables[['row_id', 'frame_id', 'x1', 'y1', 'x2', 'y2']].rename(
        columns={k: 't' + k for k in ['x1', 'y1', 'x2', 'y2']})
    a = c[c.scope == 'crop'].merge(t.drop(columns='frame_id'), on='row_id')
    b = c[c.scope == 'frame'].drop(columns='row_id').merge(t, on='frame_id')
    o = pd.concat([a, b], ignore_index=True)
    tw, th = o.tx2 - o.tx1, o.ty2 - o.ty1
    cx, cy = (o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2
    ok = (cx >= o.tx1 - .1 * tw) & (cx <= o.tx2 + .1 * tw) & (cy >= o.ty1 - .1 * th) & (cy <= o.ty2 + .1 * th) & \
         ((o.x2 - o.x1) * (o.y2 - o.y1) <= .5 * tw * th)
    o = o[ok].copy()
    o['dist'] = np.hypot(cx - (o.tx1 + o.tx2) / 2, cy - (o.ty1 + o.ty2) / 2)[ok]
    key = ['model', 'scope', 'frame_id', 'label', 'x1', 'y1', 'x2', 'y2']
    return o.sort_values('dist').drop_duplicates(key)


def main():
    d = load()
    L = d['train_labels'].set_index('row_id')
    idx = {day: L.index[L.date_utc == day] for day in DAYS}
    c = pd.read_csv(ROOT / 'out/det_closed/cands_train.csv', keep_default_na=False)
    c['label'] = c.label.str.lower()
    c = assign(c, d['tables'][d['tables'].row_id.isin(L.index)])
    print('боксы на столах (score>=0.3), метки:')
    top = c[c.score >= .3].groupby(['model', 'scope']).label.value_counts().groupby(level=[0, 1]).head(25)
    for (m, s), g in top.groupby(level=[0, 1]):
        print(f'  {m}/{s}: ' + ', '.join(f'{k[2]} {v}' for k, v in g.items()))
    base = predict(build(d, P(walk=2.0)), {}, 0.0).set_index('row_id').reindex(L.index)
    cur = {'cups': {'prod': base.cups}, 'dishes': {'prod': base.dishes, 'prod_x2': base.dishes * 2}, 'items': {}}
    owl = ROOT / 'out/owl_train.csv'
    if owl.exists():
        oc = prepare(pd.read_csv(owl), d['tables'])
        cur['items']['owl_phone'] = counts(oc, L.index, {'a mobile phone': .25}, ['items'])['items']
        cur['items']['owl_phone_wallet'] = counts(oc, L.index, {'a mobile phone': .25, 'a wallet': .25}, ['items'])['items']
    else:
        print('нет out/owl_train.csv: items сравниваются с нулём')
        cur['items']['zero'] = pd.Series(0.0, index=L.index)
    pd.set_option('display.width', 220)
    for cls in ['dishes', 'cups', 'items']:
        y = L[cls]
        ev = lambda p: [quality(y.loc[idx[day]].to_numpy(), p.reindex(idx[day]).fillna(0).to_numpy()) for day in DAYS]
        print(f'\n===== {cls}: всего {int(y.loc[idx[DAYS[0]]].sum())} (26.10) и {int(y.loc[idx[DAYS[1]]].sum())} (25.10)')
        ref = {}
        for k, v in cur[cls].items():
            ref[k] = ev(v)
            print(f'  текущее {k:18s} 26: {ref[k][0]:.3f}  25: {ref[k][1]:.3f}  среднее {np.mean(ref[k]):.3f}')
        best_ref = max(ref.values(), key=np.mean)
        rows = []
        for (m, s), g in c.groupby(['model', 'scope']):
            for sname, labels in SETS[cls].items():
                k = g[g.label.isin(labels)]
                if not len(k):
                    continue
                k = pd.concat([nms(x) for _, x in k.groupby('row_id')])
                for thr in THRS:
                    new = k[k.score >= thr].groupby('row_id').size().reindex(L.index).fillna(0)
                    variants = {'new': new}
                    for bn, bv in cur[cls].items():
                        variants[f'max({bn})'] = np.maximum(bv, new)
                        variants[f'sum({bn})'] = bv + new
                    for mode, p in variants.items():
                        q = ev(p)
                        rows.append(dict(model=m, scope=s, labels=sname, thr=thr, mode=mode, q26=q[0], q25=q[1],
                                         avg=np.mean(q), n=int(new.sum())))
        R = pd.DataFrame(rows)
        if not len(R):
            print('  нет кандидатов')
            continue
        print(R.sort_values('avg', ascending=False).head(12).round(3).to_string(index=False))
        print('  лучший по каждой модели/области:')
        print(R.loc[R.groupby(['model', 'scope']).avg.idxmax()].round(3).to_string(index=False))
        a, b = R.loc[R.q25.idxmax()], R.loc[R.q26.idxmax()]
        both = ((R.q26 > best_ref[0]) & (R.q25 > best_ref[1])).sum()
        print(f'  кросс-выбор: выбран на 25.10 -> 26.10 {a.q26:.3f} (текущее {best_ref[0]:.3f}); '
              f'выбран на 26.10 -> 25.10 {b.q25:.3f} (текущее {best_ref[1]:.3f}); '
              f'среднее {np.mean([a.q26, b.q25]):.3f} против {np.mean(best_ref):.3f}')
        print(f'  конфигов лучше текущего на обоих днях: {both} из {len(R)}')
        R.to_csv(ROOT / f'out/det_closed/grid_{cls}.csv', index=False)


if __name__ == '__main__':
    main()
