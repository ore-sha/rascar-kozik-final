"""ID 26: число тарелок на стол по кандидатам D-FINE (Objects365).

Правило (выбрано на train, см. dishes.py): метка Plate, бокс из кропа стола со score >= 0.6 или из целого кадра
со score >= 0.3; повторы убирает NMS (IoU 0.5); бокс относится к столу, если его центр внутри рамки (+10%).
На train: 11 тарелок, все на столах с посудой, на столах без посуды ни одной.
Запуск: python exp/det_closed/plates.py --split test   (после detect.py --split test --models dfine_o365)
Пишет out/det_closed/plates_<split>.csv (row_id, plates; только ненулевые строки) и печатает номера этих строк
в sample_submission.csv / train_labels.csv с контрольными суммами: выходные файлы с Kaggle не скачиваются,
результат переносится с экрана скриптом from_screen.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'exp/det_closed'), str(ROOT / 'src')]
from owl_counts import nms  # noqa: E402
from run import assign  # noqa: E402

MODEL, T_CROP, T_FRAME = 'dfine_o365', 0.6, 0.3


def plates(cands: pd.DataFrame, tables: pd.DataFrame) -> pd.Series:
    c = cands[(cands.model == MODEL) & (cands.label.str.lower() == 'plate')].copy()
    c = c[((c.scope == 'crop') & (c.score >= T_CROP)) | ((c.scope == 'frame') & (c.score >= T_FRAME))]
    c = assign(c, tables)
    if not len(c):
        return pd.Series(dtype=float)
    c = pd.concat([nms(x) for _, x in c.groupby('row_id')])
    return c.groupby('row_id').size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'test'], default='test')
    a = ap.parse_args()
    raw = ROOT / 'data/raw'
    tg = pd.read_csv(raw / 'targets.csv')
    tabs = pd.read_csv(raw / 'tables.csv')
    tabs = tabs[tabs.frame_id.isin(tg[tg.split == a.split].frame_id)]
    c = pd.read_csv(ROOT / f'out/det_closed/cands_{a.split}.csv', keep_default_na=False)
    c = c[c.label.str.lower() == 'plate']
    p = plates(c, tabs)
    out = ROOT / f'out/det_closed/plates_{a.split}.csv'
    p.rename('plates').rename_axis('row_id').reset_index().to_csv(out, index=False)
    print(f'{a.split}: столов {len(tabs)}, с тарелками {len(p)} ({len(p) / len(tabs):.1%}), тарелок {int(p.sum())}; '
          f'распределение {p.value_counts().sort_index().to_dict()}')
    ids = pd.read_csv(raw / ('sample_submission.csv' if a.split == 'test' else 'train_labels.csv')).row_id
    pos = pd.Series(range(len(ids)), index=ids)
    for k in sorted(p.unique()):   # номера строк (с нуля) в sample_submission.csv / train_labels.csv
        v = sorted(pos[p.index[p == k]].dropna().astype(int))
        print(f'== plates={k}: {len(v)} строк, сумма номеров {sum(v)}')
        for i in range(0, len(v), 10):
            ch = v[i:i + 10]
            print('  ' + ' '.join(f'{x:4d}' for x in ch) + f'  | {sum(ch)}')

if __name__ == '__main__':
    main()
