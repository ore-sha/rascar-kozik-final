"""ID 26: детекторы с готовым словарём (Objects365 / COCO) по кропам столов.

Модели (все Apache-2.0, веса с Hugging Face, точный commit SHA пишется в <out>.json):
  ustc-community/dfine-xlarge-obj365     D-FINE-X, словарь Objects365 (есть Plate, Bowl/Basin, Cup, Cell Phone, Handbag)
  ustc-community/dfine-xlarge-obj2coco   D-FINE-X, предобучение Objects365, словарь COCO
  PekingU/rtdetr_v2_r101vd               RT-DETRv2-R101, словарь COCO
Недоступная модель пропускается с сообщением, остальные считаются.

Кроп = рамка стола, расширенная (влево/вправо 0.25w, вверх 0.6h, вниз 0.25h); процессор модели сам приводит его
к своему входу (640x640), то есть мелкий стол увеличивается в 2-4 раза. Дополнительно scope=frame: кадр целиком.
Сохраняются все боксы со score >= 0.1 в координатах кадра; выбор классов и порогов делает run.py.
Запуск (Kaggle GPU): python exp/det_closed/detect.py --split train
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    'dfine_o365': 'ustc-community/dfine-xlarge-obj365',
    'dfine_o2c': 'ustc-community/dfine-xlarge-obj2coco',
    'rtdetrv2': 'PekingU/rtdetr_v2_r101vd',
}
PAD = dict(left=.25, right=.25, up=.6, down=.25)
MIN_SCORE = 0.1


def crop_box(r, W, H):
    w, h = r.x2 - r.x1, r.y2 - r.y1
    return [int(max(0, r.x1 - PAD['left'] * w)), int(max(0, r.y1 - PAD['up'] * h)),
            int(min(W, r.x2 + PAD['right'] * w)), int(min(H, r.y2 + PAD['down'] * h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, default=ROOT / 'data/raw')
    ap.add_argument('--split', choices=['train', 'test'], default='train')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--models', default=','.join(MODELS))
    ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    out = a.out or ROOT / f'out/det_closed/cands_{a.split}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, __version__ as tv
    tg = pd.read_csv(a.raw / 'targets.csv')
    tabs = pd.read_csv(a.raw / 'tables.csv')
    tabs = tabs[tabs.frame_id.isin(tg[tg.split == a.split].frame_id)].sort_values(['frame_id', 'table_id'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    meta = dict(transformers=tv, torch=torch.__version__, pad=PAD, min_score=MIN_SCORE, models={})
    res = []
    for name in a.models.split(','):
        repo = MODELS[name]
        try:
            sha = os.environ.get(f'REV_{name.upper()}') or HfApi().model_info(repo).sha   # pipeline фиксирует commit
            proc = AutoImageProcessor.from_pretrained(repo, revision=sha)
            model = AutoModelForObjectDetection.from_pretrained(repo, revision=sha).to(dev).eval()
        except Exception as e:
            print(f'SKIP {name} ({repo}): {repr(e)[:300]}', flush=True)
            continue
        id2label = model.config.id2label
        t0, n0 = time.time(), len(res)

        def run(images, metas):
            inp = proc(images=images, return_tensors='pt').to(dev)
            with torch.no_grad():
                o = model(**inp)
            sizes = torch.tensor([im.size[::-1] for im in images], device=dev)
            for det, (scope, row_id, frame_id, ox, oy) in zip(
                    proc.post_process_object_detection(o, target_sizes=sizes, threshold=MIN_SCORE), metas):
                for s, l, b in zip(det['scores'].tolist(), det['labels'].tolist(), det['boxes'].tolist()):
                    res.append(dict(model=name, scope=scope, row_id=row_id, frame_id=frame_id, label=id2label[l],
                                    score=round(s, 4), x1=b[0] + ox, y1=b[1] + oy, x2=b[2] + ox, y2=b[3] + oy))

        for fid, g in tabs.groupby('frame_id'):
            im = Image.open(a.raw / f'{fid}.jpg').convert('RGB')
            run([im], [('frame', '', fid, 0, 0)])
            rows = list(g.itertuples())
            for i in range(0, len(rows), a.batch):
                ch = rows[i:i + a.batch]
                boxes = [crop_box(r, *im.size) for r in ch]
                run([im.crop(b) for b in boxes], [('crop', r.row_id, fid, b[0], b[1]) for r, b in zip(ch, boxes)])
        meta['models'][name] = dict(repo=repo, revision=sha, license='Apache-2.0', n_labels=len(id2label),
                                    seconds=round(time.time() - t0), boxes=len(res) - n0)
        print(f'{name}: {len(res) - n0} boxes, {time.time() - t0:.0f}s, {len(id2label)} labels', flush=True)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(res).to_csv(out, index=False)
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print('saved', out, len(res))


if __name__ == '__main__':
    main()
