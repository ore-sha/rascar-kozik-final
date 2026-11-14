"""ID46: SAM 3 (facebook/sam3 @ 3c879f39, SAM License) - сегментация всех экземпляров по тексту на кропе стола.
Запуск: python src/sam3_counts.py train|test [limit]  ->  out/sam3_<split>.csv (row_id, cls, score, бокс в пикселях кадра).
Сохраняются экземпляры со score >= 0.3; пороги выбираются потом на train-днях."""
import os
import sys
import time
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
REPO, REV = 'facebook/sam3', '3c879f39826c281e95690f02c7821c4de09afae7'
PAD = (0.35, 0.6, 0.35, 0.25)
PROMPTS = {'dish': 'plate', 'bowl': 'bowl', 'tray': 'tray', 'cup': 'cup', 'hand': 'hand',
           'phone': 'mobile phone', 'wallet': 'wallet', 'paper': 'sheet of paper', 'bag': 'handbag'}
if os.environ.get('SAM_PROMPTS'):   # ID65b: другой набор запросов, JSON {имя: текст}
    import json; PROMPTS = json.loads(os.environ['SAM_PROMPTS'])


def rep(x, n):
    """Повторить все тензоры выхода энкодера по батчу n раз."""
    if torch.is_tensor(x):
        return x.expand(n, *x.shape[1:])
    if isinstance(x, (list, tuple)):
        return type(x)(rep(v, n) for v in x)
    if hasattr(x, 'keys'):
        for k in list(x.keys()):
            if x[k] is not None:
                x[k] = rep(x[k], n)
        return x
    return x


def main():
    split = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    dev = os.environ.get('SAM_DEVICE') or ('cuda' if torch.cuda.is_available() else 'mps')
    src = os.environ.get('SAM_PATH') or REPO   # на Kaggle - локальная копия весов того же коммита
    kw = {} if os.environ.get('SAM_PATH') else dict(revision=REV)
    dtype = torch.float16 if dev == 'cuda' else torch.float32
    model = Sam3Model.from_pretrained(src, torch_dtype=dtype, **kw).to(dev).eval()
    proc = Sam3Processor.from_pretrained(src, **kw)
    txt_in = proc(text=[PROMPTS[n] for n in PROMPTS], return_tensors='pt').to(dev)
    with torch.no_grad():
        txt = model.get_text_features(input_ids=txt_in['input_ids'], attention_mask=txt_in['attention_mask'], return_dict=True)
    tg = pd.read_csv(RAW / 'targets.csv'); T = pd.read_csv(RAW / 'tables.csv')
    T = T[T.frame_id.isin(tg[tg.split == split].frame_id)]
    if limit:
        T = T.head(limit)
    if os.environ.get('SAM_ROWS'):   # только эти строки (json-список row_id)
        import json; T = T[T.row_id.isin(set(json.load(open(os.environ['SAM_ROWS']))))]
    out_p = Path(os.environ.get('SAM_OUT_DIR') or ROOT / 'out') / f"sam3{os.environ.get('SAM_TAG', '')}_{split}.csv"
    done = set(pd.read_csv(out_p).row_id) if out_p.exists() else set()
    todo = [r for r in T.itertuples() if r.row_id not in done]
    print(split, len(T), 'todo', len(todo), dev, flush=True)
    rows, t0, cache = [], time.time(), {}
    for i, r in enumerate(todo):
        if r.frame_id not in cache:
            cache.clear(); cache[r.frame_id] = Image.open(RAW / f'{r.frame_id}.jpg').convert('RGB')
        w, h = r.x2 - r.x1, r.y2 - r.y1
        b = (int(max(0, r.x1 - PAD[0] * w)), int(max(0, r.y1 - PAD[1] * h)), int(min(1280, r.x2 + PAD[2] * w)), int(min(720, r.y2 + PAD[3] * h)))
        crop = cache[r.frame_id].crop(b)
        names = list(PROMPTS)
        img_in = proc(images=crop, return_tensors='pt').to(dev)
        with torch.no_grad():
            # картинка кодируется один раз, 9 текстовых запросов идут батчем по готовым признакам
            ve = model.get_vision_features(pixel_values=img_in['pixel_values'].to(dtype))
            ve = rep(ve, len(names))
            out = model(vision_embeds=ve, text_embeds=txt, attention_mask=txt_in['attention_mask'])
        inputs = {'original_sizes': img_in['original_sizes'].repeat(len(names), 1)}
        res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5,
                                                      target_sizes=inputs['original_sizes'].tolist())
        for n, rr in zip(names, res):
            for bb, s in zip(rr['boxes'].tolist(), rr['scores'].tolist()):
                rows.append(dict(row_id=r.row_id, frame_id=r.frame_id, cls=n, score=round(s, 4),
                                 x1=bb[0] + b[0], y1=bb[1] + b[1], x2=bb[2] + b[0], y2=bb[3] + b[1]))
        rows.append(dict(row_id=r.row_id, frame_id=r.frame_id, cls='_done', score=0, x1=0, y1=0, x2=0, y2=0))
        if (i + 1) % 20 == 0 or i == len(todo) - 1:
            pd.DataFrame(rows).to_csv(out_p, mode='a', header=not out_p.exists(), index=False); rows = []
            print(f'{i + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    print('saved', out_p)


if __name__ == '__main__':
    main()
