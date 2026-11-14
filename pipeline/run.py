"""Полный воспроизводимый запуск решения команды kozik (RASCAR.HACK, кейс 2) одной командой:

    python pipeline/run.py --weights WEIGHTS_DIR [--device cuda|mps|cpu]

Ожидает данные соревнования в data/raw (как в архиве Kaggle). WEIGHTS_DIR содержит:
    yolo_aux_s.pt     YOLO11s (Ultralytics, AGPL-3.0, старт yolo11s.pt), дообученный на 45 aux-кадрах: src/yolo_aux.py
    yolo_weak_ALL.pt  YOLO11s, дообученный на aux + слабой разметке train: src/yolo_weak.py + ядро rascar-yolo-weak
Предобученные модели с Hugging Face, commit зафиксирован:
    google/owlv2-base-patch16-ensemble @ cfd3195b (Apache-2.0)
    ustc-community/dfine-xlarge-obj365 @ a99923ec (Apache-2.0)
    facebook/sam3 @ 3c879f39 (SAM License, доступ по запросу на HF) - только если задан --sam3 (локальная папка с весами
    этого коммита) или есть токен HF с доступом; без него собираются v12/v13
Шаги: OWLv2 по кропам -> YOLO aux (обычный + TTA) -> YOLO weak -> D-FINE + тарелки -> [SAM 3 по кропам] -> сборка
pipeline/assemble.py. Кэши пишутся в out/pipeline_cache, сабмиты в out/pipeline_out (submission_v14.csv - основной)."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE, OUT = ROOT / 'out' / 'pipeline_cache', ROOT / 'out' / 'pipeline_out'
REV = dict(OWL_REV='cfd3195ba4ea9592eec887ded089f4c08eff231d', REV_DFINE_O365='a99923ec2727499f769a82edba864299fceaca48')


def sh(args, **env):
    print('>>', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True, env={**os.environ, **REV, **env})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--device', default=None)
    ap.add_argument('--sam3', default=None, help='папка с весами facebook/sam3 @ 3c879f39 или "hf"')
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    dev = a.device or ('cuda' if __import__('torch').cuda.is_available() else 'mps')
    ydev = '0' if dev == 'cuda' else dev
    if not (CACHE / 'owl_test.csv').exists():
        sh(['src/owl.py', '--split', 'test', '--out', CACHE / 'owl_test.csv'])
    for name, w, extra in [('yolo_test.csv', 'yolo_aux_s.pt', {}), ('yolo_test_tta.csv', 'yolo_aux_s.pt', {'YOLO_TTA': '1'}),
                           ('yolo_weak_ALL.csv', 'yolo_weak_ALL.pt', {})]:
        if not (CACHE / name).exists():
            sh(['src/yolo_counts.py', 'detect', 'test'], YOLO_WEIGHTS=str(a.weights / w), YOLO_OUT=str(CACHE / name),
               YOLO_DEVICE=ydev, **extra)
    if not (CACHE / 'plates_test.csv').exists():
        sh(['exp/det_closed/detect.py', '--split', 'test', '--models', 'dfine_o365'])
        sh(['exp/det_closed/plates.py', '--split', 'test'])
        shutil.copy(ROOT / 'out' / 'det_closed' / 'plates_test.csv', CACHE / 'plates_test.csv')
    if a.sam3 and not (CACHE / 'sam3_test.csv').exists():
        env = dict(SAM_OUT_DIR=str(CACHE), SAM_DEVICE=dev)
        if a.sam3 != 'hf':
            env['SAM_PATH'] = str(a.sam3)
        sh(['src/sam3_counts.py', 'test'], **env)
    sh(['pipeline/assemble.py', '--cache', CACHE, '--out', OUT])


if __name__ == '__main__':
    main()
