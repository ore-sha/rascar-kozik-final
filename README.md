# RASCAR.HACK, кейс 2 «Table State»: команда kozik

Финальное решение - v17. Валидация на днях train 0.837 (26.10: 0.804, 25.10: 0.870), public 0.755.
Подробности подхода, валидации и всех проверенных гипотез - в [REPORT.md](REPORT.md).

## Воспроизведение одной командой

Нужны Python 3.12, GPU (проверено на Kaggle T4, около 2.5 ч) или Apple Silicon (MPS, дольше).

```bash
pip install -r requirements.txt

# 1. Данные соревнования (в репозиторий не кладём)
kaggle competitions download rascar-hack-case-2 -p data
unzip -q data/rascar-hack-case-2.zip -d data/raw          # в data/raw лежат frames.csv, tables.csv, ... и *.jpg

# 2. Веса SAM 3: доступ по запросу на huggingface.co/facebook/sam3, затем
huggingface-cli login

# 3. Весь пайплайн: OWLv2 -> три YOLO11s -> D-FINE -> SAM 3 -> сборка
python pipeline/run.py --weights weights --sam3 hf
```

Результат: `out/pipeline_out/submission_v17.csv` (финальный). Рядом лежат v16, v14, v13 и v12 - промежуточные шаги.
Вместо `--sam3 hf` можно передать локальную папку с весами facebook/sam3 того же коммита. Без `--sam3`
собираются только v12 и v13, им SAM 3 не нужен.

Все предобученные модели берутся с Hugging Face с зафиксированным commit, так что результат детерминирован.
Контрольные суммы отправленных файлов (md5):

| файл | md5 |
|---|---|
| submission_v17.csv (финальный, public 0.75475) | 8c92d54517489701ce3f4f2748ffa848 |
| submission_v13.csv (без SAM 3, public 0.71868) | 93303802f9b1de3c6157a3f4933eeb78 |

Проверено 14.11: запуск с нуля на Kaggle T4 дал v12, v13 и v14 побайтно как отправленные. v16 и v17 из тех же
кэшей тоже совпали побайтно.

## Что где лежит

| путь | что это |
|---|---|
| `pipeline/run.py` | запуск всего решения: детекторы по test, кэши в `out/pipeline_cache` |
| `pipeline/assemble.py` | сборка сабмитов v12-v17 из кэшей детекций, все правила по классам |
| `weights/` | наши дообученные YOLO11s: `yolo_aux_s.pt` (45 aux-кадров), `yolo_weak_ALL.pt` (aux + слабая разметка train) |
| `src/owl.py`, `src/owl_counts.py` | OWLv2 по увеличенным кропам столов: телефоны и кошельки |
| `src/yolo_counts.py`, `src/yolo_aux.py`, `src/yolo_weak.py` | инференс и обучение YOLO11s |
| `src/sam3_counts.py` | SAM 3 по кропам столов: чашки и руки |
| `exp/people_ctx/people.py` | привязка людей и предметов к столам |
| `exp/det_closed/` | D-FINE Objects365: тарелки |
| `src/metric.py` | метрика соревнования |

## Обучение своих моделей с нуля

```bash
python src/yolo_aux.py prepare && python src/yolo_aux.py train     # YOLO11s на 45 aux-кадрах
python src/yolo_weak.py                                            # слабая разметка train + дообучение
```

## Модели и лицензии

OWLv2 base (Apache-2.0), D-FINE-X Objects365 (Apache-2.0), SAM 3 (SAM License), YOLO11s (AGPL-3.0, наши веса -
производные). Закрытых облачных моделей и API нет, ручной разметки test нет.
