# SmolVLM2 mini web app (Docker)

Gradio UI для SmolVLM2: мультимодальные VQA (картинки/видео + текст) и OCR с выгрузкой в .txt. В UI есть две вкладки: VQA и OCR.

## Важное
- Если CUDA недоступна, приложение показывает предупреждение и автоматически работает на CPU.
- Кэши монтируются в контейнер, чтобы не скачивать модели заново.

## Требования
- Docker и Docker Compose v2.
- Для GPU: драйвер NVIDIA и nvidia-container-toolkit (в `docker-compose.yml` уже есть `gpus: all`). Без GPU всё работает на CPU.

## Настройка `.env` (обязательно)
Скопируйте пример и отредактируйте под себя:
```
cp .env.example .env
```
Переменные:
- `APP_PORT` — порт UI (по умолчанию 7860).
- `SMOLVLM2_MODEL_ID` — выбираем модель:
  - `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
  - `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  - `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- `MODEL_DEVICE` — `auto` (по умолчанию), `cuda` или `cpu`.
- `MODEL_DTYPE` (опционально) — `float16` / `bfloat16` / `float32`; если не указано, то bfloat16 на Ampere+ или float16 на более старых картах, float32 на CPU.
- Кэши Hugging Face/Transformers (используются и в Docker):
  - `HF_HOME=/data/hf_cache`
  - `HF_HUB_CACHE=/data/hf_cache/hub`
  - `TRANSFORMERS_CACHE=/data/hf_cache/hub`
  - `HF_DATASETS_CACHE=/data/hf_cache/datasets`
- `HF_HUB_OFFLINE` — `0` (можно скачивать) или `1` (только локальный кэш).

## Быстрый старт в Docker
1) Подготовьте папку для кэшей на хосте (смонтируется в контейнер):
```
mkdir -p data/hf_cache
```
2) Соберите образ:
```
docker compose build
```
3) Запустите:
```
docker compose up
```
UI: http://localhost:${APP_PORT:-7860}  
Остановить: `docker compose down`

### Офлайн-режим
- Заранее скачайте модель в `./data/hf_cache/hub`.
- В `.env` выставьте `HF_HUB_OFFLINE=1`. Без кэша ничего не загрузится.

## Проверка
- При старте ищите лог `[init] device=..., dtype=..., capability=..., gpu=...]` — так видно, выбрался ли GPU.
