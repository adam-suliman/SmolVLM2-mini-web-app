# SmolVLM2 mini web app (Docker)

Gradio UI для SmolVLM2: мультимодальные вопросы-ответы (картинки/видео + текст) и OCR с сохранением в .txt. В интерфейсе две вкладки: VQA и OCR. 

## Что важно
- Если CUDA недоступна, приложение выводит предупреждение и автоматически работает на CPU.
- Кэши моделей/датасетов монтируются с хоста, поэтому веса не попадают в образ.
- Compose не требует GPU: без видеокарты контейнер запустится на CPU. Для запуска с GPU добавьте `gpus: all` в сервис `app` или используйте `docker run --gpus all ...`.

## Требования
- Docker и Docker Compose v2.
- Для GPU: драйвер NVIDIA и `nvidia-container-toolkit` (если планируете добавлять `gpus: all`). Без GPU приложение работает на CPU.

## Настройка `.env` 
Скопируйте пример и задайте свои значения:
```
cp .env.example .env
```
Основные переменные:
- `APP_PORT` — порт UI (по умолчанию 7860).
- `SMOLVLM2_MODEL_ID` — выбор модели:
  - `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
  - `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
  - `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`
- `MODEL_DEVICE` — `auto` (по умолчанию), `cuda` или `cpu`.
- `MODEL_DTYPE` (опционально) — `float16` / `bfloat16` / `float32`; если не задано: bfloat16 на Ampere+ или float16 на старших GPU, float32 на CPU.
- Кэши Hugging Face/Transformers (используются и в Docker):
  - `HF_HOME=/data/hf_cache`
  - `HF_HUB_CACHE=/data/hf_cache/hub`
  - `TRANSFORMERS_CACHE=/data/hf_cache/hub`
  - `HF_DATASETS_CACHE=/data/hf_cache/datasets`
- `HF_HUB_OFFLINE` — `0` (можно скачивать) или `1` (только локальный кэш).

## Сборка и запуск Docker-контейнера
1) Подготовьте папку для кэшей на хосте (будет примонтирована в контейнер):
```
mkdir -p data/hf_cache
```
2) Соберите образ:
```
docker compose build
```
3) Запустите приложение:
```
docker compose up
```
UI доступен на http://localhost:7860
Остановить: `docker compose down`


## Как поменять порт
Измените `APP_PORT` в `.env`, затем перезапустите контейнеры (`docker compose down` и `docker compose up`). Приложение будет на `http://localhost:<APP_PORT>`.

## GPU или CPU
- GPU: `MODEL_DEVICE=cuda`, раскомментиируйте `gpus: all` в `docker-compose.yml` или запускайте с `docker run --gpus all ...`; на хосте должны быть драйвер и `nvidia-container-toolkit`.
- CPU: `MODEL_DEVICE=cpu` (или `auto` без GPU) — работает без дополнительных настроек.

## Как выбрать размер модели
Установите нужный `SMOLVLM2_MODEL_ID` в `.env` (2.2B / 500M / 256M) и перезапустите контейнер — новая модель скачается в кэш.

## Как примонтировать веса с хоста
- По умолчанию `./data/hf_cache` на хосте монтируется как `/data/hf_cache` в контейнере (см. `volumes` в `docker-compose.yml`). Все скачанные веса и датасеты лежат там.
- Чтобы использовать другой путь, измените `source:` в `docker-compose.yml`, например:
  ```yaml
  volumes:
    - type: bind
      source: /mnt/models/hf_cache
      target: /data/hf_cache
      bind:
        create_host_path: true
  ```
- Для офлайн-режима заранее сложите веса в эту директорию и поставьте `HF_HUB_OFFLINE=1` в `.env`.

## Проверка работы
- В логах при старте ищите строку `[init] device=..., dtype=..., capability=..., gpu=...]` — видно, выбран ли GPU.
- Очистить кэши можно удалением папки `data/hf_cache` на хосте (при остановленных контейнерах).

## Ссылки
- Статья (arXiv): https://arxiv.org/abs/2504.05299
- Model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
