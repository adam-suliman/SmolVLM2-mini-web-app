# README.md
# SmolVLM2 mini web app (Docker)

Минимальный demo веб-интерфейс (Gradio) для SmolVLM2: мультимодальный чат (изображение/видео + текст), OCR с выгрузкой `.txt`, описание по координатам.

## ВАЖНО про кэш/веса (чтобы не скачивать каждый раз)
Контейнер использует примонтированную директорию `./data/hf_cache` (на хосте) как единый корень кэшей Hugging Face:
- hub cache → `/data/hf_cache/hub`
- datasets cache → `/data/hf_cache/datasets`

Docker Compose автоматически создаёт `./data/hf_cache` на хосте при первом запуске.

Поведение загрузки модели в приложении:
- Сначала приложение пытается загрузить модель **только из локального кэша** (без интернета).
- Если файлов нет:
  - при `HF_HUB_OFFLINE=1` запуск завершится с понятной ошибкой,
  - при `HF_HUB_OFFLINE=0` модель будет скачана в примонтированный кэш и дальше будет переиспользоваться.

Справка:
- Кэши HF datasets: https://huggingface.co/docs/datasets/cache
- Кэш Hugging Face Hub (модели/датасеты/spaces): https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- Transformers cache/offline/local_files_only: https://huggingface.co/docs/transformers/en/installation

---

## Запуск

### 1) Скачайте репозиторий
~bash
git clone https://github.com/adam-suliman/SmolVLM2-mini-web-app
cd SmolVLM2-mini-web-app
~

### 2) Создайте .env
~bash
cp .env.example .env
~

### 3) Соберите образ
~bash
docker compose build
~

### 4) Первый запуск (онлайн, чтобы скачать веса если их нет в ./data/hf_cache)
Убедитесь, что в `.env` стоит:
~env
HF_HUB_OFFLINE=0
~
Запуск:
~bash
docker compose up
~

Открыть в браузере (с хоста):
- http://localhost:7860 (по умолчанию)
- или http://localhost:<APP_PORT>

Остановить:
~bash
docker compose down
~

---

## Как поменять порт
В `.env`:
~env
APP_PORT=8080
~
Перезапуск:
~bash
docker compose down
docker compose up
~
Доступ:
- http://localhost:8080

---

## GPU / CPU
В `.env`:
- авто:
~env
MODEL_DEVICE=auto
~
- принудительно CPU:
~env
MODEL_DEVICE=cpu
~
- принудительно GPU:
~env
MODEL_DEVICE=cuda
~

Чтобы контейнер увидел GPU, в `docker-compose.yml` раскомментируйте:
~yaml
gpus: all
~
Перезапуск:
~bash
docker compose down
docker compose up --build
~

---

## Как изменить размер/вариант SmolVLM2
В `.env` задайте `SMOLVLM2_MODEL_ID`, например:
- 2.2B:
~env
SMOLVLM2_MODEL_ID=HuggingFaceTB/SmolVLM2-2.2B-Instruct
~
- 500M:
~env
SMOLVLM2_MODEL_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
~
- 256M:
~env
SMOLVLM2_MODEL_ID=HuggingFaceTB/SmolVLM2-256M-Video-Instruct
~

Перезапуск:
~bash
docker compose down
docker compose up
~

---

## Как использовать свой путь к кэшу/весам на хосте
По умолчанию используется `./data/hf_cache:/data/hf_cache`.
Если хотите другой путь на хосте — измените `source:` в `docker-compose.yml`, например:
~yaml
volumes:
  - type: bind
    source: /mnt/models/hf_cache
    target: /data/hf_cache
    bind:
      create_host_path: true
~

---

## Полный офлайн-режим (после того как веса уже скачаны)
1) Один раз запустите онлайн (см. выше), чтобы модель попала в `./data/hf_cache`.
2) В `.env` включите офлайн:
~env
HF_HUB_OFFLINE=1
~
3) Запуск:
~bash
docker compose up
~

---

## Ссылка на тех. отчёт / описание модели
- Paper (arXiv): https://arxiv.org/abs/2504.05299
- Hugging Face paper page: https://huggingface.co/papers/2504.05299
- Model card (пример): https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
