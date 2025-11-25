# SmolVLM2 + Gradio (Docker)

Демо-веб интерфейс (Gradio) для SmolVLM2: мультимодальный чат (изображение/видео + текст), OCR с выгрузкой .txt, описание по координатам.

## Требования
- Docker + Docker Compose v2 (`docker compose`)
- Зависит от настройки .env(e.g. NVIDIA драйвер + `nvidia-container-toolkit`, чтобы контейнер видел GPU)

---

## Быстрый старт (первый запуск с интернетом, чтобы скачать веса модели)
1) Скачайте репозиторий:
    git clone <REPO_URL>
    cd <REPO_DIR>

2) Создайте директорию для кэша/весов HF на хосте (она будет примонтирована в контейнер):
    mkdir -p data/hf_cache

3) Создайте `.env`:
    cp .env.example .env

4) Запустите приложение:
    docker compose up

5) Откройте в браузере (с хоста):
- По умолчанию: http://localhost:7860
- Или: http://localhost:<APP_PORT> (если меняли порт)

Остановить:
    Ctrl+C
или:
    docker compose down

---

## Где и на каком порту доступно приложение
- Порт берётся из переменной `APP_PORT` в .env (по умолчанию 7860).
- Доступ с хоста: http://localhost:APP_PORT

---

## Как поменять порт
В `.env` измените:
    APP_PORT=xxxx

Перезапустите:
    docker compose down
    docker compose up

Открывайте: http://localhost:xxxx

---

## Как включить GPU / принудительно CPU
В `.env`:
- Авто (GPU если доступен, иначе CPU):
    MODEL_DEVICE=auto
- Принудительно GPU (если GPU недоступен — будет fallback на CPU с предупреждением):
    MODEL_DEVICE=cuda
- Принудительно CPU:
    MODEL_DEVICE=cpu

ВАЖНО: чтобы контейнер реально увидел GPU, добавьте в `docker-compose.yml` (в сервис `app`) строку:
    gpus: all

Перезапуск:
    docker compose down
    docker compose up --build

---

## Как изменить размер/вариант модели SmolVLM2
В `.env` установите `SMOLVLM2_MODEL_ID`, например:
- Большая (по умолчанию):
    SMOLVLM2_MODEL_ID=HuggingFaceTB/SmolVLM2-2.2B-Instruct
- Средняя:
    SMOLVLM2_MODEL_ID=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
- Маленькая:
    SMOLVLM2_MODEL_ID=HuggingFaceTB/SmolVLM2-256M-Video-Instruct

Перезапуск:
    docker compose down
    docker compose up

Примечание: при первом запуске с новой моделью она докачается в тот же кэш `data/hf_cache`.

---

## Как примонтировать с хоста директорию с весами модели (HF cache)
По умолчанию уже настроено в `docker-compose.yml`:
- Хост: `./data/hf_cache`
- Контейнер: `/data/hf_cache` (переменная `HF_HOME=/data/hf_cache`)

Фрагмент:
    volumes:
      - ./data/hf_cache:/data/hf_cache

Если хотите другой путь на хосте — замените левую часть, например:
    volumes:
      - /mnt/models/hf_cache:/data/hf_cache

---

## Офлайн запуск после первого скачивания
1) Один раз запустите онлайн (как в “Быстрый старт”), чтобы модель скачалась в `data/hf_cache`.
2) Затем в `.env`:
    HF_HUB_OFFLINE=1
3) Запуск:
    docker compose up

---

## Ссылки на описание модели / тех. отчёт
- Технический отчёт (arXiv): https://arxiv.org/abs/2504.05299
- Официальный блог HF про SmolVLM2: https://huggingface.co/blog/smolvlm2
- Model card (пример, 256M Instruct): https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
