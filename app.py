import os
import time
import warnings
from typing import List, Tuple, Optional

import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# -----------------------------
# Конфигурация через env vars
# -----------------------------
MODEL_ID = os.getenv("SMOLVLM2_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DEVICE_SETTING = os.getenv("MODEL_DEVICE", "auto").lower()
APP_PORT = int(os.getenv("APP_PORT", "7860"))

# Hugging Face caches (в docker-compose они направлены в примонтированную директорию)
HF_HOME = os.getenv("HF_HOME", None)
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", None) or os.getenv("TRANSFORMERS_CACHE", None)
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", None)
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE", "0").strip().lower() in ("1", "true", "yes", "y")

# Подготовим директории кэша (если заданы)
for _p in [HF_HOME, HF_HUB_CACHE, HF_DATASETS_CACHE]:
    if _p:
        os.makedirs(_p, exist_ok=True)

# -----------------------------
# Выбор устройства
# -----------------------------
if DEVICE_SETTING == "cuda":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn("Запрошен режим CUDA, но GPU недоступен. Используется CPU.")
        device = "cpu"
elif DEVICE_SETTING == "cpu":
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.bfloat16 if device == "cuda" else torch.float32


# -----------------------------
# Загрузка модели/процессора
# Требование: скачивать только если нет в примонтированном кэше
# -----------------------------
def _load_processor_and_model() -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    """
    1) Сначала пытаемся загрузить ТОЛЬКО из локального кэша (local_files_only=True).
       Если файлы есть в примонтированной директории — интернет не нужен и скачивания не будет.
    2) Если локально нет, то:
       - при HF_HUB_OFFLINE=1: падаем с понятным сообщением
       - иначе: скачиваем в cache_dir (примонтированную директорию)
    """
    cache_dir = HF_HUB_CACHE  # если задан, всё (модели/токенизаторы) ложится сюда

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # 1) Пробуем локально
    try:
        proc = AutoProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        mdl = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=dtype,
        )
        return proc, mdl
    except Exception:
        if HF_HUB_OFFLINE:
            raise RuntimeError(
                "Офлайн-режим включён (HF_HUB_OFFLINE=1), но файлы модели отсутствуют в примонтированном кэше. "
                "Запустите контейнер один раз с интернетом и HF_HUB_OFFLINE=0, чтобы скачать веса в кэш."
            )

    # 2) Скачиваем (если не офлайн)
    proc = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        local_files_only=False,
    )
    mdl = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        local_files_only=False,
        torch_dtype=dtype,
    )
    return proc, mdl


processor, model = _load_processor_and_model()
model.to(device)


# -----------------------------
# Валидация файлов
# -----------------------------
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4"}


def _ensure_image_readable(path: str) -> None:
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        raise gr.Error("Ожидается изображение (PNG/JPEG), а не текстовый файл.")


def _validate_chat_media(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        _ensure_image_readable(path)
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise gr.Error("Ожидается изображение (PNG/JPEG) или видео (.mp4).")


# -----------------------------
# Инференс
# -----------------------------
def _run_model(messages: List[dict], max_new_tokens: int = 128) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return texts[0]


def infer_vqa(
    image_or_video: Optional[str],
    question: str,
    history: Optional[List[Tuple[str, str]]],
    state: Optional[str],
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """
    Мультимодальный чат: VQA + captioning + описание видео.
    Требование: можно задавать несколько вопросов к одному файлу без повторной загрузки.
    """
    try:
        # Используем новый файл, если предоставлен, иначе — сохранённый state
        if image_or_video:
            current_path = image_or_video
        else:
            if state:
                current_path = state
            else:
                raise gr.Error("Сначала загрузите изображение или видео.")

        media_type = _validate_chat_media(current_path)

        if not question or not question.strip():
            raise gr.Error("Пожалуйста, введите вопрос.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": media_type, "path": current_path},
                    {"type": "text", "text": question.strip()},
                ],
            }
        ]

        answer = _run_model(messages, max_new_tokens=192)

        if history is None:
            history = []
        history.append((question.strip(), answer))

        return history, current_path
    except gr.Error:
        raise
    except Exception:
        raise gr.Error("Не удалось обработать запрос. Проверьте файл и попробуйте ещё раз.")


def ocr(image: Optional[str]) -> Tuple[str, str]:
    """
    OCR (распознавание текста с изображения) + выгрузка .txt
    """
    try:
        if not image:
            raise gr.Error("Сначала загрузите изображение.")
        ext = os.path.splitext(image)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise gr.Error("Ожидается изображение (PNG/JPEG), а не текстовый файл.")
        _ensure_image_readable(image)

        prompt = (
            "Считай это задачей OCR. Пожалуйста, перепиши ВСЁ читаемое текстовое содержимое на изображении "
            "как простой текст без дополнительных комментариев."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = _run_model(messages, max_new_tokens=512)

        ts = int(time.time())
        out_name = f"ocr_result_{ts}.txt"
        out_path = os.path.join("/tmp", out_name)
        os.makedirs("/tmp", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        return text, out_path
    except gr.Error:
        raise
    except Exception:
        raise gr.Error("Не удалось распознать текст. Проверьте изображение и попробуйте ещё раз.")


def describe_by_bbox(image: Optional[str], coords_str: str) -> str:
    """
    Описание объекта по координатам (без реального кропа, координаты передаются текстом).
    """
    try:
        if not image:
            raise gr.Error("Сначала загрузите изображение.")
        ext = os.path.splitext(image)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise gr.Error("Ожидается изображение (PNG/JPEG), а не текстовый файл.")
        _ensure_image_readable(image)

        if coords_str is None:
            coords_str = ""
        parts = [p.strip() for p in coords_str.split(",") if p.strip() != ""]
        if len(parts) != 4:
            raise gr.Error("Координаты должны быть четырьмя числами через запятую: x1,y1,x2,y2.")

        try:
            x1, y1, x2, y2 = map(float, parts)
        except Exception:
            # строго как в требовании 2.3
            raise gr.Error("Координаты должны быть в формате x1,y1,x2,y2, только числа")

        prompt = (
            "На изображении задан прямоугольник интереса (ROI) в пиксельных координатах. "
            f"Координаты ROI: x1={x1}, y1={y1}, x2={x2}, y2={y2}. "
            "Опиши объект(ы) внутри этого прямоугольника. Ответ должен быть подробным, но кратким и по делу."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return _run_model(messages, max_new_tokens=192)
    except gr.Error:
        raise
    except Exception:
        raise gr.Error("Не удалось получить описание. Проверьте входные данные и попробуйте ещё раз.")


# -----------------------------
# UI
# -----------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Демо SmolVLM2") as demo:
        gr.Markdown(
            "## Демо SmolVLM2\n"
            "- Вкладка **Мультимодальный чат**: вопросы к изображению/видео\n"
            "- Вкладка **OCR**: распознавание текста\n"
            "- Вкладка **Описание по координатам**: описание ROI x1,y1,x2,y2\n"
        )

        # Tab 1
        with gr.Tab("Мультимодальный чат"):
            state = gr.State(None)
            upload = gr.File(
                label="Загрузите изображение или видео (.mp4)",
                file_types=["image", "video"],
                type="filepath",
            )
            question = gr.Textbox(label="Ваш вопрос", placeholder="Например: Опиши изображение", lines=1)
            send_btn = gr.Button("Отправить")
            chatbot = gr.Chatbot(label="История чата")

            send_btn.click(
                infer_vqa,
                inputs=[upload, question, chatbot, state],
                outputs=[chatbot, state],
            )
            send_btn.click(lambda: "", inputs=None, outputs=question)

        # Tab 2
        with gr.Tab("OCR"):
            ocr_image = gr.Image(label="Загрузите изображение", type="filepath")
            ocr_btn = gr.Button("Распознать текст")
            ocr_output = gr.Textbox(label="Распознанный текст", interactive=False)
            ocr_download = gr.File(label="Скачать результат (.txt)")

            def _ocr_wrapper(img_path: Optional[str]):
                text, path = ocr(img_path)
                return text, path

            ocr_btn.click(
                _ocr_wrapper,
                inputs=[ocr_image],
                outputs=[ocr_output, ocr_download],
            )

        # Tab 3
        with gr.Tab("Описание по координатам"):
            bbox_image = gr.Image(label="Загрузите изображение", type="filepath")
            coords_input = gr.Textbox(label="Координаты (x1,y1,x2,y2)", placeholder="x1,y1,x2,y2")
            bbox_btn = gr.Button("Получить описание")
            bbox_output = gr.Textbox(label="Описание", interactive=False)

            bbox_btn.click(
                describe_by_bbox,
                inputs=[bbox_image, coords_input],
                outputs=bbox_output,
            )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=APP_PORT)
