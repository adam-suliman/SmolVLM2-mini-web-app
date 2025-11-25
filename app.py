import os
import time
import warnings
from typing import List, Tuple, Optional

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import gradio as gr

# Determine model and device from environment variables
MODEL_ID = os.getenv("SMOLVLM2_MODEL_ID", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DEVICE_SETTING = os.getenv("MODEL_DEVICE", "auto").lower()

# Resolve device
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

# Choose dtype: bfloat16 on GPU, float32 on CPU
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# Load processor and model once at startup
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=dtype)
model.to(device)

# Allowed extensions for file validation
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4"}

def _validate_file_type(file_path: str, allowed_exts: set, error_message: str) -> None:
    """
    Validate file extension against a set of allowed extensions.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_exts:
        raise gr.Error(error_message)

def _run_model(messages: List[dict], max_new_tokens: int = 128) -> str:
    """
    Apply the chat template and generate a response from the model.
    """
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts[0]

def infer_vqa(file_path: Optional[str], question: str, history: Optional[List[Tuple[str, str]]], state: Optional[str]) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """
    Handle multimodal chat (VQA, captioning, video description).
    Returns updated chat history and remembered file path.
    """
    # If a new file is provided, validate and remember it; otherwise use previous state
    if file_path:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            media_type = "image"
        elif ext in VIDEO_EXTENSIONS:
            media_type = "video"
        else:
            raise gr.Error("Ожидается изображение (PNG/JPEG) или видео (.mp4).")
        current_path = file_path
    else:
        if state:
            current_path = state
            ext = os.path.splitext(current_path)[1].lower()
            media_type = "video" if ext in VIDEO_EXTENSIONS else "image"
        else:
            raise gr.Error("Сначала загрузите изображение или видео.")

    if not question or question.strip() == "":
        raise gr.Error("Пожалуйста, введите вопрос.")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": media_type, "path": current_path},
                {"type": "text", "text": question.strip()},
            ],
        },
    ]
    answer = _run_model(messages)
    if history is None:
        history = []
    history.append((question, answer))
    return history, current_path

def ocr(image_path: Optional[str]) -> Tuple[str, str]:
    """
    Perform OCR on an image and return the recognized text along with a .txt file path.
    """
    if not image_path:
        raise gr.Error("Сначала загрузите изображение.")
    _validate_file_type(image_path, IMAGE_EXTENSIONS, "Ожидается изображение (PNG/JPEG), а не другой файл.")
    prompt = (
        "Считай это задачей OCR. Пожалуйста, перепиши ВСЁ читаемое текстовое содержимое на изображении как простой текст без дополнительных комментариев."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = _run_model(messages, max_new_tokens=256)
    timestamp = int(time.time())
    filename = f"ocr_result_{timestamp}.txt"
    tmp_dir = "/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text, file_path

def describe_by_bbox(image_path: Optional[str], coords_str: str) -> str:
    """
    Describe the object within the provided rectangle coordinates on an image.
    """
    if not image_path:
        raise gr.Error("Сначала загрузите изображение.")
    _validate_file_type(image_path, IMAGE_EXTENSIONS, "Ожидается изображение (PNG/JPEG), а не другой файл.")

    if not coords_str:
        raise gr.Error("Сначала введите координаты.")
    parts = [p.strip() for p in coords_str.split(",")]
    if len(parts) != 4:
        raise gr.Error("Координаты должны быть четырьмя числами через запятую: x1,y1,x2,y2.")
    try:
        x1, y1, x2, y2 = map(float, parts)
    except ValueError:
        raise gr.Error("Координаты должны быть числами: x1,y1,x2,y2.")

    prompt = (
        f"Опиши объект внутри прямоугольника с координатами ({x1}, {y1}, {x2}, {y2}) на изображении. "
        "Дай краткое, но подробное описание."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return _run_model(messages, max_new_tokens=128)

def build_ui() -> gr.Blocks:
    """
    Assemble the Gradio interface with three tabs: chat, OCR, and coordinate description.
    """
    with gr.Blocks(title="Демо SmolVLM2") as demo:
        # Tab 1: Multimodal chat
        with gr.Tab("Мультимодальный чат"):
            state = gr.State(None)
            chat_history = gr.State([])
            upload = gr.File(
                label="Загрузите изображение или видео (.mp4)",
                file_types=["image", "video"],
                type="filepath",
            )
            question = gr.Textbox(label="Ваш вопрос", placeholder="Введите вопрос", lines=1)
            send_btn = gr.Button("Отправить")
            chatbot = gr.Chatbot(label="История чата")
            send_btn.click(
                infer_vqa,
                inputs=[upload, question, chatbot, state],
                outputs=[chatbot, state],
            )
            send_btn.click(
                lambda: "",
                inputs=None,
                outputs=question,
            )

        # Tab 2: OCR
        with gr.Tab("OCR"):
            ocr_image = gr.Image(label="Загрузите изображение", type="filepath")
            ocr_btn = gr.Button("Распознать текст")
            ocr_output = gr.Textbox(label="Распознанный текст", interactive=False)
            ocr_download = gr.File(label="Скачать результат")
            def ocr_wrapper(img):
                text, path = ocr(img)
                return text, path
            ocr_btn.click(
                ocr_wrapper,
                inputs=[ocr_image],
                outputs=[ocr_output, ocr_download],
            )

        # Tab 3: Description by coordinates
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
    port = int(os.getenv("APP_PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port)
