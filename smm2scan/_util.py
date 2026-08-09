import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yt_dlp

from smm2scan._types import *


def download_video(base_dir: str | Path, url: str) -> Path:
    base_dir = str(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    ydl_params: "yt_dlp._Params" = {
        "paths": {"home": base_dir},
        "outtmpl": {"default": "%(id)s.%(ext)s"},
        "format": "bestvideo[height=360][fps=30]",
    }
    with yt_dlp.YoutubeDL(ydl_params) as ydl:
        info_dict = ydl.extract_info(url)
        video_file = Path(ydl.prepare_filename(info_dict))

    assert info_dict.get("width") == 640
    assert info_dict.get("height") == 360
    assert info_dict.get("fps") == 30
    return video_file


class OCRException(Exception):
    pass


def load_ocr_rec() -> Any:
    ocr = globals().get("_ocr_rec")
    if ocr is None:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import TextRecognition

        ocr = TextRecognition(
            device="cpu",
            model_name="PP-OCRv5_server_rec",
        )
        globals()["_ocr_rec"] = ocr
    return ocr


def load_ocr_full() -> Any:
    ocr = globals().get("_ocr_full")
    if ocr is None:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            device="cpu",
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        globals()["_ocr_full"] = ocr
    return ocr


def preprocess_text_img(img: np.ndarray) -> np.ndarray:
    background = img[:, 0].mean(axis=0).round().clip(0, 255)

    padding = 12
    img = cv2.copyMakeBorder(
        img,
        padding,
        padding,
        padding,
        padding,
        borderType=cv2.BORDER_REPLICATE,
    )

    is_background = ((img.astype(np.float32) - background) ** 2).sum(axis=-1) < 140
    is_background_column = is_background.all(axis=0)
    is_background_row = is_background.all(axis=1)

    buffer = 10
    column_indices = np.nonzero(~is_background_column)[0]
    if len(column_indices) >= 2:
        x0 = max(column_indices[0] - buffer, 0)
        x1 = column_indices[-1] + buffer
    else:
        x0 = 0
        x1 = img.shape[1]
    row_indices = np.nonzero(~is_background_row)[0]
    if len(row_indices) >= 2:
        y0 = max(np.nonzero(~is_background_row)[0][0] - buffer, 0)
        y1 = np.nonzero(~is_background_row)[0][-1] + buffer
    else:
        y0 = 0
        y1 = img.shape[1]
    img = img[y0:y1, x0:x1]

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.stack([img] * 3, axis=-1)
    return img


def read_text(img: np.ndarray, box: list[int]) -> str:
    subimg = preprocess_text_img(get_box(img, box))
    result = load_ocr_rec().predict(subimg)
    return result[0]["rec_text"].strip()


def get_box(img: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0 : y1 + 1, x0 : x1 + 1]


def read_img(file: str | Path) -> np.ndarray:
    file = Path(file)
    if not file.exists():
        raise Exception(f"File not found: {file}")
    img = cv2.imread(file)
    if img is None:
        raise Exception(f"Failed to read image: {file}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_img(file: str | Path, img: np.ndarray) -> None:
    if not cv2.imwrite(file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        raise Exception(f"Failed to write image: {file}")


template_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def matches_template(
    img: np.ndarray,
    template_name: str,
    pixel_threshold: int = 40,
    percent_threshold: float = 0.8,
) -> bool:
    template_data = template_cache.get(template_name)
    if template_data is None:
        template = read_img(f"templates/{template_name}.png")
        mask = np.flatnonzero((template != [169, 69, 169]).all(axis=-1))
        template_data = (template.reshape(-1, 3)[mask].astype(np.int16), mask)
        template_cache[template_name] = template_data
    template, mask = template_data

    # diff = np.abs(img.reshape(-1, 3)[mask] - template)
    # matches = (diff < pixel_threshold).all(axis=-1)
    # percent = matches.mean()

    # Micro-optimized version:
    px = img.reshape(-1, 3)[mask]
    matches = (
        (np.abs(px[:, 0] - template[:, 0]) < pixel_threshold)
        & (np.abs(px[:, 1] - template[:, 1]) < pixel_threshold)
        & (np.abs(px[:, 2] - template[:, 2]) < pixel_threshold)
    )
    percent = np.count_nonzero(matches) / matches.size

    return bool(percent > percent_threshold)
