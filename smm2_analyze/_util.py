import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from smm2_analyze._types import *


class OCRException(Exception):
    pass


def load_ocr() -> Any:
    ocr = globals().get("_ocr")
    if ocr is None:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import TextRecognition

        ocr = TextRecognition(
            device="cpu",
            model_name="PP-OCRv5_server_rec",
        )
        globals()["_ocr"] = ocr
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
    result = load_ocr().predict(subimg)
    return result[0]["rec_text"].strip()


def read_int(img: np.ndarray, box: list[int]) -> int:
    text = read_text(img, box).replace("フ", "7")
    try:
        return int(text)
    except ValueError:
        raise OCRException(f"OCR integer parse failed: {text}")


def get_box(img: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0 : y1 + 1, x0 : x1 + 1]


template_cache: dict[str, np.ndarray] = {}


def matches_template(
    img: np.ndarray,
    template_name: str,
    pixel_threshold: float = 40.0,
    percent_threshold: float = 0.8,
) -> bool:
    template = template_cache.get(template_name)
    if template is None:
        path = Path(f"templates/{template_name}.png")
        assert path.exists(), f"Missing file: {path}"
        template = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)  # type: ignore
        template_cache[template_name] = template

    mask = (template != [169, 69, 169]).all(axis=-1)
    matches = np.abs(img - template.astype(np.float32)).max(axis=-1) < pixel_threshold
    percent = matches[mask].mean()
    return bool(percent > percent_threshold)
