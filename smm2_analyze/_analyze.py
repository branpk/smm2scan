import os
from typing import Any

import numpy as np

from smm2_analyze._types import *


class OCRException(Exception):
    pass


def load_ocr() -> Any:
    ocr = globals().get("_ocr")
    if ocr is None:
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        globals()["_ocr"] = ocr
    return ocr


def read_text(label: str, img: np.ndarray, box: list[int]) -> str:
    subimg = get_box(img, box)
    result = load_ocr().predict(subimg)[0]
    count = len(result["rec_texts"])
    if count != 1:
        raise OCRException(f"OCR read failed: {label} -> {count} results")
    return result["rec_texts"][0]


def read_int(label: str, img: np.ndarray, box: list[int]) -> int:
    text = read_text(label, img, box).replace("フ", "7")
    try:
        return int(text)
    except ValueError:
        raise OCRException(f"OCR integer parse failed: {label} -> {text}")


def get_box(img: np.ndarray, box: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0 : y1 + 1, x0 : x1 + 1]


def is_level_start(img: np.ndarray) -> bool:
    background = np.array([1, 1, 1], dtype=np.uint8)
    template = np.tile(background, (360, 640, 1))
    get_box(template, [33, 29, 606, 107])[:] = [255, 202, 5]

    diff = ((img - template) ** 2).sum(axis=-1)[:130]
    percent = (diff < 100).mean().item()
    return percent > 0.7


def read_level_start_data(img: np.ndarray) -> LevelStartData:
    return {
        "frame_type": "level_start",
        "level_code": read_text("level_code", img, [42, 89, 116, 100]),
        "level_title": read_text("level_title", img, [0, 40, 640, 70]),
        "level_creator": read_text("level_creator", img, [300, 84, 640, 106]),
        "level_tags": [],  # TODO
        "level_condition": None,  # TODO
        "game_style": "SMB",  # TODO
        "character": "Mario",  # TODO
        "life_count": read_int("life_count", img, [333, 192, 408, 248]),
    }


def analyze_frame(img: np.ndarray) -> FrameData:
    assert img.shape == (360, 640, 3)
    assert img.dtype == np.uint8
    if is_level_start(img):
        return read_level_start_data(img)
    else:
        return {"frame_type": "unknown"}


__all__ = ["analyze_frame"]
