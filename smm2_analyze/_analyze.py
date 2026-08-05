import os
import re
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


def read_lines_opt(label: str, img: np.ndarray, box: list[int]) -> list[str]:
    subimg = get_box(img, box)
    result = load_ocr().predict(subimg)[0]
    return result["rec_texts"]


def read_lines_exact(
    label: str, img: np.ndarray, box: list[int], expected_count: int
) -> list[str]:
    texts = read_lines_opt(label, img, box)
    count = len(texts)
    if count != expected_count:
        results_txt = "result" if count == 1 else "results"
        raise OCRException(
            f"OCR read failed: {label} -> {count} {results_txt}: {" ".join(map(repr, texts))}"
        )
    return texts


def read_line(label: str, img: np.ndarray, box: list[int]) -> str:
    return read_lines_exact(label, img, box, 1)[0]


def read_int(label: str, img: np.ndarray, box: list[int]) -> int:
    text = read_line(label, img, box).replace("フ", "7")
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


def validate_tags(tags: list[str]) -> list[LevelTag]:
    EN_TAGS: list[LevelTag] = [
        "Standard",
        "Puzzle-solving",
        "Speedrun",
        "Autoscroll",
        "Auto-Mario",
        "Short and Sweet",
        "Multiplayer Versus",
        "Themed",
        "Music",
        "Art",
        "Technical",
        "Shooter",
        "Boss battle",
        "Single player",
        "Link",
    ]
    JP_TAGS = [
        "標準",
        "謎解き",
        "スピードラン",
        "自動スクロール",
        "オートマリオ",
        "短くて甘い",
        "マルチプレイヤー対戦",
        "テーマ別",
        "音楽",
        "美術",
        "テクニカル",
        "シューター",
        "ボス戦",
        "シングルプレーヤー",
        "リンク",
    ]
    validated_tags: list[LevelTag] = []
    for tag in tags:
        if tag == "---":
            continue
        elif tag in EN_TAGS:
            validated_tags.append(tag)
        elif tag in JP_TAGS:
            validated_tags.append(EN_TAGS[JP_TAGS.index(tag)])
        else:
            raise Exception(f"Invalid tag: {repr(tag)}")
    return validated_tags


def validate_clear_condition(lines: list[str]) -> str | None:
    condition = " ".join(lines).strip()
    if not condition:
        return None
    # Separate adjacent letter/digit pairs.
    condition = re.sub(r"(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)", " ", condition)
    if not condition.endswith("."):
        condition += "."
    return condition


def has_life_count(img: np.ndarray) -> bool:
    subimg = get_box(img, [343, 192, 408, 248])
    black_percent = ((subimg**2).sum(axis=-1) < 20).mean().item()
    return black_percent < 0.9


def read_level_start_data(img: np.ndarray) -> LevelStartData:
    return {
        "frame_type": "level_start",
        "level_code": read_line("level_code", img, [42, 89, 116, 100]),
        "level_title": read_line("level_title", img, [40, 40, 600, 70]),
        "level_creator": read_line("level_creator", img, [300, 84, 532, 106]),
        "level_tags": validate_tags(
            read_lines_opt("level_tags", img, [533, 110, 639, 170])
        ),
        "level_condition": validate_clear_condition(
            read_lines_opt("level_condition", img, [185, 265, 510, 310])
        ),
        "life_count": (
            read_int("life_count", img, [333, 192, 408, 248])
            if has_life_count(img)
            else None
        ),
    }


def analyze_frame(img: np.ndarray) -> FrameData:
    assert img.shape == (360, 640, 3)
    assert img.dtype == np.uint8
    if is_level_start(img):
        return read_level_start_data(img)
    else:
        return {"frame_type": "unknown"}


__all__ = ["analyze_frame"]
