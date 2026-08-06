import os
import re
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


def validate_level_code(code: str) -> str:
    pattern = r"^[A-Z0-9]{3}-[A-Z0-9]{3}-[A-Z0-9]{3}$"
    if not re.fullmatch(pattern, code):
        raise Exception(f"Invalid level code: {repr(code)}")
    return code


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
        tag = tag.strip("-.")
        if not tag:
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


def validate_time(s: str) -> int:
    if m := re.fullmatch(r"(\d\d):(\d\d)\.(\d\d\d)", s):
        return 60000 * int(m.group(1)) + 1000 * int(m.group(2)) + int(m.group(3))
    raise Exception(f"Invalid time: {repr(s)}")


def is_level_start(img: np.ndarray) -> bool:
    background = np.array([1, 1, 1], dtype=np.uint8)
    template = np.tile(background, (360, 640, 1))
    get_box(template, [33, 29, 606, 107])[:] = [255, 202, 5]

    diff = ((img.astype(np.float32) - template) ** 2).sum(axis=-1)[:130]
    percent = (diff < 100).mean().item()
    return percent > 0.7


def level_start_has_life_count(img: np.ndarray) -> bool:
    subimg = get_box(img, [343, 192, 408, 248])
    black_percent = ((subimg.astype(np.float32) ** 2).sum(axis=-1) < 20).mean().item()
    return black_percent < 0.9


def read_level_start_data(img: np.ndarray) -> LevelStartData:
    return LevelStartData(
        frame_type="level_start",
        level_code=validate_level_code(read_text(img, [42, 89, 116, 100])),
        level_title=read_text(img, [40, 40, 600, 70]),
        level_creator=read_text(img, [300, 84, 532, 106]),
        level_tags=validate_tags(
            [
                read_text(img, [500, 115, 639, 140]),
                read_text(img, [500, 140, 639, 170]),
            ]
        ),
        life_count=(
            read_int(img, [333, 192, 408, 248])
            if level_start_has_life_count(img)
            else None
        ),
    )


def is_level_end(img: np.ndarray) -> bool:
    # TODO: Version with comments
    template = np.zeros((360, 640, 3), dtype=np.uint8)
    get_box(template, [0, 0, 640, 65])[:] = [34, 46, 112]
    get_box(template, [0, 66, 640, 291])[:] = [255, 202, 5]
    get_box(template, [0, 292, 640, 360])[:] = [0, 59, 87]
    get_box(template, [200, 245, 400, 285])[:] = [254, 254, 254]
    get_box(template, [410, 245, 620, 285])[:] = [254, 254, 254]

    mask = np.zeros((360, 640), dtype=np.bool)
    mask[:, 240:400] = True
    mask[:65, :] = True
    mask[292:, :] = True

    diff = np.abs(img.astype(np.float32) - template).max(axis=-1)
    percent = ((diff < 40) | ~mask).mean().item()
    return percent > 0.85


# TODO: For Like and Boo, check for presence of something rather than absence


def level_end_has_like(img: np.ndarray) -> bool:
    subimg = get_box(img, [45, 153, 82, 180])
    dist = np.abs(subimg.astype(np.float32) - [93, 86, 190]).max(axis=-1)
    percent = (dist < 40).mean().item()
    return percent < 0.15


def level_end_has_boo(img: np.ndarray) -> bool:
    subimg = get_box(img, [146, 156, 181, 184])
    dist = np.abs(subimg.astype(np.float32) - [234, 98, 93]).max(axis=-1)
    percent = (dist < 40).mean().item()
    return percent < 0.15


def level_end_is_first_clear(img: np.ndarray) -> bool:
    subimg = get_box(img, [431, 137, 572, 160])
    dist = np.abs(subimg.astype(np.float32) - [118, 188, 0]).max(axis=-1)
    percent = (dist < 40).mean().item()
    return percent > 0.5


def level_end_is_world_record(img: np.ndarray) -> bool:
    subimg = get_box(img, [431, 137, 572, 160])
    dist = np.abs(subimg.astype(np.float32) - [232, 72, 64]).max(axis=-1)
    percent = (dist < 40).mean().item()
    return percent > 0.5


def read_level_end_data(img: np.ndarray) -> LevelEndData:
    return LevelEndData(
        frame_type="level_end",
        level_title=read_text(img, [13, 76, 420, 100]),
        level_creator=read_text(img, [400, 105, 565, 120]),
        rating=(
            "like"
            if level_end_has_like(img)
            else "boo" if level_end_has_boo(img) else None
        ),
        play_time_ms=validate_time(read_text(img, [300, 170, 400, 200])),
        world_record_ms=validate_time(read_text(img, [490, 170, 580, 200])),
        ranking=(
            "first_clear"
            if level_end_is_first_clear(img)
            else "world_record" if level_end_is_world_record(img) else None
        ),
    )


def analyze_frame(img: np.ndarray) -> FrameData:
    assert img.shape == (360, 640, 3)
    assert img.dtype == np.uint8
    if is_level_start(img):
        return read_level_start_data(img)
    elif is_level_end(img):
        return read_level_end_data(img)
    else:
        return UnknownData(frame_type="unknown")


__all__ = ["analyze_frame"]
