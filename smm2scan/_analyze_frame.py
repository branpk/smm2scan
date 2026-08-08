import re

import numpy as np

from smm2scan._types import *
from smm2scan._util import *


def validate_course_id(code: str) -> str:
    code = code.upper()
    pattern = r"^[A-Z0-9]{3}-[A-Z0-9]{3}-[A-Z0-9]{3}$"
    if not re.fullmatch(pattern, code):
        raise Exception(f"Invalid course id: {repr(code)}")
    return code


def find_similar_str[T](items: list[T], value: str) -> T | None:
    canonicalize = lambda s: s.lower().replace("-", "").replace(" ", "")
    canon_items = [canonicalize(x) for x in items]
    try:
        return items[canon_items.index(canonicalize(value))]
    except ValueError:
        return None


def validate_tags(tags: list[str]) -> list[CourseTag]:
    # course_tags=validate_tags(
    #     [
    #         read_text(img, [500, 115, 639, 140]),
    #         read_text(img, [500, 140, 639, 170]),
    #     ]
    # ),
    EN_TAGS: list[CourseTag] = [
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
    validated_tags: list[CourseTag] = []
    for s in tags:
        s = s.strip("-.…·")
        if not s:
            continue
        elif tag := find_similar_str(EN_TAGS, s):
            validated_tags.append(tag)
        elif tag := find_similar_str(JP_TAGS, s):
            validated_tags.append(EN_TAGS[JP_TAGS.index(tag)])
        else:
            raise Exception(f"Invalid tag: {repr(s)}")
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
    if m := re.fullmatch(r"(\d\d)[:;](\d\d)\.(\d\d\d)", s):
        return 60000 * int(m.group(1)) + 1000 * int(m.group(2)) + int(m.group(3))
    raise Exception(f"Invalid time: {repr(s)}")


def validate_life_count(s: str) -> int:
    s = s.strip().lstrip("xX×").replace("フ", "7").replace("G", "6").replace("O", "0")
    try:
        return int(s)
    except ValueError:
        raise Exception(f"Invalid life count: {repr(s)}")


def read_course_start_data(img: np.ndarray) -> CourseStartFrame | None:
    if not matches_template(img, "course_start"):
        return None

    return CourseStartFrame(
        frame_type="course_start",
        course_id=validate_course_id(read_text(img, [40, 89, 180, 100])),
        course_title=read_text(img, [40, 40, 600, 70]),
        course_maker=read_text(img, [300, 84, 532, 106]),
        life_count=(
            validate_life_count(read_text(img, [333, 192, 408, 248]))
            if matches_template(img, "course_start_lives")
            else None
        ),
    )


def read_course_end_data(img: np.ndarray) -> CourseEndFrame | None:
    if matches_template(img, "course_end"):
        pass
    elif matches_template(img, "course_end_shifted"):
        shift = 65
        img = np.roll(img, shift, axis=0)
        img[:shift] = 0
    else:
        return None

    return CourseEndFrame(
        frame_type="course_end",
        course_title=read_text(img, [13, 76, 420, 100]),
        course_maker=read_text(img, [400, 105, 565, 122]),
        rating=(
            "like"
            if matches_template(img, "course_end_like")
            else "boo" if matches_template(img, "course_end_boo") else None
        ),
        play_time_ms=validate_time(read_text(img, [300, 170, 400, 200])),
        world_record_ms=validate_time(read_text(img, [490, 170, 580, 200])),
        ranking=(
            "first_clear"
            if matches_template(img, "course_end_first_clear")
            else (
                "world_record"
                if matches_template(img, "course_end_world_record")
                else None
            )
        ),
    )


def read_gameplay_data(img: np.ndarray) -> GameplayFrame | None:
    shift = 22
    y_bound = 100
    x_bound = 200
    shifted_img = img.copy()
    shifted_img[:y_bound, :x_bound] = np.roll(img[:y_bound, :x_bound], shift, axis=0)
    shifted_img[:shift, :x_bound] = 0

    template_to_style: dict[str, GameStyle] = {
        "gameplay_SM3DW": "SM3DW",
        "gameplay_SMB3": "SMB3",
        "gameplay_SMB3_P": "SMB3",
        "gameplay_SMW": "SMW",
        "gameplay_NSMBU": "NSMBU",
        "gameplay_SMB": "SMB",
    }
    for template, style in template_to_style.items():
        if matches_template(img, template):
            game_style = style
            is_shifted = False
            break
        if matches_template(shifted_img, template):
            game_style = style
            img = shifted_img
            is_shifted = True
            break
    else:
        return None

    return GameplayFrame(
        frame_type="gameplay",
        game_style=game_style,
        life_count=(
            None
            if is_shifted
            else validate_life_count(read_text(img, [38, 16, 84, 34]))
        ),
    )


def read_course_menu_data(img: np.ndarray) -> CourseMenuFrame | None:
    for shift_x, shift_y in [(0, 0), (8, 19), (1, 7)]:
        shifted_img = np.zeros_like(img)
        shifted_img[: img.shape[0] - shift_y, : img.shape[1] - shift_x] = img[
            shift_y:, shift_x:
        ]
        if matches_template(shifted_img, "course_menu"):
            img = shifted_img
            break
    else:
        return None

    return CourseMenuFrame(
        frame_type="course_menu",
        course_title=read_text(img, [158, 70, 500, 90]),
        course_maker=read_text(img, [400, 120, 520, 150]),
        course_id=validate_course_id(read_text(img, [350, 220, 440, 245])),
        play_button_pressed=matches_template(img, "course_menu_play"),
    )


def analyze_frame(img: np.ndarray) -> SMM2Frame:
    assert img.shape == (360, 640, 3)
    assert img.dtype == np.uint8
    return (
        read_course_start_data(img)
        or read_course_end_data(img)
        or read_gameplay_data(img)
        or read_course_menu_data(img)
        or UnknownFrame(frame_type="unknown")
    )


__all__ = ["analyze_frame"]
