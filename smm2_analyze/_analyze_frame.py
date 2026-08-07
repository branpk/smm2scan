import re

import numpy as np

from smm2_analyze._types import *
from smm2_analyze._util import *


def validate_course_code(code: str) -> str:
    code = code.upper()
    pattern = r"^[A-Z0-9]{3}-[A-Z0-9]{3}-[A-Z0-9]{3}$"
    if not re.fullmatch(pattern, code):
        raise Exception(f"Invalid course code: {repr(code)}")
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


def read_course_start_data(img: np.ndarray) -> CourseStartData:
    return CourseStartData(
        frame_type="course_start",
        course_code=validate_course_code(read_text(img, [40, 89, 180, 100])),
        course_title=read_text(img, [40, 40, 600, 70]),
        course_creator=read_text(img, [300, 84, 532, 106]),
        life_count=(
            read_int(img, [333, 192, 408, 248])
            if matches_template(img, "course_start_w_lives")
            else None
        ),
    )


def read_course_end_data(img: np.ndarray) -> CourseEndData:
    return CourseEndData(
        frame_type="course_end",
        course_title=read_text(img, [13, 76, 420, 100]),
        course_creator=read_text(img, [400, 105, 565, 122]),
        rating=template_select(  # type: ignore
            img,
            {
                "course_end_wo_comments_like": "like",
                "course_end_w_comments_like": "like",
                "course_end_wo_comments_boo": "boo",
                "course_end_w_comments_boo": "boo",
            },
        ),
        play_time_ms=validate_time(read_text(img, [300, 170, 400, 200])),
        world_record_ms=validate_time(read_text(img, [490, 170, 580, 200])),
        ranking=template_select(  # type: ignore
            img,
            {
                "course_end_wo_comments_first_clear": "first_clear",
                "course_end_w_comments_first_clear": "first_clear",
                "course_end_wo_comments_world_record": "world_record",
                "course_end_w_comments_world_record": "world_record",
            },
        ),
    )


def analyze_frame(img: np.ndarray) -> FrameData:
    assert img.shape == (360, 640, 3)
    assert img.dtype == np.uint8
    read_fn = template_select(
        img,
        {
            "course_start": read_course_start_data,
            "course_end_wo_comments": read_course_end_data,
            "course_end_w_comments": read_course_end_data,
        },
    ) or (lambda img: UnknownData(frame_type="unknown"))
    return read_fn(img)


__all__ = ["analyze_frame"]
