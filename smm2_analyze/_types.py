from typing import Literal, TypedDict

type GameStyle = Literal["SMB", "SMB3", "SMW", "NSMBU", "SM3DW"]
type Character = Literal["Mario", "Luigi", "Toad", "Toadette"]

LevelStartData = TypedDict(
    "LevelStartData",
    {
        "frame_type": Literal["level_start"],
        "level_code": str,
        "level_title": str,
        "level_creator": str,
        "level_tags": list[str],
        "level_condition": str | None,
        "game_style": GameStyle,
        "character": Character,
        "life_count": int | None,
    },
)

LevelEndData = TypedDict(
    "LevelEndData",
    {
        "frame_type": Literal["level_end"],
    },
)

UnknownData = TypedDict(
    "UnknownData",
    {
        "frame_type": Literal["unknown"],
    },
)

type FrameData = LevelStartData | LevelEndData | UnknownData


__all__ = [
    "GameStyle",
    "Character",
    "LevelStartData",
    "LevelEndData",
    "UnknownData",
    "FrameData",
]
