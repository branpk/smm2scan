from typing import Literal, TypedDict

type GameStyle = Literal["SMB", "SMB3", "SMW", "NSMBU", "SM3DW"]
type Character = Literal["Mario", "Luigi", "Toad", "Toadette"]

type LevelTag = Literal[
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

LevelStartData = TypedDict(
    "LevelStartData",
    {
        "frame_type": Literal["level_start"],
        "level_code": str,
        "level_title": str,
        "level_creator": str,
        "level_tags": list[LevelTag],
        "level_condition": str | None,
        "life_count": int | None,
    },
)

LevelEndData = TypedDict(
    "LevelEndData",
    {
        "frame_type": Literal["level_end"],
        "level_title": str,
        "level_creator": str,
        "rating": Literal["like", "boo"] | None,
        "play_seconds": int,
        "world_record_seconds": int,
        "ranking": Literal["world_record", "first_clear"] | None,
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
    "LevelTag",
    "LevelStartData",
    "LevelEndData",
    "UnknownData",
    "FrameData",
]
