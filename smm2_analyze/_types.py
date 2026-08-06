from typing import Literal, Required, TypedDict

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


class LevelStartData(TypedDict, total=False):
    frame_type: Required[Literal["level_start"]]
    level_code: str
    level_title: str
    level_creator: str
    level_tags: list[LevelTag]
    life_count: int | None


class LevelEndData(TypedDict, total=False):
    frame_type: Required[Literal["level_end"]]
    level_title: str
    level_creator: str
    rating: Literal["like", "boo"] | None
    play_seconds: float
    world_record_seconds: float
    ranking: Literal["world_record", "first_clear"] | None


class UnknownData(TypedDict, total=False):
    frame_type: Required[Literal["unknown"]]


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
