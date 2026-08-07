from typing import Literal, Required, TypedDict

type GameStyle = Literal["SMB", "SMB3", "SMW", "NSMBU", "SM3DW"]
type Character = Literal["Mario", "Luigi", "Toad", "Toadette"]

type CourseTag = Literal[
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


class CourseStartData(TypedDict, total=False):
    frame_type: Required[Literal["course_start"]]
    course_code: str
    course_title: str
    course_creator: str
    life_count: int | None


class CourseEndData(TypedDict, total=False):
    frame_type: Required[Literal["course_end"]]
    course_title: str
    course_creator: str
    rating: Literal["like", "boo"] | None
    play_time_ms: int
    world_record_ms: int
    ranking: Literal["world_record", "first_clear"] | None


class GameplayData(TypedDict, total=False):
    frame_type: Required[Literal["gameplay"]]
    game_style: GameStyle
    life_count: int | None


class UnknownData(TypedDict, total=False):
    frame_type: Required[Literal["unknown"]]


type FrameData = CourseStartData | CourseEndData | GameplayData | UnknownData


__all__ = [
    "GameStyle",
    "Character",
    "CourseTag",
    "CourseStartData",
    "CourseEndData",
    "GameplayData",
    "UnknownData",
    "FrameData",
]
