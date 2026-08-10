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


class CourseStartFrame(TypedDict, total=False):
    frame_type: Required[Literal["course_start"]]
    course_id: str
    course_title: str
    course_maker: str
    life_count: int | None


class CourseEndFrame(TypedDict, total=False):
    frame_type: Required[Literal["course_end"]]
    course_title: str
    course_maker: str
    rating: Literal["like", "boo"] | None
    play_time_ms: int
    world_record_ms: int
    ranking: Literal["world_record", "first_clear"] | None


class GameplayFrame(TypedDict, total=False):
    frame_type: Required[Literal["gameplay"]]
    game_style: GameStyle
    life_count: int | None


class CourseMenuFrame(TypedDict, total=False):
    frame_type: Required[Literal["course_menu"]]
    course_title: str
    course_maker: str
    course_id: str
    play_button_pressed: bool


class UnknownFrame(TypedDict, total=False):
    frame_type: Required[Literal["unknown"]]


type SMM2Frame = CourseStartFrame | CourseEndFrame | GameplayFrame | CourseMenuFrame | UnknownFrame


class PlayedCourse(TypedDict):
    course_id: str | None
    course_id_source: Literal["course_start", "course_menu"] | None
    start_timestamp_s: float
    end_timestamp_s: float


class SMM2Video(TypedDict):
    video_url: str
    played_courses: list[PlayedCourse]


__all__ = [
    "GameStyle",
    "Character",
    "CourseTag",
    "CourseStartFrame",
    "CourseEndFrame",
    "GameplayFrame",
    "CourseMenuFrame",
    "UnknownFrame",
    "SMM2Frame",
    "PlayedCourse",
    "SMM2Video",
]
