from pathlib import Path
import sys
import time
from typing import TypedDict

import av
import requests
from tqdm import tqdm

from smm2scan._types import *
from smm2scan._analyze_frame import analyze_frame
from smm2scan._util import format_timestamp, load_ocr_full, load_ocr_rec


class PartialPlayedCourse(TypedDict):
    course_id: str | None
    course_start_timestamp_s: float | None
    gameplay_start_timestamp_s: float | None
    gameplay_end_timestamp_s: float | None


class VideoState:
    def __init__(self) -> None:
        self.played_courses: list[PartialPlayedCourse] = []
        self.course_id: str | None = None
        self.course_start_time: float | None = None
        self.gameplay_start_time: float | None = None
        self.latest_gameplay_time: float | None = None

    def _end_course(self) -> None:
        if (
            self.course_id is None
            and self.course_start_time is None
            and self.gameplay_start_time is None
        ):
            return
        self.played_courses.append(
            PartialPlayedCourse(
                course_id=self.course_id,
                course_start_timestamp_s=self.course_start_time,
                gameplay_start_timestamp_s=self.gameplay_start_time,
                gameplay_end_timestamp_s=self.latest_gameplay_time,
            )
        )
        self.course_id = None
        self.course_start_time = None
        self.gameplay_start_time = None
        self.latest_gameplay_time = None

    def set_course_id(self, time: float, course_id: str | None) -> None:
        if course_id is None or self.course_id != course_id:
            self._end_course()
            self.course_id = course_id
            if course_id is not None:
                self.course_start_time = time

    def record_gameplay(self, time: float) -> None:
        if self.gameplay_start_time is None:
            self.gameplay_start_time = time
        self.latest_gameplay_time = time

    def get_status(self) -> str:
        if (
            self.course_id is None
            and self.course_start_time is None
            and self.gameplay_start_time is None
        ):
            return ""
        status = "playing" if self.gameplay_start_time is not None else "starting"
        course_id = self.course_id or "???-???-???"
        return f"{status} {course_id}"

    def finish(self) -> list[PartialPlayedCourse]:
        self._end_course()
        return self.played_courses


def fetch_invalid_course_ids(course_ids: set[str]) -> set[str]:
    backoffs = 0
    invalid_ids: set[str] = set()
    while True:
        id_string = ",".join(course_ids - invalid_ids)
        response = requests.get(
            f"https://tgrcode.com/mm2/level_info_multiple/{id_string}"
        )
        if response.status_code == 200:
            return invalid_ids
        elif response.status_code == 400 and "application/json" in response.headers.get(
            "Content-Type", ""
        ):
            invalid_id = response.json().get("course_id")
            if invalid_id is not None:
                invalid_id = "-".join([invalid_id[:3], invalid_id[3:6], invalid_id[6:]])
                assert invalid_id in course_ids
                invalid_ids.add(invalid_id)
        else:
            if backoffs >= 3:
                return invalid_ids
            else:
                time.sleep(2**backoffs)
                backoffs += 1


def get_similar_course_ids(course_id: str, prefix: str = "") -> list[str]:
    if not course_id:
        return [prefix]
    elif course_id[0] == "S" or course_id[0] == "5":
        return get_similar_course_ids(
            course_id[1:], prefix + "S"
        ) + get_similar_course_ids(course_id[1:], prefix + "5")
    else:
        return get_similar_course_ids(course_id[1:], prefix + course_id[0])


def resolve_course_ids(course_ids: set[str]) -> dict[str, str]:
    print(f"Checking {len(course_ids)} course IDs", file=sys.stderr)

    invalid_course_ids = fetch_invalid_course_ids(set(course_ids))
    invalid_course_id_to_alts = {
        course_id: set(get_similar_course_ids(course_id)) - {course_id}
        for course_id in invalid_course_ids
    }
    invalid_alt_course_ids = fetch_invalid_course_ids(
        set().union(*invalid_course_id_to_alts.values())
    )

    resolve_course_ids = {}
    for course_id in course_ids:
        if course_id not in invalid_course_ids:
            resolve_course_ids[course_id] = course_id
            continue
        for alt_course_id in invalid_course_id_to_alts.get(course_id, []):
            if alt_course_id not in invalid_alt_course_ids:
                resolve_course_ids[course_id] = alt_course_id
                print(f"  {course_id} -> {alt_course_id}", file=sys.stderr)
                break
        else:
            print(f"  {course_id} unfixed", file=sys.stderr)
            resolve_course_ids[course_id] = course_id
    return resolve_course_ids


def sanitize_played_courses(
    partial_played_courses: list[PartialPlayedCourse],
) -> list[PlayedCourse]:
    resolved_course_ids = resolve_course_ids(
        {
            course_id
            for course in partial_played_courses
            if (course_id := course.get("course_id")) is not None
        }
    )

    played_courses = []
    prev_course_id = None
    for played_course in partial_played_courses:
        course_id = played_course.get("course_id")
        start_timestamp_s = played_course.get(
            "course_start_timestamp_s", played_course.get("gameplay_start_timestamp_s")
        )
        if course_id is None or start_timestamp_s is None:
            continue
        course_id = resolved_course_ids[course_id]
        if prev_course_id == course_id:
            continue
        prev_course_id = course_id
        played_courses.append(
            PlayedCourse(course_id=course_id, start_timestamp_s=start_timestamp_s)
        )
    return played_courses


class AnalyzeVideoException(Exception):
    def __init__(self, timestamp: float, cause: BaseException | None = None) -> None:
        super().__init__(f"Error processing frame at {format_timestamp(timestamp)}")
        self.timestamp = timestamp
        self.__cause__ = cause


def analyze_video(
    video_file: str | Path,
) -> tuple[list[PlayedCourse], list[AnalyzeVideoException]]:
    load_ocr_rec()

    video_file = Path(video_file)
    with av.open(video_file) as container:
        video_stream = container.streams.video[0]
        assert video_stream.width == 640
        assert video_stream.height == 360

        if video_stream.frames:
            total_frames = video_stream.frames
        elif (
            video_stream.duration
            and video_stream.time_base
            and video_stream.average_rate
        ):
            total_frames = int(
                video_stream.duration
                * video_stream.time_base
                * video_stream.average_rate
            )
        else:
            total_frames = None

        next_scanned_frame_time = -float("inf")
        state = VideoState()

        iter = tqdm(
            container.decode(video_stream),
            total=total_frames,
            desc=video_file.stem,
        )
        exceptions = []
        for frame in iter:
            assert frame.time is not None
            if frame.time > next_scanned_frame_time:
                img = frame.to_ndarray(format="rgb24")
                try:
                    frame_data = analyze_frame(img, mode="course_id_only")
                except Exception as e:
                    exceptions.append(AnalyzeVideoException(frame.time, e))
                    continue

                if frame_data["frame_type"] == "course_start":
                    course_id = frame_data.get("course_id")
                    assert course_id is not None
                    state.set_course_id(frame.time, course_id)
                elif frame_data["frame_type"] == "course_menu":
                    if frame_data.get("play_button_pressed"):
                        course_id = frame_data.get("course_id")
                        assert course_id is not None
                        state.set_course_id(frame.time, course_id)
                    else:
                        state.set_course_id(frame.time, None)
                elif frame_data["frame_type"] == "course_end":
                    state.set_course_id(frame.time, None)
                elif frame_data["frame_type"] == "gameplay":
                    state.record_gameplay(frame.time)

                next_scanned_frame_time = frame.time + 0.2
                iter.set_postfix_str(state.get_status())

        played_courses = sanitize_played_courses(state.finish())
        return played_courses, exceptions
