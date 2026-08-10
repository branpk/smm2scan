from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import av
from tqdm import tqdm

from smm2scan._types import *
from smm2scan._analyze_frame import analyze_frame
from smm2scan._util import load_ocr_full, load_ocr_rec


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


def analyze_video(video_file: str | Path) -> list[PartialPlayedCourse]:
    load_ocr_rec()
    load_ocr_full()

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

        prev_scanned_frame_time = -float("inf")
        state = VideoState()

        iter = tqdm(
            container.decode(video_stream),
            total=total_frames,
            desc=video_file.stem,
        )
        for frame in iter:
            assert frame.time is not None
            if frame.time > prev_scanned_frame_time + 0.2:
                prev_scanned_frame_time = frame.time

                img = frame.to_ndarray(format="rgb24")
                frame_data = analyze_frame(img)

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

                iter.set_postfix_str(state.get_status())

        return state.finish()
