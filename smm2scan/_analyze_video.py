from pathlib import Path

import av
from tqdm import tqdm

from smm2scan._types import *
from smm2scan._analyze_frame import analyze_frame
from smm2scan._util import load_ocr_full, load_ocr_rec


def analyze_video(video_file: str | Path) -> SMM2Video:
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
        for frame in tqdm(
            container.decode(video_stream),
            total=total_frames,
            desc=video_file.stem,
        ):
            assert frame.time is not None
            if frame.time > prev_scanned_frame_time + 0.2:
                prev_scanned_frame_time = frame.time

                img = frame.to_ndarray(format="rgb24")
                frame_data = analyze_frame(img)
                if frame_data["frame_type"] != "unknown":
                    print(frame.time, frame_data)
                    break
