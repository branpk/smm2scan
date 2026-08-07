import re
from pathlib import Path
import shutil
import sys
import av
import numpy as np
import cv2
from tqdm import tqdm
import yt_dlp
import os


def download_video(url: str) -> Path:
    base_dir = "data/videos"
    os.makedirs(base_dir, exist_ok=True)

    ydl_params: "yt_dlp._Params" = {
        "paths": {"home": base_dir},
        "outtmpl": {"default": "%(id)s.%(ext)s"},
        "format": "bestvideo[height=360][fps=30]",
    }
    with yt_dlp.YoutubeDL(ydl_params) as ydl:
        info_dict = ydl.extract_info(url)
        return Path(ydl.prepare_filename(info_dict))


def timestamp_to_seconds(s: str) -> float:
    if match := re.fullmatch(r"(\d+):([\d\.]+)", s):
        seconds = int(match.group(1)) * 60
        seconds += float(match.group(2))
    return seconds


def seconds_to_frame(seconds: float) -> int:
    return round(8 + 30 * seconds)


def save_frames(
    group_name: str, url: str, selected_timestamps: dict[int | str, str]
) -> None:
    if group_name not in sys.argv and "all" not in sys.argv:
        return

    video_file = download_video(url)

    base_dir = Path("data/screenshots") / video_file.stem
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    selected_frames = {
        (
            ts if isinstance(ts, int) else seconds_to_frame(timestamp_to_seconds(ts))
        ): name
        for ts, name in selected_timestamps.items()
    }
    end_frame = max(selected_frames, default=0) + 1
    saved_names = set()

    with av.open(video_file) as container:
        for i, frame in tqdm(
            enumerate(container.decode(0)), total=end_frame, desc=group_name
        ):
            if i >= end_frame:
                break
            if name := selected_frames.get(i):
                assert name not in saved_names, f"Duplicate: {name}"

                img = frame.to_ndarray(format="rgb24")
                assert img.dtype == np.uint8
                assert img.shape == (360, 640, 3)

                img_file = base_dir / f"{name}.png"
                cv2.imwrite(img_file, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                img2 = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)  # type: ignore
                assert img.dtype == img2.dtype
                assert img.shape == img2.shape
                assert (img == img2).all()

                saved_names.add(name)

    not_saved = []
    for name in selected_timestamps.values():
        if name not in saved_names:
            not_saved.append(name)
    if not_saved:
        raise Exception(f"Invalid timestamps: {not_saved}")


if len(sys.argv) == 1:
    print("all or group name required")
    sys.exit(1)


save_frames(
    "royru1",
    "http://youtube.com/watch?v=2OJxILaNfp0",
    {
        "00:01.6": "course_start_NSMBU_Luigi",
        "02:22": "course_end_wo_comments",
        "02:30.5": "endless_super_expert",
        "02:33": "course_start_SMB_Luigi",
        "02:39": "course_start_SMB3_Luigi",
        "02:42.5": "course_start_SMW_Luigi",
        "02:45": "pause",
        4936: "pause_pre",
        4937: "pause_partial",
        "02:46": "gameplay_SMW_Luigi",
        "03:29": "gameplay_SMB_Luigi",
        "08:32.5": "course_end_wo_comments_boo_first_clear",
        "08:51": "course_select",
        "11:11.5": "course_end_wo_comments_boo",
        "11:18": "course_select_2",
        "11:24": "endless_super_expert_lightning",
        "15:22": "course_start_SM3DW_Luigi",
        "15:44": "gameplay_NSMBU_Luigi",
        "18:25": "pause_2",
        33134: "pause_2_pre",
        33135: "pause_2_partial_1",
        33139: "pause_2_partial_2",
        "18:33": "gameplay_NSMBU_Luigi_3_1ups",
    },
)

save_frames(
    "panga1",
    "http://youtube.com/watch?v=IXMlYirpyvQ",
    {
        "0:01": "course_start_SM3DW_Mario",
        "0:05": "gameplay_SM3DW_Mario",
        "0:51": "course_end_wo_comments_world_record",
        "0:54": "course_start_SMW_Mario",
        "0:56": "gameplay_SMW_Mario",
        "2:23": "course_end_w_comments_like",
        "2:29": "course_start_SMB3_Mario",
        "2:44": "gameplay_SMB3_Mario",
        "3:49": "gameplay_SMB3_Mario_2_1ups",
        "3:59": "course_end_wo_comments_like",
        "7:17": "course_start_NSMBU_Mario",
        "8:09": "gameplay_NSMBU_Mario",
        "10:38": "course_start_NSMBU_Toad",
        "10:59": "gameplay_NSMBU_Toad",
        "11:05": "course_start_SMW_Toad",
        "11:18": "gameplay_SMW_Toad",
        "12:38": "gameplay_SMW_Toad_2_1ups",
    },
)


save_frames(
    "panga2",
    "http://youtube.com/watch?v=AdDmZytS1fQ",
    {
        "00:12.9": "gameplay_SM3DW_Toadette_timer_dark",
        "16:10": "gameplay_SM3DW_Toadette",
    },
)
