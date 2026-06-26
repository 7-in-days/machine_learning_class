#!/usr/bin/env python3
import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwatu.config import IMAGE_EXTENSIONS, data_paths  # noqa: E402


CAPTURE_INTERVAL = 0.5
FPS = 30
RESOLUTION = (640, 480)
WINDOW_NAME = "RealSense Hwatu Capture"
NO_CARD_FOLDER = "no_card"

TYPE_CHOICES = {
    "1": ("gwang", "광"),
    "2": ("tti", "띠"),
    "3": ("yeolggeut", "열끗"),
    "4": ("pi", "피"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Capture Hwatu images with RealSense")
    parser.add_argument("--root", default=None, help="project/data root; defaults to HWATU_ROOT or repo root")
    return parser.parse_args()


def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_card_rows(labels_csv):
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv}")

    with labels_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    required = {"folder_path", "card_id", "month", "type", "type_id", "description"}
    columns = set(reader.fieldnames or [])
    missing = required - columns
    if missing:
        raise ValueError(f"labels.csv missing columns: {sorted(missing)}")
    return rows


def no_card_choice():
    return {
        "folder_path": NO_CARD_FOLDER,
        "card_id": NO_CARD_FOLDER,
        "description": "no card",
        "month": "no_card",
        "type": "no_card",
        "type_id": "0",
    }


def print_type_choices():
    print("type select:")
    print(" 1: 광")
    print(" 2: 띠")
    print(" 3: 열끗")
    print(" 4: 피")


def matching_rows(card_rows, month, card_type):
    return [
        row for row in card_rows
        if row["month"].isdigit() and int(row["month"]) == month and row["type"] == card_type
    ]


def make_group_choice(month, card_type, type_label, matches):
    folder_path = f"{month:02d}/{card_type}"
    type_id = next(key for key, value in TYPE_CHOICES.items() if value[0] == card_type)
    descriptions = ", ".join(row["description"] for row in matches)
    return {
        "folder_path": folder_path,
        "card_id": folder_path,
        "description": descriptions or f"{month}월 {type_label}",
        "month": str(month),
        "type": card_type,
        "type_id": type_id,
    }


def select_month():
    while True:
        value = input("month select [1-12, n/b=no_card]: ").strip().lower()
        if value in {"n", "b", "no_card"}:
            return None
        if value.isdigit() and 1 <= int(value) <= 12:
            return int(value)
        print("1~12 숫자를 입력하거나 no_card는 n 또는 b를 입력하세요.")


def select_type():
    while True:
        print_type_choices()
        value = input("type select [1-4]: ").strip()
        if value in TYPE_CHOICES:
            return TYPE_CHOICES[value]
        print("1, 2, 3, 4 중 하나를 입력하세요.")


def print_valid_types(card_rows, month):
    valid_types = []
    for row in card_rows:
        if row["month"].isdigit() and int(row["month"]) == month and row["type"] not in valid_types:
            valid_types.append(row["type"])
    print(f"{month}월에 있는 타입: {', '.join(valid_types)}")


def select_card(card_rows):
    while True:
        month = select_month()
        if month is None:
            return no_card_choice()

        card_type, type_label = select_type()
        matches = matching_rows(card_rows, month, card_type)
        if matches:
            selected = make_group_choice(month, card_type, type_label, matches)
            print(f"[SELECTED] {selected['folder_path']} {selected['description']}")
            return selected

        print(f"{month}월에는 {type_label} 타입이 없습니다.")
        print_valid_types(card_rows, month)


def count_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def make_session_name():
    session = input("session name [default session]: ").strip()
    return (session or "session").replace(" ", "_")


def append_capture_label(paths, save_path, choice, session_name, timestamp, mode):
    captures_csv = paths["captures_csv"]
    ensure_folder(captures_csv.parent)
    fieldnames = [
        "image_path",
        "filename",
        "session",
        "timestamp",
        "folder_path",
        "card_id",
        "month",
        "type",
        "type_id",
        "description",
        "mode",
    ]
    needs_header = not captures_csv.exists() or captures_csv.stat().st_size == 0
    with captures_csv.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "image_path": save_path.relative_to(paths["data_dir"]),
                "filename": save_path.name,
                "session": session_name,
                "timestamp": timestamp,
                "folder_path": choice["folder_path"],
                "card_id": choice["card_id"],
                "month": choice["month"],
                "type": choice["type"],
                "type_id": choice["type_id"],
                "description": choice["description"],
                "mode": mode,
            }
        )


def save_sample(paths, color_image, choice, session_name, mode):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_folder = paths["raw_dir"] / choice["folder_path"]
    ensure_folder(save_folder)
    save_path = save_folder / f"{session_name}_{timestamp}.jpg"
    if not cv2.imwrite(str(save_path), color_image):
        raise RuntimeError(f"image save failed: {save_path}")
    append_capture_label(paths, save_path, choice, session_name, timestamp, mode)
    return save_path


def draw_status(display, choice, count, session_count, auto_status, session_name):
    cv2.putText(display, f"FOLDER: {choice['folder_path']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(display, f"LABEL: month={choice['month']} type={choice['type']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
    cv2.putText(display, f"DESC: {choice['description']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 0), 2)
    cv2.putText(display, f"AUTO: {auto_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(display, f"COUNT: {count}   SESSION SAVED: {session_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
    cv2.putText(display, f"SESSION: {session_name}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)
    cv2.putText(display, "A:auto/manual  S:save  Enter:select again  B:no_card  Q:quit", (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def print_selection_guide(reason):
    print("\n" + "=" * 60)
    print(f"[SELECT MODE] {reason}")
    print("카메라 창이 아니라 이 터미널에 값을 입력하세요.")
    print("=" * 60)


def main(args):
    paths = data_paths(args.root)
    print(f"[DATA ROOT] {paths['root']}")
    print(f"[LABELS] {paths['labels_csv']}")
    print(f"[CAPTURES] {paths['captures_csv']}")

    card_rows = load_card_rows(paths["labels_csv"])
    ensure_folder(paths["raw_dir"])
    session_name = make_session_name()
    current_choice = select_card(card_rows)
    current_count = count_images(paths["raw_dir"] / current_choice["folder_path"])
    session_count = 0

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RESOLUTION[0], RESOLUTION[1], rs.format.bgr8, FPS)
    pipeline.start(config)

    last_capture_time = time.time()
    auto_capture = False
    auto_paused = False

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError as error:
                print(f"[WARN] RealSense frame timeout: {error}")
                time.sleep(0.5)
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            display = color_image.copy()
            auto_status = "PAUSED" if auto_paused else ("ON" if auto_capture else "OFF")
            draw_status(display, current_choice, current_count, session_count, auto_status, session_name)
            cv2.imshow(WINDOW_NAME, display)

            now = time.time()
            if auto_capture and not auto_paused and now - last_capture_time >= CAPTURE_INTERVAL:
                save_path = save_sample(paths, color_image, current_choice, session_name, "auto")
                current_count += 1
                session_count += 1
                last_capture_time = now
                print(f"[AUTO SAVED] {save_path}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto_capture = not auto_capture
                auto_paused = False
                last_capture_time = time.time()
                print(f"[MODE] {'AUTO' if auto_capture else 'MANUAL'}")
            elif key in [10, 13]:
                was_auto_capture = auto_capture
                auto_capture = False
                auto_paused = True
                cv2.destroyWindow(WINDOW_NAME)
                cv2.waitKey(1)
                print_selection_guide("월/타입을 다시 선택합니다.")
                current_choice = select_card(card_rows)
                current_count = count_images(paths["raw_dir"] / current_choice["folder_path"])
                auto_capture = was_auto_capture
                auto_paused = False
                last_capture_time = time.time()
            elif key == ord("s"):
                save_path = save_sample(paths, color_image, current_choice, session_name, "manual")
                current_count += 1
                session_count += 1
                print(f"[SAVED] {save_path}")
            elif key == ord("b"):
                auto_capture = False
                auto_paused = True
                current_choice = no_card_choice()
                current_count = count_images(paths["raw_dir"] / current_choice["folder_path"])
                auto_paused = False
                print("[SELECTED] no_card")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_args())

