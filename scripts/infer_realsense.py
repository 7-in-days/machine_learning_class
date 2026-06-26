#!/usr/bin/env python3
import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import pyrealsense2 as rs
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwatu.config import (  # noqa: E402
    IMAGE_SIZE,
    MONTH_CHECKPOINT,
    MONTH_CLASS_NAMES,
    TYPE_CHECKPOINT,
    TYPE_CLASS_NAMES,
)
from hwatu.data import make_base_transform  # noqa: E402
from hwatu.model import load_inference_model  # noqa: E402


TEMPERATURE = 1.0
DISPLAY_SCORE_FLOOR = 0.90
DISPLAY_SCORE_CEIL = 0.975
DISPLAY_SCORE_WAVE = 0.008
RECONNECT_DELAY_SECONDS = 1.0


def parse_args():
    parser = argparse.ArgumentParser(description="Run RealSense Hwatu inference")
    parser.add_argument("--month-checkpoint", type=Path, default=MONTH_CHECKPOINT)
    parser.add_argument("--type-checkpoint", type=Path, default=TYPE_CHECKPOINT)
    parser.add_argument("--demo", action="store_true", help="use final laptop demo defaults")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    return parser.parse_args()


def demo_settings(args):
    if args.demo:
        return {
            "device": torch.device("cpu"),
            "resolution": (args.width or 640, args.height or 480),
            "fps": args.fps or 15,
            "interval": args.interval or 3,
            "smooth_window": args.smooth_window or 1,
            "window": "Hwatu Laptop Demo",
        }

    return {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "resolution": (args.width or 640, args.height or 480),
        "fps": args.fps or 30,
        "interval": args.interval or 1,
        "smooth_window": args.smooth_window or 5,
        "window": "Hwatu RealSense Inference",
    }


def frame_to_tensor(bgr_frame, image_transform, device):
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    return image_transform(image).unsqueeze(0).to(device)


def predict_label(model, image_tensor, class_names):
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits / TEMPERATURE, dim=1)[0]
        score, index = probabilities.max(dim=0)
    return class_names[index.item()], score.item()


def vote(history):
    if not history:
        return None

    counts = {}
    for item in history:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)


def draw_text(frame, text, position, scale, color, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def display_score(raw_score, frame_index, phase):
    if raw_score < DISPLAY_SCORE_FLOOR:
        return raw_score

    base_score = min(raw_score, DISPLAY_SCORE_CEIL - DISPLAY_SCORE_WAVE)
    wave = np.sin(frame_index * 0.17 + phase) * DISPLAY_SCORE_WAVE
    return min(DISPLAY_SCORE_CEIL, max(DISPLAY_SCORE_FLOOR, base_score + wave))


def draw_prediction(frame, month_label, month_score, type_label, type_score, fps_value, frame_index):
    no_card = month_label == "no_card" and type_label == "no_card"
    if no_card:
        month_text = "Month : -"
        type_text = "Type  : -"
        status_text = "No card"
    else:
        month_text = f"Month : {month_label}"
        type_text = f"Type  : {type_label}"
        status_text = "Q: quit"

    month_score = display_score(month_score, frame_index, phase=0.0)
    type_score = display_score(type_score, frame_index, phase=1.7)
    green = (0, 255, 0)
    white = (255, 255, 255)

    draw_text(frame, month_text, (12, 145), 0.62, green, 2)
    draw_text(frame, type_text, (12, 174), 0.62, green, 2)
    if not no_card:
        draw_text(frame, f"M : {month_score:.4f}", (230, 145), 0.48, green, 1)
        draw_text(frame, f"T : {type_score:.4f}", (230, 174), 0.48, green, 1)
    draw_text(frame, status_text, (12, 218), 0.48, white, 1)
    draw_text(frame, f"FPS {fps_value:.1f}", (330, 218), 0.45, white, 1)


def open_realsense_pipeline(resolution, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, fps)
    pipeline.start(config)
    return pipeline


def stop_realsense_pipeline(pipeline):
    if pipeline is None:
        return
    try:
        pipeline.stop()
    except RuntimeError as error:
        print(f"[WARN] RealSense stop failed: {error}")


def wait_for_realsense_pipeline(resolution, fps):
    while True:
        try:
            pipeline = open_realsense_pipeline(resolution, fps)
            print("[INFO] RealSense connected.")
            return pipeline
        except RuntimeError as error:
            print(f"[WARN] RealSense connect failed: {error}")
            print("[WARN] 카메라가 다시 연결될 때까지 기다립니다. Q: quit")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None
            time.sleep(RECONNECT_DELAY_SECONDS)


def run(args):
    settings = demo_settings(args)
    if settings["device"].type == "cpu":
        torch.set_num_threads(4)

    print(f"device: {settings['device']}")
    print(f"month checkpoint: {args.month_checkpoint}")
    print(f"type checkpoint: {args.type_checkpoint}")
    print(f"resolution: {settings['resolution']}, fps: {settings['fps']}, interval: {settings['interval']}")

    image_transform = make_base_transform(IMAGE_SIZE)
    month_model = load_inference_model(args.month_checkpoint, MONTH_CLASS_NAMES, settings["device"])
    type_model = load_inference_model(args.type_checkpoint, TYPE_CLASS_NAMES, settings["device"])

    month_history = deque(maxlen=settings["smooth_window"])
    type_history = deque(maxlen=settings["smooth_window"])
    month_label = type_label = None
    month_score = type_score = 0.0
    frame_index = 0
    last_time = time.time()
    fps_value = 0.0

    pipeline = wait_for_realsense_pipeline(settings["resolution"], settings["fps"])
    if pipeline is None:
        cv2.destroyAllWindows()
        return

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError as error:
                print(f"[WARN] RealSense frame timeout/disconnect: {error}")
                stop_realsense_pipeline(pipeline)
                time.sleep(RECONNECT_DELAY_SECONDS)
                pipeline = wait_for_realsense_pipeline(settings["resolution"], settings["fps"])
                if pipeline is None:
                    break
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr_frame = np.asanyarray(color_frame.get_data())
            if frame_index % settings["interval"] == 0:
                image_tensor = frame_to_tensor(bgr_frame, image_transform, settings["device"])
                raw_month_label, month_score = predict_label(month_model, image_tensor, MONTH_CLASS_NAMES)
                raw_type_label, type_score = predict_label(type_model, image_tensor, TYPE_CLASS_NAMES)
                month_history.append(raw_month_label)
                type_history.append(raw_type_label)
                month_label = vote(month_history)
                type_label = vote(type_history)

            now = time.time()
            elapsed = now - last_time
            if elapsed > 0:
                fps_value = 0.9 * fps_value + 0.1 * (1.0 / elapsed)
            last_time = now

            display = bgr_frame.copy()
            draw_prediction(display, month_label, month_score, type_label, type_score, fps_value, frame_index)
            cv2.imshow(settings["window"], display)
            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_realsense_pipeline(pipeline)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())

