# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
camera state
"""     

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import sys
import os
import threading
import queue

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
multi_image_writer = MultiImageWriter()

def set_writer_options(enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
    try:
        multi_image_writer.set_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=skip_cvtcolor)
        print(f"[camera_state] writer options: jpeg={enable_jpeg}, quality={jpeg_quality}, skip_cvtcolor={skip_cvtcolor}")
    except Exception as e:
        print(f"[camera_state] failed to set writer options: {e}")


_camera_allowlist = None  # None = allow all


def set_camera_allowlist(names: list):
    global _camera_allowlist
    _camera_allowlist = set(names) if names else None


_camera_cache = {
    'available_cameras': None,
    'camera_keys': None,
    'last_scene_id': None,
    'frame_step': 0,
    'write_interval_steps': 2,
}


_return_placeholder = None
_async_queue = None
_async_thread = None
_async_started = False

def _async_writer_loop(q: "queue.Queue", writer: MultiImageWriter):
    while True:
        try:
            item = q.get()
            if item is None:
                break
            writer.write_images(item)
        except Exception as e:
            print(f"[camera_state] Async writer error: {e}")

def _ensure_async_started():
    global _async_started, _async_queue, _async_thread
    if not _async_started:
        _async_queue = queue.Queue(maxsize=1)
        _async_thread = threading.Thread(target=_async_writer_loop, args=(_async_queue, multi_image_writer), daemon=True)
        _async_thread.start()
        _async_started = True


def _extract(env, key: str):
    """Return a HWC uint8 numpy array from an Isaac Lab camera sensor, or None on failure."""
    try:
        img = env.scene[key].data.output["rgb"][0]
        return img.numpy() if img.device.type == 'cpu' else img.cpu().numpy()
    except Exception as e:
        print(f"[camera_state] Failed to read {key}: {e}")
        return None


def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    """Get multiple camera images and write them to shared memory.

    Supports both the legacy monocular head pipeline (front_camera → \"head\")
    and the new stereo head pipeline (front_left_camera → \"head_left\",
    front_right_camera → \"head_right\").  Wrist cameras are unchanged.

    Args:
        env: ManagerBasedRLEnv instance

    Returns:
        torch.Tensor: placeholder tensor (callers ignore the return value)
    """
    global _return_placeholder
    if _return_placeholder is None:
        _return_placeholder = torch.zeros((1, 480, 640, 3))

    _camera_cache['frame_step'] = (_camera_cache['frame_step'] + 1) % max(1, _camera_cache['write_interval_steps'])

    scene_id = id(env.scene)
    if _camera_cache['last_scene_id'] != scene_id:
        _camera_cache['camera_keys'] = list(env.scene.keys())
        _camera_cache['available_cameras'] = [name for name in _camera_cache['camera_keys'] if "camera" in name.lower()]
        _camera_cache['last_scene_id'] = scene_id

    if _camera_cache['frame_step'] == 0:
        try:
            dt = getattr(env, 'physics_dt', 0.02)
            if hasattr(env.scene, 'sensors') and env.scene.sensors:
                for sensor in env.scene.sensors.values():
                    try:
                        sensor.update(dt, force_recompute=False)
                    except Exception:
                        pass
        except Exception:
            pass

    # get the camera images
    images = {}
    # env.sim.render()
    
    camera_keys = _camera_cache['camera_keys']

    # ── Stereo head cameras ────────────────────────────────────
    if "front_left_camera" in camera_keys:
        img = _extract(env, "front_left_camera")
        if img is not None:
            images["head_left"] = img

    if "front_right_camera" in camera_keys:
        img = _extract(env, "front_right_camera")
        if img is not None:
            images["head_right"] = img

    # ── Legacy monocular head camera (kept for backward compatibility) ─────────
    if "front_camera" in camera_keys and "head_left" not in images:
        img = _extract(env, "front_camera")
        if img is not None:
            images["head"] = img

    # ── Wrist cameras ─────────────────────────────────────────────
    if "left_wrist_camera" in camera_keys:
        img = _extract(env, "left_wrist_camera")
        if img is not None:
            images["left"] = img

    if "right_wrist_camera" in camera_keys:
        img = _extract(env, "right_wrist_camera")
        if img is not None:
            images["right"] = img

    # ── Fallback: auto-assign first available cameras if nothing matched ───────
    if not images:
        available_cameras = _camera_cache['available_cameras']
        if available_cameras:
            print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")
            fallback_keys = ["head_left", "head_right", "left", "right"]
            for i, camera_name in enumerate(available_cameras[:4]):
                img = _extract(env, camera_name)
                if img is not None:
                    images[fallback_keys[i]] = img

    # ── Write to shared memory (async, frame-skipped) ─────────────────────────
    if images and _camera_cache['frame_step'] == 0:
        _ensure_async_started()
        try:
            if _async_queue.full():
                _async_queue.get_nowait()
            _async_queue.put_nowait(images)
        except Exception:
            pass
    elif not images:
        print("[camera_state] No camera images found in the environment")

    return _return_placeholder