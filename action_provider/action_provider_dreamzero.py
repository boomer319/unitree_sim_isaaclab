from action_provider.action_base import ActionProvider
from typing import Optional
import torch
import numpy as np
import msgpack
import functools
import os

import zmq

ZMQ_RECV_TIMEOUT_MS = int(os.environ.get("DREAMZERO_ZMQ_RECV_TIMEOUT_MS", "300000"))


def _pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class _DreamZeroClient:
    def __init__(self, host: str, port: int):
        self._endpoint = f"tcp://{host}:{port}"
        self._sock = None
        self._ctx = None
        self._metadata = None

    def connect(self):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, ZMQ_RECV_TIMEOUT_MS)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self._endpoint)
        self._sock.send(_packer().pack({"endpoint": "metadata"}))
        self._metadata = _unpackb(self._sock.recv())
        return self._metadata

    def _request(self, payload: dict) -> dict:
        self._sock.send(_packer().pack(payload))
        response = self._sock.recv()
        result = _unpackb(response)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"Server error:\n{result['error']}")
        return result

    def infer(self, obs: dict) -> dict:
        msg = dict(obs)
        msg["endpoint"] = "infer"
        return self._request(msg)

    def save_video(self) -> dict:
        return self._request({"endpoint": "save_video"})

    def reset(self) -> dict:
        return self._request({"endpoint": "reset"})

    def close(self):
        if self._sock:
            try:
                self._sock.close(linger=0)
            except Exception:
                pass


class DreamZeroActionProvider(ActionProvider):
    _ACTION_CHUNK_KEYS = (
        "action.left_arm_pos",
        "action.right_arm_pos",
        "action.left_hand_pos",
        "action.right_hand_pos",
    )

    def __init__(self, env, args_cli):
        super().__init__("DreamZeroActionProvider")
        self.env = env
        self.host = args_cli.dreamzero_host
        self.port = args_cli.dreamzero_port
        self.prompt = args_cli.dreamzero_prompt

        self.all_joint_names = env.scene["robot"].data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}

        self._setup_joint_mapping()

        device = self.env.device

        self._arm_target_indices = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
        self._left_hand_target_indices = [self.joint_to_index[name] for name in self.left_hand_joint_mapping.keys()]
        self._right_hand_target_indices = [self.joint_to_index[name] for name in self.right_hand_joint_mapping.keys()]

        self._arm_target_idx_t = torch.tensor(self._arm_target_indices, dtype=torch.long, device=device)
        self._left_hand_target_idx_t = torch.tensor(self._left_hand_target_indices, dtype=torch.long, device=device)
        self._right_hand_target_idx_t = torch.tensor(self._right_hand_target_indices, dtype=torch.long, device=device)

        self._full_action_buf = torch.zeros(len(self.all_joint_names), device=device, dtype=torch.float32)

        self._arm_source_indices = [self.joint_to_index[name] for name in self.arm_joint_mapping.keys()]
        self._left_hand_source_indices = [self.joint_to_index[name] for name in self.left_hand_joint_mapping.keys()]
        self._right_hand_source_indices = [self.joint_to_index[name] for name in self.right_hand_joint_mapping.keys()]

        self._arm_source_idx_t = torch.tensor(self._arm_source_indices, dtype=torch.long, device=device)
        self._left_hand_source_idx_t = torch.tensor(self._left_hand_source_indices, dtype=torch.long, device=device)
        self._right_hand_source_idx_t = torch.tensor(self._right_hand_source_indices, dtype=torch.long, device=device)

        self._client = None
        self._connected = False
        self._stop_requested = False
        self._infer_count = 0
        self._video_saved = False

        self._call_count = 0
        self._action_queue = []

        self.camera_mapping = {
            "front_left_camera": "cam_left_high",
            "front_right_camera": "cam_right_high",
            "left_wrist_camera": "cam_left_wrist",
            "right_wrist_camera": "cam_right_wrist",
        }
        self.camera_names = list(self.camera_mapping.keys())

    def _setup_joint_mapping(self):
        self.arm_joint_mapping = {
            "left_shoulder_pitch_joint": 0,
            "left_shoulder_roll_joint": 1,
            "left_shoulder_yaw_joint": 2,
            "left_elbow_joint": 3,
            "left_wrist_roll_joint": 4,
            "left_wrist_pitch_joint": 5,
            "left_wrist_yaw_joint": 6,
            "right_shoulder_pitch_joint": 7,
            "right_shoulder_roll_joint": 8,
            "right_shoulder_yaw_joint": 9,
            "right_elbow_joint": 10,
            "right_wrist_roll_joint": 11,
            "right_wrist_pitch_joint": 12,
            "right_wrist_yaw_joint": 13,
        }
        self.left_hand_joint_mapping = {
            "left_hand_thumb_0_joint": 0,
            "left_hand_thumb_1_joint": 1,
            "left_hand_thumb_2_joint": 2,
            "left_hand_middle_0_joint": 3,
            "left_hand_middle_1_joint": 4,
            "left_hand_index_0_joint": 5,
            "left_hand_index_1_joint": 6,
        }
        self.right_hand_joint_mapping = {
            "right_hand_thumb_0_joint": 0,
            "right_hand_thumb_1_joint": 1,
            "right_hand_thumb_2_joint": 2,
            "right_hand_middle_0_joint": 3,
            "right_hand_middle_1_joint": 4,
            "right_hand_index_0_joint": 5,
            "right_hand_index_1_joint": 6,
        }

    def _connect(self):
        try:
            self._client = _DreamZeroClient(host=self.host, port=self.port)
            metadata = self._client.connect()
            self._connected = True
            print(f"[{self.name}] Connected to DreamZero server at {self.host}:{self.port}")
            print(f"[{self.name}] Server metadata: {metadata}")
        except Exception as e:
            print(f"[{self.name}] Failed to connect: {e}")
            self._connected = False

    def start(self):
        self._connect()
        super().start()

    def stop(self):
        if self._stop_requested:
            return
        self._stop_requested = True
        self._connected = False
        if self._client and self._infer_count > 0:
            try:
                print(f"[{self.name}] Requesting DreamZero save_video ...")
                save_response = self._client.save_video()
                print(f"[{self.name}] save_video response: {save_response}")
                if save_response.get("status") == "saved" and save_response.get("path"):
                    self._video_saved = True
                    print(f"[{self.name}] DreamZero video saved: {save_response['path']}")
                else:
                    print(f"[{self.name}] DreamZero video not saved: {save_response}")
            except Exception as e:
                print(f"[{self.name}] Failed to request DreamZero save_video: {e}")
            try:
                reset_response = self._client.reset()
                print(f"[{self.name}] reset response: {reset_response}")
            except Exception as e:
                print(f"[{self.name}] Failed to reset DreamZero: {e}")
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        super().stop()

    def _read_camera_frame(self, camera_name):
        try:
            img = self.env.scene[camera_name].data.output["rgb"][0]
            if img.device.type == 'cpu':
                return img.numpy()
            else:
                return img.cpu().numpy()
        except Exception as e:
            print(f"[{self.name}] Failed to read camera {camera_name}: {e}")
            return None

    def _extract_state_28dim(self, joint_pos):
        arm_vals = joint_pos.index_select(1, self._arm_source_idx_t)[0]
        left_hand_vals = joint_pos.index_select(1, self._left_hand_source_idx_t)[0]
        right_hand_vals = joint_pos.index_select(1, self._right_hand_source_idx_t)[0]

        left_arm = arm_vals[:7].cpu().numpy().astype(np.float64)
        right_arm = arm_vals[7:].cpu().numpy().astype(np.float64)
        left_hand = left_hand_vals.cpu().numpy().astype(np.float64)
        right_hand = right_hand_vals.cpu().numpy().astype(np.float64)

        return left_arm, right_arm, left_hand, right_hand

    def _build_observation_dict(self):
        obs = {}

        for sim_name in self.camera_names:
            img = self._read_camera_frame(sim_name)
            if img is not None:
                obs[f"observation/{self.camera_mapping[sim_name]}"] = img
            else:
                obs[f"observation/{self.camera_mapping[sim_name]}"] = np.zeros((480, 640, 3), dtype=np.uint8)

        joint_pos = self.env.scene["robot"].data.joint_pos
        left_arm, right_arm, left_hand, right_hand = self._extract_state_28dim(joint_pos)

        obs["observation/left_arm_pos"] = left_arm
        obs["observation/right_arm_pos"] = right_arm
        obs["observation/left_hand_pos"] = left_hand
        obs["observation/right_hand_pos"] = right_hand

        obs["prompt"] = self.prompt

        return obs

    def _action_dict_to_28dim(self, action_dict: dict) -> np.ndarray:
        missing = [key for key in self._ACTION_CHUNK_KEYS if key not in action_dict]
        if missing:
            raise ValueError(f"DreamZero action response missing keys: {missing}")
        arrays = []
        for key in self._ACTION_CHUNK_KEYS:
            arr = np.asarray(action_dict[key], dtype=np.float32)
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(-1, 7)
            arrays.append(arr)
        horizon = arrays[0].shape[0]
        if any(arr.shape != (horizon, 7) for arr in arrays):
            raise ValueError(f"DreamZero action chunk shape mismatch: {[arr.shape for arr in arrays]}")
        return np.concatenate(arrays, axis=1)

    def get_action(self, env) -> Optional[torch.Tensor]:
        try:
            if self._action_queue:
                action_28dim = self._action_queue.pop(0)
                full_action = self._full_action_buf
                full_action.zero_()
                full_action.index_copy_(0, self._arm_target_idx_t,
                    torch.tensor(action_28dim[:14], dtype=torch.float32, device=self.env.device))
                full_action.index_copy_(0, self._left_hand_target_idx_t,
                    torch.tensor(action_28dim[14:21], dtype=torch.float32, device=self.env.device))
                full_action.index_copy_(0, self._right_hand_target_idx_t,
                    torch.tensor(action_28dim[21:28], dtype=torch.float32, device=self.env.device))
                return full_action.unsqueeze(0)

            if not self._connected:
                return torch.zeros((1, len(self.all_joint_names)), dtype=torch.float32, device=self.env.device)

            obs = self._build_observation_dict()

            action_dict = self._client.infer(obs)
            self._infer_count += 1
            print(f"[{self.name}] DreamZero inference #{self._infer_count} complete")

            action_chunk = self._action_dict_to_28dim(action_dict)

            if len(action_chunk) > 0:
                action_chunk_list = [action_chunk[i] for i in range(len(action_chunk))]
                self._action_queue = action_chunk_list[1:]

                first_action = action_chunk[0]
                full_action = self._full_action_buf
                full_action.zero_()
                full_action.index_copy_(0, self._arm_target_idx_t,
                    torch.tensor(first_action[:14], dtype=torch.float32, device=self.env.device))
                full_action.index_copy_(0, self._left_hand_target_idx_t,
                    torch.tensor(first_action[14:21], dtype=torch.float32, device=self.env.device))
                full_action.index_copy_(0, self._right_hand_target_idx_t,
                    torch.tensor(first_action[21:28], dtype=torch.float32, device=self.env.device))
                return full_action.unsqueeze(0)

            return None

        except Exception as e:
            print(f"[{self.name}] get_action failed: {e}")
            return None

    def cleanup(self):
        self.stop()
        print(f"[{self.name}] Cleanup complete")
