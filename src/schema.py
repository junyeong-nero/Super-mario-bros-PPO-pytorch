import base64
import datetime
import io
import json
import os
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class SuperMarioObs:
    time: str = field(
        default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), init=False
    )
    state: dict
    info: dict
    reward: dict
    object_pattern_file: str = field(
        default=os.path.join("assets", "game", "all_object_patterns.json"),
        init=False,
    )
    thresholds: dict = field(
        default_factory=lambda: {
            "0_brick_brown": 0.8,
            "1_question_block_light": 0.8,
            "1_question_block_mild": 0.8,
            "1_question_block_dark": 0.8,
            "2_inactivated_block": 0.8,
            "3_monster_mushroom": 0.6,
            "4_monster_turtle": 0.6,
            "5_pit_1start": 0.8,
            "5_pit_2end": 0.8,
            "6_pipe_green": 0.8,
            "7_item_mushroom_red": 0.8,
            "7_item_mushroom_green": 0.7,
            "8_stair": 0.75,
            "9_flag": 0.8,
        },
        init=False,
    )
    # LazyFrames나 numpy array를 받기 위해 Any 사용
    image: Any = None

    def __post_init__(self):
        # 패턴 파일 로드
        if os.path.exists(self.object_pattern_file):
            with open(self.object_pattern_file, "r") as json_file:
                self.object_patterns = json.load(json_file)
        else:
            # 파일이 없을 경우 빈 딕셔너리로 초기화 (에러 방지)
            self.object_patterns = {}

    def _process_obs_image(self, state_input) -> np.ndarray:
        """
        LazyFrames 또는 (1, H, W, C) 형태의 입력을 (H, W, C) 형태의 uint8 numpy array로 변환
        """
        # 1. LazyFrames -> Numpy 변환
        img = np.array(state_input)

        # 2. Tensor일 경우 처리 (혹시 모를 상황 대비)
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # 3. 차원 축소: (1, 240, 256, 3) -> (240, 256, 3)
        # 첫 번째 차원이 1이면 제거 (Batch 차원 제거)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        # 4. 정규화 확인 및 복원 (0~1 float -> 0~255 uint8)
        if img.dtype != np.uint8:
            if img.max() <= 1.5:  # 정규화된 데이터라고 가정
                img = img * 255.0
            img = img.astype(np.uint8)

        return img

    def save_state_image(self, state):
        # 전처리된 이미지 가져오기 (H, W, C) 형태의 RGB/BGR
        array_255 = self._process_obs_image(state)

        # array_255는 (240, 256, 3) 형태
        image = Image.fromarray(array_255)

        self.time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("screenshot", exist_ok=True)
        image.save(f"screenshot/{self.time}.png")

    def find_objects_in_state(self, state, object_patterns):
        found_objects = {}

        # 1. 메인 이미지 전처리 (H, W, C)
        big_image = self._process_obs_image(state)

        # 이미지 크기 저장 (좌표 계산용)
        img_height = big_image.shape[0]  # 240

        # 2. 템플릿 매칭을 위해 Grayscale로 변환
        if big_image.ndim == 3:
            # Gym env는 보통 RGB로 나오므로 RGB2GRAY 사용 (OpenCV 기본은 BGR이지만 매칭에는 흑백 명도만 중요)
            big_image_gray = cv2.cvtColor(big_image, cv2.COLOR_RGB2GRAY)
        else:
            big_image_gray = big_image

        for object_name, small_object in object_patterns.items():
            # 키 매핑 로직
            if "question_block_" in object_name:
                object_key = "1_question_block"
            elif "item_mushroom" in object_name:
                object_key = "8_item_mushroom"
            else:
                object_key = object_name

            # 이미 찾은 블록 패스
            if (
                "question_block_" in object_name
                and "1_question_block" in found_objects
                and len(found_objects["1_question_block"]) != 0
            ):
                continue

            if object_key not in found_objects:
                found_objects[object_key] = []

            # --- 템플릿(작은 오브젝트) 전처리 ---
            small_object = np.array(small_object)
            if isinstance(small_object, torch.Tensor):
                small_object = small_object.numpy()

            # 템플릿 정규화 복원
            if small_object.max() <= 1.5:
                small_object = small_object * 255.0
            small_object = small_object.astype(np.uint8)

            # 템플릿 차원 정리 (CHW -> HWC 변환 등)
            if small_object.ndim == 3:
                # (C, H, W) 형태인 경우 (H, W, C)로 변경
                if small_object.shape[0] in [1, 3]:
                    small_object = np.transpose(small_object, (1, 2, 0))
                # 흑백 변환
                small_object = cv2.cvtColor(small_object, cv2.COLOR_RGB2GRAY)

            # --- 매칭 수행 ---
            result = cv2.matchTemplate(
                big_image_gray, small_object, cv2.TM_CCOEFF_NORMED
            )

            threshold = self.thresholds.get(object_name, 0.8)
            locations = np.where(result >= threshold)

            for loc in zip(*locations[::-1]):  # Switch to (x, y)
                # Bottom-to-Top 좌표계 변환 (이미지 높이 기준)
                loc = (loc[0], img_height - loc[1])
                found_objects[object_key].append(loc)

        return found_objects

    def to_text(self):
        observation = ""

        if not hasattr(self, "object_patterns") or not self.object_patterns:
            return "Object patterns not loaded or empty."

        found_objects = self.find_objects_in_state(
            self.state["image"], self.object_patterns
        )

        # Mario loc
        x_pos = min(128, self.info.get("x_pos", 0)) - 6
        y_pos = self.info.get("y_pos", 0) - 34
        observation += f"Position of Mario: ({x_pos}, {y_pos})\n"

        # Object Mapping & Labeling (기존 로직 유지)
        object_transform = {
            "brick": lambda x, y: f"({x},{y+1})",
            "question_block": lambda x, y: f"({x-1},{y+1})",
            "inactivated_block": lambda x, y: f"({x-1},{y+1})",
            "monster_mushroom": lambda x, y: f"({x-4},{y+2})",
            "monster_turtle": lambda x, y: f"({x-4},{y+10})",
            "pit_1start": lambda x, y: f"({x+8},{y})",
            "pit_2end": lambda x, y: f"({x+4},{y})",
            "pipe": lambda x, y: f"({x-4},{y},{y-32})",
            "item_mushroom": lambda x, y: f"({x-1},{y+1})",
            "stair": lambda x, y: f"({x},{y})",
            "flag": lambda x, y: f"({x-2},{y})",
        }
        object_label = {
            "brick": "- Bricks: {}",
            "question_block": "- Question Blocks: {}",
            "inactivated_block": "- Inactivated Blocks: {}",
            "monster_mushroom": "- Monster Goomba: {}",
            "monster_turtle": "- Monster Koopas: {}",
            "pit_1start": "- Pit: start at {}",
            "pit_2end": ", end at {}",
            "pipe": "- Warp Pipe: {}",
            "item_mushroom": "- Item Mushrooms: {}",
            "stair": "- Stair Blocks: {}",
            "flag": "- Flag: {}",
        }

        observation += "Positions of all objects\n"
        for object_key, loc_list in found_objects.items():
            for key in object_transform:
                if key in object_key:
                    if not loc_list:
                        locs = "None"
                    else:
                        locs = ", ".join(
                            object_transform[key](x, y) for x, y in loc_list
                        )

                    label = object_label[key].format(locs)

                    if key == "pit_1start":
                        observation += label
                    elif key == "pit_2end":
                        observation += label + "\n"
                    else:
                        observation += label + "\n"
                    break

        observation += "(Note: All (x, y) positions refer to the top-left corner of each object.)\n"
        return observation

    def evaluate(self):
        return int(self.reward.get("distance", 0)), self.reward.get("done", False)


# 사용 방법 예시
"""
# state_next는 (1, 240, 256, 3) 형태의 LazyFrames 객체라고 가정
# info는 Gym 환경에서 반환된 딕셔너리

obs = SuperMarioObs(
    state={"image": state_next},
    image=state_next,
    info=info,
    reward={"distance": info.get("x_pos", 0), "done": done},
)

# 텍스트 관찰값 생성
print(obs.to_text())
"""
