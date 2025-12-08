import os
import json
import datetime
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict, model_validator


class SuperMarioObs(BaseModel):
    """
    Pydantic V2 모델로 변환된 SuperMarioObs
    Obs 클래스가 믹스인(Mixin)이나 추상 클래스라면 (BaseModel, Obs) 형태로 상속받으세요.
    """

    # Numpy, Torch, PIL 객체 등을 필드로 허용하기 위한 설정
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 기본 필드 정의
    state: Dict[str, Any]
    info: Dict[str, Any]  # 예: {'coins': 0, ...}
    reward: Dict[str, Any]

    # init=False였던 필드들은 default_factory를 사용해 자동 생성되도록 설정
    time: str = Field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    object_pattern_file: str = Field(
        default_factory=lambda: os.path.join(
            "assets", "game", "all_object_patterns.json"
        )
    )

    thresholds: Dict[str, float] = Field(
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
        }
    )

    image: Optional[Image.Image] = None

    # __post_init__에서 로드되던 데이터를 저장할 필드
    # 초기화 시 입력받지 않으므로 default_factory=dict 로 설정 (혹은 exclude=True 사용 가능)
    object_patterns: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def load_patterns(self):
        """기존 __post_init__의 역할을 수행합니다."""
        if os.path.exists(self.object_pattern_file):
            with open(self.object_pattern_file, "r") as json_file:
                self.object_patterns = json.load(json_file)
        return self

    def save_state_image(self, state):
        state = torch.FloatTensor(state)[0]
        state = state * 255.0
        array_255 = state.numpy().astype(
            np.uint8
        )  # LeftTop to RightBottom; from (0,0,c) -> (-y, x, c)

        image = Image.fromarray(array_255)

        self.time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 디렉토리가 없으면 에러가 날 수 있으므로 체크/생성 로직이 있으면 좋습니다.
        os.makedirs("screenshot", exist_ok=True)
        image.save(f"screenshot/{self.time}.png")

    def find_objects_in_state(self, state, object_patterns):
        found_objects = {}  # Dictionary to store found objects

        # state to np.array
        state = torch.FloatTensor(state)[0]
        state = state * 255.0
        big_image = state.numpy().astype(np.uint8)

        # Pydantic 모델 내 메서드이므로 self.object_patterns 접근 가능하지만,
        # 인자로 받은 object_patterns를 사용하는 기존 로직을 유지합니다.

        if big_image.ndim == 3:
            big_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)  # (240, 256)

        for object_name, small_object in object_patterns.items():
            if "question_block_" in object_name:
                object_key = "1_question_block"
            elif "item_mushroom" in object_name:
                object_key = "8_item_mushroom"
            else:
                object_key = object_name

            if (
                "question_block_" in object_name
                and "1_question_block" in found_objects.keys()
            ):
                if len(found_objects["1_question_block"]) != 0:
                    continue

            if object_key not in found_objects:
                found_objects[object_key] = []

            # object to np.array
            small_object = torch.FloatTensor(small_object)[0]
            small_object = small_object * 255.0
            small_object = np.array(small_object, dtype=np.uint8)
            if small_object.ndim == 3 and small_object.shape[0] in [1, 3]:
                small_object = np.transpose(
                    small_object, (1, 2, 0)
                )  # Convert to HWC if in CHW
            if small_object.ndim == 3:
                small_object = cv2.cvtColor(small_object, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            result = cv2.matchTemplate(big_image, small_object, cv2.TM_CCOEFF_NORMED)

            # Check if the object is in the image
            # thresholds 키가 없는 경우에 대한 예외 처리가 필요할 수 있습니다.
            threshold = self.thresholds.get(object_name, 0.8)
            locations = np.where(result >= threshold)

            for loc in zip(*locations[::-1]):  # Switch to (x, y)
                loc = (loc[0], 240 - loc[1])  # Bottom-to-Top
                # print(f"Object '{object_name}' found at location: {loc}")
                found_objects[object_key].append(loc)

        print("self.time: ", self.time)
        print("found_objects: ", found_objects)
        return found_objects

    def to_text(self):
        observation = ""

        # self.save_state_image(self.state['image'])
        # Pydantic에서는 dict 접근 대신 self.state['image'] 그대로 사용 가능 (state가 dict이므로)
        found_objects = self.find_objects_in_state(
            self.state["image"], self.object_patterns
        )

        # Mario loc
        x_pos = min(128, self.info["x_pos"]) - 6  # 6: adjusting value
        y_pos = self.info["y_pos"] - 34  # 34: adjusting value
        observation += f"Position of Mario: ({x_pos}, {y_pos})\n"

        # Object loc
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
        for (
            object_key,
            loc_list,
        ) in found_objects.items():  # object -> object_key (shadowing 방지)
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
        return int(self.reward["distance"]), self.reward["done"]
