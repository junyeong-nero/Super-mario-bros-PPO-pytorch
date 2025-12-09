import base64
import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# --- Configuration Constants ---

DEFAULT_PATTERN_PATH = Path("assets") / "game" / "all_object_patterns.json"
SCREENSHOT_DIR = Path("screenshot")

# Thresholds for template matching confidence
OBJECT_THRESHOLDS: Dict[str, float] = {
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

# Coordinate transformations (x, y) -> (adjusted_x, adjusted_y)
OBJECT_TRANSFORMS: Dict[str, Callable[[int, int], Tuple[int, ...]]] = {
    "brick": lambda x, y: (x, y + 1),
    "question_block": lambda x, y: (x - 1, y + 1),
    "inactivated_block": lambda x, y: (x - 1, y + 1),
    "monster_mushroom": lambda x, y: (x - 4, y + 2),
    "monster_turtle": lambda x, y: (x - 4, y + 10),
    "pit_1start": lambda x, y: (x + 8, y),
    "pit_2end": lambda x, y: (x + 4, y),
    "pipe": lambda x, y: (x - 4, y, y - 32),
    "item_mushroom": lambda x, y: (x - 1, y + 1),
    "stair": lambda x, y: (x, y),
    "flag": lambda x, y: (x - 2, y),
}

# Text labels for the `to_text` method
OBJECT_LABELS: Dict[str, str] = {
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

# Key mapping for `to_dict` output
KEY_MAPPING: Dict[str, str] = {
    "brick": "bricks",
    "question_block": "question_blocks",
    "inactivated_block": "inactivated_blocks",
    "monster_mushroom": "monster_goomba",
    "monster_turtle": "monster_koopas",
    "pit_1start": "pit_start",
    "pit_2end": "pit_end",
    "pipe": "warp_pipe",
    "item_mushroom": "item_mushrooms",
    "stair": "stair_blocks",
    "flag": "flag",
}


@dataclass
class SuperMarioObs:
    """
    Handles observation processing for the Super Mario Bros environment.
    Converts raw image states into structured object detection data.
    """

    state: Dict[str, Any]
    info: Dict[str, Any]
    reward: Dict[str, Any]
    # Input image can be LazyFrames, numpy array, or Tensor
    image: Any = None

    time: str = field(init=False)
    object_pattern_file: Path = field(default=DEFAULT_PATTERN_PATH, init=False)
    thresholds: Dict[str, float] = field(
        default_factory=lambda: OBJECT_THRESHOLDS.copy(), init=False
    )
    object_patterns: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._load_patterns()

    def _load_patterns(self):
        """Loads object detection patterns from JSON file."""
        if self.object_pattern_file.exists():
            try:
                with open(self.object_pattern_file, "r") as json_file:
                    self.object_patterns = json.load(json_file)
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode pattern file: {e}")
        else:
            logging.warning(f"Pattern file not found: {self.object_pattern_file}")

    def _process_obs_image(self, state_input: Any) -> np.ndarray:
        """
        Converts input state (LazyFrames, Tensor, etc.) to (H, W, C) uint8 numpy array.
        """
        # 1. Convert to Numpy
        img = np.array(state_input)

        # 2. Handle Torch Tensor
        if isinstance(state_input, torch.Tensor):
            img = state_input.detach().cpu().numpy()

        # 3. Handle Batch Dimension (1, H, W, C) -> (H, W, C)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        # 4. Normalize/Denormalize (0~1 float -> 0~255 uint8)
        if img.dtype != np.uint8:
            if img.max() <= 1.5:  # Assume normalized 0.0-1.0
                img = img * 255.0
            img = img.astype(np.uint8)

        return img

    def save_state_image(self, state: Any) -> None:
        """Saves the current state image to the screenshot directory."""
        array_255 = self._process_obs_image(state)
        image = Image.fromarray(array_255)

        # Update time to ensure unique filename if called multiple times
        self.time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = SCREENSHOT_DIR / f"{self.time}.png"
        image.save(save_path)

    def find_objects_in_state(
        self, state: Any, object_patterns: Dict[str, Any]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """
        Performs template matching to locate objects in the game state.
        Returns a dictionary of {object_key: [(x, y), ...]} using bottom-left origin.
        """
        found_objects: Dict[str, List[Tuple[int, int]]] = {}

        # 1. Preprocess Main Image
        big_image = self._process_obs_image(state)
        img_height = big_image.shape[0]

        # Convert to Grayscale for template matching
        if big_image.ndim == 3:
            big_image_gray = cv2.cvtColor(big_image, cv2.COLOR_RGB2GRAY)
        else:
            big_image_gray = big_image

        # 2. Iterate through patterns
        for object_name, pattern_data in object_patterns.items():
            # Determine group key (e.g., merge all question_block_* into 1_question_block)
            if "question_block_" in object_name:
                object_key = "1_question_block"
            elif "item_mushroom" in object_name:
                object_key = "8_item_mushroom"
            else:
                object_key = object_name

            # Skip if we already found enough question blocks (optimization from original code)
            if object_key == "1_question_block" and found_objects.get(
                "1_question_block"
            ):
                continue

            if object_key not in found_objects:
                found_objects[object_key] = []

            # --- Prepare Template ---
            template = np.array(pattern_data)
            if isinstance(pattern_data, torch.Tensor):
                template = template.numpy()

            if template.max() <= 1.5:
                template = template * 255.0
            template = template.astype(np.uint8)

            # Fix dimensions (CHW -> HWC if needed) and convert to gray
            if template.ndim == 3:
                if template.shape[0] in [1, 3]:  # Likely CHW
                    template = np.transpose(template, (1, 2, 0))
                template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

            # --- Template Matching ---
            result = cv2.matchTemplate(big_image_gray, template, cv2.TM_CCOEFF_NORMED)

            threshold = self.thresholds.get(object_name, 0.8)
            loc_y, loc_x = np.where(result >= threshold)

            # Convert to (x, y) with Bottom-to-Top coordinate system
            for y, x in zip(loc_y, loc_x):
                # Invert Y axis: image_height - top_y
                inverted_y = img_height - y
                found_objects[object_key].append((int(x), int(inverted_y)))

        return found_objects

    def _get_mario_position(self) -> Tuple[int, int]:
        """Calculates Mario's relative position based on info."""
        x_pos = min(128, self.info.get("x_pos", 0)) - 6
        y_pos = self.info.get("y_pos", 0) - 34
        return x_pos, y_pos

    def to_text(self) -> str:
        """Returns a human-readable string description of the observation."""
        if not self.object_patterns:
            return "Object patterns not loaded or empty."

        found_objects = self.find_objects_in_state(
            self.state["image"], self.object_patterns
        )

        mario_pos = self._get_mario_position()
        lines = [f"Position of Mario: {mario_pos}", "Positions of all objects"]

        for object_key, loc_list in found_objects.items():
            for key_fragment, transform_func in OBJECT_TRANSFORMS.items():
                if key_fragment in object_key:
                    if not loc_list:
                        loc_str = "None"
                    else:
                        # Apply transformation and format
                        coords = [transform_func(x, y) for x, y in loc_list]
                        # Convert tuples to string format "(x,y)"
                        coords_str = [f"({','.join(map(str, c))})" for c in coords]
                        loc_str = ", ".join(coords_str)

                    label_template = OBJECT_LABELS.get(key_fragment, "{}: {}")
                    formatted_line = label_template.format(loc_str)

                    # Special handling for pit formatting (concatenating lines)
                    if key_fragment == "pit_2end" and lines:
                        # Append to the previous line instead of new line
                        lines[-1] += formatted_line
                    else:
                        lines.append(formatted_line)

                    break  # Stop checking transforms for this object

        lines.append(
            "(Note: All (x, y) positions refer to the top-left corner of each object.)"
        )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the detected objects and Mario.
        Applies specific coordinate offsets for each object type.
        """
        if not self.object_patterns:
            return {"error": "Object patterns not loaded or empty."}

        found_objects = self.find_objects_in_state(
            self.state["image"], self.object_patterns
        )

        result = {"mario": self._get_mario_position(), "objects": {}}

        for object_key, loc_list in found_objects.items():
            for key_fragment, transform_func in OBJECT_TRANSFORMS.items():
                if key_fragment in object_key:
                    out_key = KEY_MAPPING.get(key_fragment, key_fragment)

                    if not loc_list:
                        result["objects"][out_key] = []
                    else:
                        result["objects"][out_key] = [
                            transform_func(x, y) for x, y in loc_list
                        ]
                    break

        return result

    def evaluate(self) -> Tuple[int, bool]:
        """Returns the distance reward and done status."""
        return int(self.reward.get("distance", 0)), bool(self.reward.get("done", False))
