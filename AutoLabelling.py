from time import time
import os
import sys
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Any
import torch

try:
    from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate  # type: ignore
    import GroundingDINO.groundingdino.datasets.transforms as T  # type: ignore
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

from database.read_database import ReadImages


class AutoLabellingObjectDetect:
    def __init__(self, input_dir: str = None):
        self.data = ReadImages()

        self.cont: int = 0
        self.num_images: int = 0
        self.class_id: int = 0

        self.box_threshold: float = 0.25
        self.text_threshold: float = 0.25

        self.out_image_path: str = 'datasets/images/val'
        self.out_txt_path: str = 'datasets/labels/val'
        self.prompt: str = 'eye'
        self.home: str = os.getcwd()

        self.input_dir = input_dir or os.path.join(self.home, 'database', 'untagged_images')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.save: bool = True
        self.draw: bool = False

        self.images: list = []
        self.names: list = []
        self.bbox_info: list = []

    def determine_class_id(self, image_name: str) -> Tuple[int, str]:
        """
        Determines the YOLO class_id and label name based on filename prefix:
          - 'awake_'  -> Class 0 (open eye)
          - 'drowsy_' -> Class 1 (close eye)
          - 'yawn_'   -> Class 0 if self.prompt == 'mouth' else Class 0 (open eye)
        """
        lower = image_name.lower()
        if "drowsy" in lower or "close" in lower:
            return 1, "close eye (class 1)"
        elif "awake" in lower or "open" in lower:
            return 0, "open eye (class 0)"
        elif "yawn" in lower:
            return 0, "open eye / yawn (class 0)"
        else:
            return self.class_id, f"default (class {self.class_id})"

    def save_data(self, image_copy: np.ndarray, list_info: list, file_name: str = None):
        out_name = file_name if file_name else str(time()).replace('.', '')
        os.makedirs(self.out_image_path, exist_ok=True)
        os.makedirs(self.out_txt_path, exist_ok=True)

        cv2.imwrite(os.path.join(self.out_image_path, f"{out_name}.jpg"), image_copy)
        with open(os.path.join(self.out_txt_path, f"{out_name}.txt"), 'w') as f:
            for info in list_info:
                f.write(info + "\n")

    def config_grounding_model(self) -> Any:
        if not GROUNDING_DINO_AVAILABLE:
            print("\n" + "=" * 60)
            print("ERROR: GroundingDINO is not installed in your environment.")
            print("To use AutoLabelling, follow these steps:")
            print("  1. Clone the GroundingDINO repository into the project root:")
            print("     git clone https://github.com/IDEA-Research/GroundingDINO.git")
            print("  2. Download the model weights to GroundingDINO/weights/groundingdino_swint_ogc.pth")
            print("  3. Install GroundingDINO dependencies.")
            print("=" * 60 + "\n")
            sys.exit(1)

        config_path = os.path.join(self.home, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
        check_point_path = os.path.join(self.home, "GroundingDINO", "weights", "groundingdino_swint_ogc.pth")

        if not os.path.exists(config_path) or not os.path.exists(check_point_path):
            print("\n" + "=" * 60)
            print("ERROR: GroundingDINO config or weights were not found.")
            print(f"  Expected config: {config_path}")
            print(f"  Expected weights: {check_point_path}")
            print("=" * 60 + "\n")
            sys.exit(1)

        model = load_model(config_path, check_point_path, device=self.device)
        return model

    def main(self):
        print(f"Reading images from: {self.input_dir}")
        self.images, self.names = self.data.read_images(self.input_dir)
        self.num_images = len(self.images)

        if self.num_images == 0:
            print(f"No images found in '{self.input_dir}'.")
            print("Use CaptureData.py or place images in the input directory first.")
            return

        grounding_model = self.config_grounding_model()

        # Crear los directorios si no existen
        os.makedirs(self.out_image_path, exist_ok=True)
        os.makedirs(self.out_txt_path, exist_ok=True)

        while self.cont < self.num_images:
            self.bbox_info = []
            img_name = self.names[self.cont]
            base_name, _ = os.path.splitext(img_name)
            current_class_id, class_label = self.determine_class_id(img_name)

            print('------------------------------------')
            print(f'name_image: {img_name} -> Assigned: {class_label}')

            process_image = self.images[self.cont]
            copy_image = process_image.copy()
            draw_image = process_image.copy()

            transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )

            img_source = Image.fromarray(process_image).convert("RGB")
            img_transform, _ = transform(img_source, None)

            boxes, logits, phrases = predict(
                model = grounding_model,
                image = img_transform,
                caption = self.prompt,
                box_threshold = self.box_threshold,
                text_threshold = self.text_threshold,
                device = self.device
            )

            if len(boxes) != 0:
                h, w, _ = process_image.shape
                for box in boxes:
                    # Skip compound bounding box covering both eyes when individual eye boxes exist
                    if len(boxes) >= 2 and float(box[2]) > 0.18 and float(box[2]) > 1.5 * float(boxes[0][2]):
                        continue

                    xc = max(0.0, min(1.0, float(box[0])))
                    yc = max(0.0, min(1.0, float(box[1])))
                    an = max(0.0, min(1.0, float(box[2])))
                    al = max(0.0, min(1.0, float(box[3])))

                    self.bbox_info.append(f"{current_class_id} {xc} {yc} {an} {al}")
                    x1, y1, x2, y2 = int(xc * w), int(yc * h), int(an * w), int(al * h)
                    print(f"  [box] xc: {x1} yc: {y1} w: {x2} h: {y2} (class {current_class_id})")

                if self.save and self.bbox_info:
                    self.save_data(copy_image, self.bbox_info, file_name=base_name)

                if self.draw:
                    annotated_img = annotate(image_source=draw_image, boxes=boxes, logits=logits, phrases=phrases)
                    out_frame = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    cv2.imshow('Grounding DINO detect', out_frame)
                    cv2.waitKey(0)

            self.cont += 1


if __name__ == '__main__':
    auto_labeling = AutoLabellingObjectDetect()
    auto_labeling.main()