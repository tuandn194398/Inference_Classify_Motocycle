import csv
import os
import shutil
import sys

import numpy as np

sys.path.append(os.getcwd())  # NOQA

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

# from src.models import models_logger
# from src.models.classify.resnet18 import ResNet18
# from src.models.classify.vit import ViTBase
import motorbike_project as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18',
                    help='model name')
parser.add_argument('--folder_path', '-fp', type=str, default='',
                    help='folder path')
parser.add_argument('--csv_path', '-csv', type=str, default='',
                    help='The path to the csv file containing the labels')
parser.add_argument('--checkpoint', '-cp', type=str, default='',
                    help='The path to the checkpoint file')

args = parser.parse_args()


class Transform:
    def __init__(self):
        self.transform = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

    def __call__(self, image):
        return self.transform(image=image)["image"]


class Models(torch.nn.Module):
    def __init__(self, model: str = "resnet18", num_classes: int = 4):
        super(Models, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model == "resnet18":
            self.model = mp.ResNet18(num_classes=num_classes).to(self.device)
        elif model == "vit":
            self.model = mp.VisionTransformerBase(num_classes=num_classes).to(self.device)
        if model == 'resnet50':
            self.model = mp.ResNet50(num_classes=num_classes).to(self.device)
        elif model == 'vit_tiny':
            self.model = mp.VisionTransformerTiny(num_classes=num_classes).to(self.device)
        elif model == 'swinv2_base':
            self.model = mp.SwinV2Base(num_classes=num_classes).to(self.device)
        elif model == 'mobilenetv3_large':
            self.model = mp.MobileNetV3Large(num_classes=num_classes).to(self.device)

        self.eval()

    def forward(self, x):
        return self.model(x)

    def load_weight(self, weight_path: str):
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        # models_logger.info(f"Weight has been loaded from {weight_path}")
        print(f"Weight has been loaded from {weight_path}")

    def infer(self, image: Image) -> int:
        img_np = np.array(image.convert("RGB"))
        img = Transform()(img_np).to(self.device)

        with torch.no_grad():
            pred = self(img.unsqueeze(0))

        return torch.argmax(pred, dim=1).item()

    @property
    def name(self):
        return self.model.__class__.__name__


if __name__ == "__main__":
    model = Models(model=args.model)
    model.load_weight(args.checkpoint)

    output = []
    arr_csv = []

    with open(args.csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        wrong = 0
        total = 0
        for i, row in enumerate(reader):
            #     arr_csv.append([row[0], row[1]])
            # arr_csv.pop(0)
            if row[0] in os.listdir(args.folder_path):
                total += 1
                # print("Reading", row[0], "from", args.folder_path)
                # for img_name in os.listdir(args.folder_path):
                img_path = os.path.join(args.folder_path, row[0])
                img = Image.open(img_path)
                result = model.infer(img)
                print(f"Image: {i} {row[0]}, Prediction: {result}")
                output.append({"image": row[0], "prediction": result})
                if str(result) != row[1]:
                    wrong += 1
                    arr_csv.append([row[0], row[1], str(result)])

        print(f"accuracy: {wrong} / {total} = {wrong / total}")

    # print("Dự đoán sai:", arr_csv)
    for file in arr_csv:
        try:
            shutil.move(os.path.join(args.folder_path, file[0]), 'result_13kbb4cls/13kbb_segment1234/new/')
        except:
            continue
        # shutil.copy(os.path.join(args.folder_path, file[0]), 'result_10kbb_seg_3class/resnet50/wrong_valid_v8/')
        # print('File:', file[0])
