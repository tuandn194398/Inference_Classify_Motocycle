import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2


class Transform:
    def __init__(self, session: str):
        if session == 'train':
            self.aug = A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
                    A.VerticalFlip(p=0.3),
                    A.HorizontalFlip(p=0.3),
                    A.RGBShift(p=0.3),
                    A.RandomBrightnessContrast(p=0.3),
                    A.GaussNoise(p=0.3),
                    A.Rotate(limit=30, p=0.3),
                    A.OneOf([
                        A.Blur(blur_limit=3, p=0.3),
                        A.GaussianBlur(blur_limit=3, p=0.3),
                    ]),
                    A.Normalize(),
                    ToTensorV2()
                ]
            )
        else:
            self.aug = A.Compose(
                [
                    A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
                    A.Normalize(),
                    ToTensorV2()
                ]
            )

    def __call__(self, image):
        return self.aug(image=image)['image']
