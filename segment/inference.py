import argparse
import os
import platform
import sys
from pathlib import Path
# from keras.models import load_model
import numpy as np

from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A

import torch

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode
import motorbike_project as mp


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
        print(f"Weight has been loaded from {weight_path}")

    def infer(self, image: Image) -> int:
        # img_np = np.array(image.convert("RGB"))
        img_np = np.array(image)
        img = Transform()(img_np).to(self.device)

        with torch.no_grad():
            pred = self(img.unsqueeze(0))

        return torch.argmax(pred, dim=1).item()

    @property
    def name(self):
        return self.model.__class__.__name__


@smart_inference_mode()
def run(
        model=None,
        image=None,
        imgsz=(160, 160),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        retina_masks=False,
):
    save_img = not nosave and not (image is None)  # save inference images
    # xmax = ((max(image.shape[0], image.shape[1]) + 31) // 32) * 32
    # xmax = (((image.shape[0] + image.shape[1]) // 2 + 31) // 32) * 32
    # imgsz = (xmax, xmax)
    imgbb_seg_arr = []

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    im0 = image.copy()
    # image = cv2.resize(image, imgsz)
    im = letterbox(image, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred, proto = model(im, augment=augment)[:2]

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    s = ""
    for i, det in enumerate(pred):  # per image
        seen += 1
        s += '%gx%g ' % im.shape[2:]  # print string
        # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, font_size=8, example=str(names))
        imgwhite = im0.copy()  # tao mot anh background trang
        imgwhite[:] = 0
        annotator_white = Annotator(imgwhite, line_width=line_thickness, example=str(names))

        if len(det):
            # masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            # masks = process_mask(proto[2].squeeze(0), det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            masks = process_mask(proto[-1][i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Segments
            if save_txt:
                segments = reversed(masks2segments(masks))
                segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]

            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Mask plotting
            # masks[4] = 0        # xóa segment object thứ len(masks) - 4
            annotator.masks(masks,
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=None if retina_masks else im[i])
            annotator_white.masks(masks,
                                  colors=[colors(x, True) for x in det[:, 5]],
                                  im_gpu=None)

            # tao anh chi chua phan segment
            imglabel = annotator_white.result()
            imglabel[imglabel[:, :, 0] > 0] = 1
            imglabel[imglabel[:, :, 1] > 0] = 1
            imglabel[imglabel[:, :, 2] > 0] = 1
            imgsegs = imc * imglabel

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                if save_txt:  # Write to file
                    segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                if save_crop:
                    imgbb_seg = save_one_box(xyxy, imgsegs, BGR=True, save=False)
                    imgbb_seg[imgbb_seg[:, :, 0] == 0] = (255, 255, 255)
                    imgbb_seg_arr.append([imgbb_seg, names[c]])

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    return imgbb_seg_arr


class SegmentBB:
    def __init__(self, model, image, classes, conf_thres, retina_masks):
        self.model = model
        self.image = image
        self.conf_thres = conf_thres
        self.retina_masks = retina_masks
        self.classes = classes

    def result(self):
        res = run(model=self.model, image=self.image, classes=self.classes, conf_thres=self.conf_thres, retina_masks=self.retina_masks)
        return res


if __name__ == "__main__":
    wei = 'yolov9e-seg.pt'
    sou = 'data/images/MicrosoftTeams-image.png'
    cp = 'resnet50-v9.ckpt'
    seg123 = SegmentBB(wei, sou, cp)
