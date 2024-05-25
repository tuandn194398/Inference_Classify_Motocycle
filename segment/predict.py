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
        # models_logger.info(f"Weight has been loaded from {weight_path}")
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
        weights=ROOT / 'yolo-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        checkpoint='',
        model_mp='resnet18'
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # model classify color
    # color_classes = ['black', 'red', 'blue', 'white']
    # classify_model = load_model('color-segment 1.h5')
    # use motorbike model (1)
    model_cls = Models(model=model_mp)
    model_cls.load_weight(checkpoint)

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, font_size=8, example=str(names))
            imgwhite = im0.copy()  # tao mot anh background trang
            imgwhite[:] = 0
            # cv2.imwrite('anhtrang.png', imgwhite)
            annotator_white = Annotator(imgwhite, line_width=line_thickness, example=str(names))

            if len(det):
                # masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                # masks = process_mask(proto[2].squeeze(0), det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                masks = process_mask(proto[-1][i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if save_txt:
                    # print("mặt nạ:", type(masks), masks)
                    segments = reversed(masks2segments(masks))
                    segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]
                    # print("mặt nạ segments:", type(segments), segments)

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
                # cv2.imwrite('anhlimgseg.png', imgseg)
                # cv2.imwrite('anhlimc.png', imc)
                # print('ano:', imgseg)

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if save_txt:  # Write to file
                        segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                        line = (cls, *segj, conf) if save_conf else (cls, *segj)  # label format
                        # print('tọa độ segments', segj)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label,
                        #                     color=colors(c, False))  # False thi mau sac ko trung voi mau segment
                        # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                    if save_crop:
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / f'{p.stem}.jpg', BGR=True)
                        # imgbb là ảnh bb
                        # cv2.imwrite('anhdepbb.png', imgbb)
                        imgbb_seg = save_one_box(xyxy, imgsegs,
                                                 file=save_dir / 'crops' / names[c] / f'seg_{p.stem}.jpg', BGR=True,
                                                 save=False)
                        imgbb_seg[imgbb_seg[:, :, 0] == 0] = (255, 255, 255)
                        # path_bbseg = save_dir / 'crops' / names[c] / f'segw_{p.stem}.jpg'
                        path_bbseg = save_dir / 'test' / names[c] / f'{p.stem}.jpg'
                        path_bbseg.parent.mkdir(parents=True, exist_ok=True)  # make directory
                        fpath = str(increment_path(path_bbseg).with_suffix('.jpg'))
                        cv2.imwrite(fpath, imgbb_seg)

                    if save_img:
                        # classify color
                        # use exception model
                        # imgbb_clf = cv2.resize(imgbb_seg, (299, 299))
                        # imgbb_clf = np.expand_dims(imgbb_clf, axis=0)
                        # pred = classify_model.predict(imgbb_clf)
                        # idx = np.argmax(pred[0])
                        # color_label = color_classes[idx]

                        # use motorbike model (2)
<<<<<<< HEAD
                        model_cls = Models(model=model_mp)
                        model_cls.load_weight(checkpoint)
                        result = model_cls.infer(imgbb_seg)
                        print(f"Image: {p.stem}.jpg, Prediction: {result}")
                        if result == 0:
                            color_label = 'black'
                        elif result == 1:
                            color_label = 'blue'
                        elif result == 2:
                            color_label = 'red'
                        else:
                            color_label = 'white'
                        # draw label in the image
                        # if j == 8 or j == 5 or j == 9:
                        #     color_label = 'black'
                        # elif j == 6:
                        #     color_label = 'white'
                        # elif j == 3:
                        #     color_label = 'blue'
                        # elif j == 7:
                        #     continue
=======
                        # model_cls = Models(model=model_mp)
                        # model_cls.load_weight(checkpoint)
                        # result = model_cls.infer(imgbb_seg)
                        # print(f"Image: {p.stem}.jpg, Prediction: {result}")
                        # draw label in the image
>>>>>>> f10140a96f1718466a0517fd137b86f0a5213f31
                        print(p.stem, color_label)
                        label = None if hide_labels else (
                            # f'{names[c][:5]}:{color_label}' if hide_conf else f'{names[c][:5]}{conf:.2f}{color_label}')
                            f'{names[c][:5]}:{color_label}' if hide_conf else f'{j}{names[c][:5]}:{color_label}')
                        annotator.box_label(xyxy, label,
                                            color=colors(c, False))  # False thi mau sac ko trung voi mau segment

            # Stream results
            im0 = annotator.result()

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--checkpoint', '-cp', type=str, default='',
                        help='The path to the checkpoint file to run model predict')
    parser.add_argument('--model_mp', type=str, default='resnet18', help='model name to run predict')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
