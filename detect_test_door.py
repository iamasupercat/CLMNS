"""
YOLOv5 도어 객체 검출 테스트
도어 이미지에서 객체를 검출하고 크롭 이미지를 생성

python detect_test_door.py \
    --weights ./runs/train/FrontDoor13/weights/best.pt \
    --source /home/work/datasets/TXT/test_frontdoor.txt \
    --save-crop \
    --save-txt \
    --save-csv \
    --save-conf \
    --name FrontDoor13_test \
    --conf-thres 0.5 \
    --iou-thres 0.4 \
    --imgsz 640



python detect_test_door.py \
    --weights ./runs/train/FrontDoor13/weights/best.pt \ -> YOLO 모델 가중치 파일
    --source /path/to/images \ -> 검출할 이미지 경로 (폴더/파일/텍스트파일)
    --save-crop \ -> 검출된 객체 크롭 이미지 저장
    --save-txt \ -> 검출 결과 텍스트 파일 저장
    --name my_door_test \ -> 실험 이름 (출력 폴더명)
    --conf-thres 0.5 \ -> 검출 신뢰도 임계값
    --imgsz 640 \ -> 입력 이미지 크기
    --device cuda \ -> 사용 디바이스 (cpu/cuda)

# 기본 도어 검출
python detect_test_door.py \
    --weights best_door.pt \
    --source door_images/ \
    --save-crop \
    --save-txt

# 고급 도어 검출 (높은 신뢰도)
python detect_test_door.py \
    --weights best_door.pt \
    --source test_door_images/ \
    --name door_production_test \
    --save-crop \
    --save-txt \
    --conf-thres 0.6 \
    --iou-thres 0.4 \
    --device cuda

# 텍스트 파일에서 이미지 경로 읽기 (배치 처리)
python detect_test_door.py \
    --weights best_door.pt \
    --source ../../for_yolo/test_fixed.txt \
    --name door_batch_test \
    --save-crop \
    --save-txt

# 성능 평가용 (CSV 포함)
python detect_test_door.py \
    --weights grid_search_0804_1112_ep50_bs32_im640_optAdam_hydat.pt \
    --source ../../datasets/TXT/test.txt \
    --save-crop \
    --save-txt \
    --save-csv \
    --save-conf

# 실시간 시각화 (GUI 환경)
python detect_test_door.py \
    --weights best_door.pt \
    --source door_images/ \
    --view-img \
    --save-crop

# 생성되는 결과:
# - runs/detect/{name}/ (YOLO 검출 결과)
#   ├── labels/ (YOLO 라벨 파일, YOLO 형식)
#   ├── crops/door/ (크롭된 도어 이미지)
#   └── *.jpg (검출 박스가 그려진 이미지)
# 
# 후속 ResNet 분류를 위해서는:
# 1. --save-crop으로 크롭 이미지 생성
# 2. 생성된 crops/door/ 폴더를 ResNet 입력으로 사용
# 3. cd ../resnet && python resnet_test_door.py --name {YOLO_name}
#
# 주요 파라미터:
# --weights: YOLO 모델 파일 (best_door.pt)
# --source: 이미지 소스 (폴더, 파일, 텍스트파일)
# --conf-thres: 검출 신뢰도 (0.0~1.0, 기본값: 0.25)
# --save-crop: 검출 객체 크롭 이미지 저장 (ResNet 입력용 필수)
# --name: 실험 이름 (ResNet과 연동 시 동일하게 사용)
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
import subprocess

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=Path("yolov5s.pt"),
    source=Path("data/images"),
    data=Path("data/coco128.yaml"),
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_txt=False,
    save_format=0,
    save_csv=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    update=False,
    project=Path("runs/detect"),
    name="exp",
    exist_ok=False,
    line_thickness=3,
    hide_labels=False,
    hide_conf=False,
    half=False,
    dnn=False,
    vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # 사용자 지정 번호 매김 방식 (_1, _2, _3...)
    base_dir = Path(project) / name
    if not exist_ok:
        i = 1
        while True:
            save_dir = Path(str(base_dir) + f"_{i}")
            if not save_dir.exists():
                break
            i += 1
    else:
        save_dir = base_dir
    
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    img_crop_index = 1

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        with dt[1]:
            visualize_dir = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize_dir).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize_dir).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize_dir)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):
            seen += 1
            p, im0 = path, im0s.copy()
            p = Path(p)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                class_map = {'high': None, 'mid': None, 'low': None}
                for k, v in enumerate(names):
                    if v in class_map:
                        class_map[v] = k

                det_by_class = {'high': None, 'mid': None, 'low': None}

                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    cls_name = names[cls]
                    if cls_name in det_by_class and det_by_class[cls_name] is None:
                        det_by_class[cls_name] = (xyxy, cls)

                if all(v is not None for v in det_by_class.values()):
                    crop_dir = save_dir / "crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)

                    # good/bad 판별
                    filename = str(p)  # p는 Path 객체
                    if 'good' in filename:
                        quality = 1
                    elif 'bad' in filename:
                        quality = 0
                    else:
                        quality = 'unknown'  # 예외 처리

                    for part_name in ['high', 'mid', 'low']:
                        xyxy, cls = det_by_class[part_name]
                        crop_file = crop_dir / f"{part_name}_{quality}_{img_crop_index}.jpg"
                        save_one_box(xyxy, imc, file=crop_file, BGR=True)

                    img_crop_index += 1

    t = tuple(x.t / seen * 1e3 for x in dt)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image" % t)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default="yolov5s.pt")
    parser.add_argument("--source", type=str, default="data/images")
    parser.add_argument("--data", type=str, default="data/coco128.yaml")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-format", type=int, default=0)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="exp_test_door")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--line-thickness", default=3, type=int)
    parser.add_argument("--hide-labels", default=False, action="store_true")
    parser.add_argument("--hide-conf", default=False, action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dnn", action="store_true")
    parser.add_argument("--vid-stride", type=int, default=1)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements("requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))

    resnet_path = "resnet_test_door.py"          # 실행할 스크립트 파일명 (또는 run_resnet_inference.py)
    resnet_dir = "../resnet"           # 해당 스크립트가 위치한 디렉토리 (상대경로 or 절대경로)

    subprocess.run(["python3", resnet_path], check=True, cwd=resnet_dir)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
