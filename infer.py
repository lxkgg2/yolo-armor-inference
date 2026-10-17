#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装甲板视频推理脚本（YOLO26n 四关键点模型）

功能：
  1. 加载课程指定模型权重 best.pt
  2. 读取自己拍摄的装甲板视频，逐帧执行推理
  3. 在检测到目标时绘制：检测框、类别、置信度，以及该目标的 4 个二维关键点
  4. 保存与原视频内容对应、可正常播放的推理结果视频

关键点直接来自模型的预测结果（result.keypoints），不使用检测框四角代替。

用法示例：
  python infer.py --model weights/best.pt --source videos/armor.mp4 \
                  --output results/armor_result.mp4 --conf 0.25 --device cpu

参数均可通过命令行指定，模型路径、视频路径、输出位置可自由修改。
"""

import argparse
import os
import sys

import cv2
import torch
from ultralytics import YOLO

# 装甲板类别（与模型训练保持一致：B1 B2 B3 R1 R2 R3）
ARMOR_CLASSES = ["B1", "B2", "B3", "R1", "R2", "R3"]

# 每个目标预测的关键点数量（装甲板四角点）
NUM_KEYPOINTS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="装甲板视频推理与关键点可视化")
    parser.add_argument("--model", type=str, default="weights/best.pt",
                        help="YOLO 模型权重路径（默认 weights/best.pt）")
    parser.add_argument("--source", type=str, default="videos/armor.mp4",
                        help="输入视频路径")
    parser.add_argument("--output", type=str, default="results/armor_result.mp4",
                        help="推理结果视频保存路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值（默认 0.25）")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值（默认 0.45）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理输入尺寸（默认 640）")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备：cpu / 0 / 0,1 等；不填则自动选择 GPU 或 CPU")
    return parser.parse_args()


def resolve_device(device):
    """根据用户指定或硬件情况选择推理设备（CPU 或 GPU）。"""
    if device is not None:
        return device
    if torch.cuda.is_available():
        print("[Info] 检测到 CUDA，使用 GPU 推理。")
        return 0
    print("[Info] 未检测到 CUDA，使用 CPU 推理。")
    return "cpu"


def main():
    args = parse_args()

    # 1. 检查输入文件
    if not os.path.isfile(args.model):
        sys.exit(f"[错误] 找不到模型权重: {args.model}")
    if not os.path.isfile(args.source):
        sys.exit(f"[错误] 找不到输入视频: {args.source}")

    # 2. 加载模型
    print(f"[Info] 加载模型: {args.model}")
    model = YOLO(args.model)
    # 若权重中未内嵌类别名，则手动指定装甲板类别，保证可视化标签正确
    if not model.names or len(model.names) < len(ARMOR_CLASSES):
        print("[Info] 模型未内嵌类别名，使用手动指定类别列表。")
        model.names = {i: ARMOR_CLASSES[i] for i in range(len(ARMOR_CLASSES))}

    device = resolve_device(args.device)
    print(f"[Info] 推理设备: {device}")

    # 3. 打开输入视频
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"[错误] 无法打开视频: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Info] 视频信息: {fps:.2f} FPS, {width}x{height}, 共约 {total} 帧")

    # 4. 准备输出视频（保持与输入一致的分辨率与帧率）
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        sys.exit(f"[错误] 无法创建输出视频: {args.output}（可尝试在代码中改用 avc1 编码）")

    # 5. 逐帧推理与可视化
    frame_idx = 0
    det_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                iou=args.iou, device=device, verbose=False)
        r = results[0]

        boxes = r.boxes
        keypoints = r.keypoints

        if boxes is not None and len(boxes) > 0:
            det_frames += 1
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            # 关键点像素坐标，shape: (N, K, 2)
            kps_xy = None
            if keypoints is not None and keypoints.xy is not None:
                kps_xy = keypoints.xy.cpu().numpy()

            for i in range(len(boxes)):
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                cls_name = model.names.get(clss[i], f"class{clss[i]}")
                conf = confs[i]

                # 检测框：B 系列用蓝色，R 系列用红色
                color = (255, 0, 0) if cls_name.startswith("B") else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 类别 + 置信度（绘制在框上方）
                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                top = max(0, y1 - th - 10)
                cv2.rectangle(frame, (x1, top), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # 4 个关键点：绿色圆点 + 序号，来自模型预测
                if kps_xy is not None:
                    for j in range(min(NUM_KEYPOINTS, kps_xy.shape[1])):
                        kx, ky = int(kps_xy[i, j, 0]), int(kps_xy[i, j, 1])
                        cv2.circle(frame, (kx, ky), 6, (0, 255, 0), -1)
                        cv2.putText(frame, str(j + 1), (kx - 8, ky - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        writer.write(frame)

        if frame_idx % 50 == 0:
            print(f"[Info] 已处理 {frame_idx}/{total} 帧，含目标的帧数: {det_frames}")

    cap.release()
    writer.release()

    print(f"\n[完成] 推理结束。")
    print(f"  输入视频: {args.source}")
    print(f"  输出视频: {args.output}")
    print(f"  共处理 {frame_idx} 帧，其中 {det_frames} 帧检测到目标。")

    if det_frames == 0:
        print("[提醒] 全片未检测到任何目标，请检查：权重是否正确、视频中是否确有装甲板、")
        print("        推理参数（conf/imgsz/device）是否合理，并核对模型类别是否匹配。")


if __name__ == "__main__":
    main()
