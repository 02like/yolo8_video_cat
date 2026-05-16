"""
交通监控视频智能分析系统 — NPU 版本
YOLOv8s 推理 → Intel NPU（OpenVINO）
ByteTrack 跟踪 → Ultralytics（独立调用）
"""
import cv2
import numpy as np
import torch
from openvino import Core
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from collections import defaultdict, Counter as CollectionsCounter
from PIL import Image, ImageDraw, ImageFont
from types import SimpleNamespace
import time

# ==================== 配置 ====================
MODEL_PATH = "yolov8s_openvino_model/"
VIDEO_PATH = "4月22日.mp4"
OUTPUT_VIDEO = "output_yolov8s_npu.mp4"
CONF_THRESH = 0.25
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
DEVICE = "NPU"  # 改为 "CPU" 即可对比
# ==============================================

VEHICLE_CLASSES = {2: "小汽车", 3: "其他", 5: "公交车", 7: "卡车"}


# ---- YOLO 预处理 ----
def letterbox(img, new_shape=(640, 640)):
    """等比缩放 + 填充至 640x640"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (r, dw, dh)


def preprocess(img):
    """BGR → RGB → 归一化 → NCHW"""
    img, params = letterbox(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, 0)        # 加 batch 维度
    return img, params


# ---- 检测结果包装器（供 ByteTrack 使用） ----
class Detections:
    __slots__ = ("xyxy", "conf", "cls", "_xywh")

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.atleast_2d(xyxy)
        self.conf = np.atleast_1d(conf)
        self.cls = np.atleast_1d(cls)
        self._xywh = None

    @property
    def xywh(self):
        if self._xywh is None:
            x1 = self.xyxy[:, 0]; y1 = self.xyxy[:, 1]
            x2 = self.xyxy[:, 2]; y2 = self.xyxy[:, 3]
            self._xywh = np.column_stack([
                (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            ])
        return self._xywh

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            idx = [idx]
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])

    def __len__(self):
        return len(self.conf)


# ---- 颜色分类 & 绘图（与 main.py 完全一致） ----
def classify_color(roi):
    if roi.size == 0:
        return "其他"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    cy, cx = h // 2, w // 2
    crop = hsv[int(cy - h * 0.3):int(cy + h * 0.3),
               int(cx - w * 0.3):int(cx + w * 0.3)]
    if crop.size == 0:
        return "其他"
    total = crop.shape[0] * crop.shape[1]

    white = cv2.inRange(crop, np.array([0, 0, 150]), np.array([180, 45, 255]))
    black = cv2.inRange(crop, np.array([0, 0, 0]), np.array([180, 60, 110]))
    red1 = cv2.inRange(crop, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(crop, np.array([170, 50, 50]), np.array([180, 255, 255]))

    w_ratio = np.count_nonzero(white) / total
    b_ratio = np.count_nonzero(black) / total
    r_ratio = (np.count_nonzero(red1) + np.count_nonzero(red2)) / total

    if r_ratio > 0.2:
        return "红色"
    if b_ratio > 0.25:
        return "黑色"
    if w_ratio > 0.25:
        return "白色"
    return "其他"


def draw_chinese_labels(frame, annotations, font_path, info_texts):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        ft14 = ImageFont.truetype(font_path, 14)
    except Exception:
        ft14 = ImageFont.load_default()
    for text, pos, color in annotations:
        draw.text(pos, text, font=ft14, fill=color)

    for text, pos, color, size in info_texts:
        try:
            ft = ImageFont.truetype(font_path, size)
        except Exception:
            ft = ImageFont.load_default()
        draw.text(pos, text, font=ft, fill=color)

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[:] = result[:]


def main():
    # ---- 加载 OpenVINO 模型 ----
    print(f"加载 OpenVINO 模型... 设备: {DEVICE}")
    core = Core()
    ov_model = core.read_model(MODEL_PATH + "yolov8s.xml")
    compiled_model = core.compile_model(ov_model, DEVICE)
    infer_request = compiled_model.create_infer_request()

    # ---- 初始化 ByteTrack ----
    tracker_args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    tracker = BYTETracker(tracker_args, frame_rate=30)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_PATH}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

    color_votes = defaultdict(list)
    cls_votes = defaultdict(list)

    print(f"检测: YOLOv8s (NPU)  +  跟踪: ByteTrack")
    print(f"视频: {VIDEO_PATH}")
    print(f"分辨率: {frame_w}x{frame_h}, {total_frames} 帧, {fps:.0f} fps")

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- NPU 推理 ----
        blob, (ratio, dw, dh) = preprocess(frame)
        result = infer_request.infer({0: blob})
        output = result[0]  # [1, 84, 8400]

        # ---- YOLO 后处理 ----
        preds = torch.from_numpy(output)
        preds = non_max_suppression(
            preds, CONF_THRESH, 0.7, classes=[2, 5, 7], max_det=300
        )
        dets = preds[0]  # [N, 6] tensor: x1,y1,x2,y2,conf,cls

        # ---- 缩放回原始分辨率 ----
        if len(dets):
            dets[:, :4] = scale_boxes(
                (640, 640), dets[:, :4], frame.shape[:2]
            )

        # ---- ByteTrack 跟踪 ----
        if len(dets):
            xyxy = dets[:, :4].cpu().numpy()
            conf = dets[:, 4].cpu().numpy()
            cls_ = dets[:, 5].cpu().numpy()
            dets_wrapper = Detections(xyxy, conf, cls_)
        else:
            dets_wrapper = Detections(
                np.empty((0, 4)), np.empty(0), np.empty(0)
            )

        tracked = tracker.update(dets_wrapper, img=frame)
        # tracked 格式: [x1, y1, x2, y2, track_id, score, cls, idx]

        # ---- 处理跟踪结果 ----
        annotations = []
        for t in tracked:
            x1, y1, x2, y2 = map(int, t[:4])
            tid = int(t[4])
            conf = float(t[5])
            cls_id = int(t[6])

            # 公交车误判纠正（与 main.py 一致）
            if cls_id == 5:
                w, h = x2 - x1, y2 - y1
                aspect_ratio = w / h if h > 0 else 0
                if conf < 0.5 or aspect_ratio < 1.7:
                    cls_id = 3

            # 颜色识别 + 投票
            if y2 > y1 and x2 > x1:
                roi = frame[y1:y2, x1:x2]
                color = classify_color(roi)
                color_votes[tid].append(color)
                cls_votes[tid].append(cls_id)

            final_color = CollectionsCounter(color_votes[tid]).most_common(1)[0][0]
            final_cls = CollectionsCounter(cls_votes[tid]).most_common(1)[0][0]
            vtype = VEHICLE_CLASSES.get(final_cls, "其他")
            label = f"#{tid} {final_color} {vtype}"

            color_map = {"白色": (255, 255, 255), "黑色": (128, 128, 128),
                         "红色": (255, 0, 0), "其他": (255, 255, 0)}
            box_color_bgr = color_map.get(final_color, (0, 255, 0))
            label_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color_bgr, 2)
            annotations.append((label, (x1, max(y1 - 20, 0)), label_color_rgb))

        # ---- 批量绘制中文 ----
        info = [
            (f"累计车辆: {len(color_votes)}", (10, 8), (0, 255, 255), 18),
            (f"YOLOv8s + ByteTrack (NPU) | {DEVICE}", (10, 34), (0, 255, 255), 13),
        ]
        draw_chinese_labels(frame, annotations, FONT_PATH, info)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            speed = frame_idx / elapsed
            eta = (total_frames - frame_idx) / speed if speed > 0 else 0
            print(f"  进度: {frame_idx}/{total_frames} "
                  f"({100*frame_idx/total_frames:.1f}%), "
                  f"速度: {speed:.1f} fps, "
                  f"累计唯一ID: {len(color_votes)}, "
                  f"预计剩余: {eta:.0f}s")

    cap.release()
    out.release()
    elapsed = time.time() - start_time

    # ---- 汇总统计 ----
    type_stats = defaultdict(int)
    color_stats = defaultdict(int)

    for tid in color_votes:
        final_cls = CollectionsCounter(cls_votes[tid]).most_common(1)[0][0]
        vtype = VEHICLE_CLASSES.get(final_cls, "其他")
        final_color = CollectionsCounter(color_votes[tid]).most_common(1)[0][0]
        type_stats[vtype] += 1
        color_stats[final_color] += 1

    total_vehicles = len(color_votes)

    print(f"\n处理完成! 总耗时: {elapsed:.1f}s, 平均速度: {frame_idx/elapsed:.1f} fps")
    print(f"设备: {DEVICE}")
    print("\n" + "=" * 50)
    print("            统 计 结 果")
    print("=" * 50)
    print(f"  总车流量:        {total_vehicles} 辆")
    print(f"  ── 车辆类型分布 ──")
    for t in ["小汽车", "公交车", "卡车", "其他"]:
        print(f"    {t}:            {type_stats.get(t, 0)} 辆")
    print(f"  ── 车身颜色分布 ──")
    for c in ["白色", "黑色", "红色", "其他"]:
        label = "其他颜色" if c == "其他" else c
        print(f"    {label}:        {color_stats.get(c, 0)} 辆")
    print("=" * 50)
    print(f"\n输出视频: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
