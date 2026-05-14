"""
交通监控视频智能分析系统
功能: 车流量统计、车辆颜色识别、车辆类型分类
检测 + 跟踪: YOLOv8n + ByteTrack (Ultralytics)
"""
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter as CollectionsCounter
from PIL import Image, ImageDraw, ImageFont
import time

# ==================== 配置 ====================
MODEL_PATH = "yolov8n_openvino_model/"
VIDEO_PATH = "4月22日.mp4"
OUTPUT_VIDEO = "output_annotated.mp4"
CONF_THRESH = 0.25
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
# ==============================================

VEHICLE_CLASSES = {2: "小汽车", 3: "其他", 5: "公交车", 7: "卡车"}


def draw_chinese_labels(frame, annotations, font_path, info_texts):
    """
    在帧上批量绘制中文标注
    annotations: [(text, (x, y), color_rgb), ...]
    info_texts: [(text, (x, y), color_rgb, font_size), ...]
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 绘制车辆标签（字号14）
    try:
        ft14 = ImageFont.truetype(font_path, 14)
    except Exception:
        ft14 = ImageFont.load_default()
    for text, pos, color in annotations:
        draw.text(pos, text, font=ft14, fill=color)

    # 绘制信息文字（不同字号）
    for text, pos, color, size in info_texts:
        try:
            ft = ImageFont.truetype(font_path, size)
        except Exception:
            ft = ImageFont.load_default()
        draw.text(pos, text, font=ft, fill=color)

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    frame[:] = result[:]


def classify_color(roi):
    """HSV 颜色分类"""
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
    black = cv2.inRange(crop, np.array([0, 0, 0]), np.array([180, 255, 80]))
    red1 = cv2.inRange(crop, np.array([0, 50, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(crop, np.array([170, 50, 50]), np.array([180, 255, 255]))

    w_ratio = np.count_nonzero(white) / total
    b_ratio = np.count_nonzero(black) / total
    r_ratio = (np.count_nonzero(red1) + np.count_nonzero(red2)) / total

    if r_ratio > 0.25:
        return "红色"
    if b_ratio > 0.4:
        return "黑色"
    if w_ratio > 0.3:
        return "白色"
    return "其他"


def main():
    model = YOLO(MODEL_PATH, task='detect')

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
    id_to_type = {}

    print(f"检测 + 跟踪: YOLOv8n + ByteTrack (Ultralytics)")
    print(f"视频: {VIDEO_PATH}")
    print(f"分辨率: {frame_w}x{frame_h}, {total_frames} 帧, {fps:.0f} fps")

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, tracker='bytetrack.yaml',
            conf=CONF_THRESH, classes=[2, 5, 7], verbose=False
        )

        # 收集本帧所有标注
        annotations = []
        boxes_data = results[0].boxes
        if boxes_data.id is not None:
            track_ids = boxes_data.id.int().tolist()
            bboxes = boxes_data.xyxy.tolist()
            cls_ids = boxes_data.cls.int().tolist()

            for tid, bbox, cls_id in zip(track_ids, bboxes, cls_ids):
                x1, y1, x2, y2 = map(int, bbox)
                roi = frame[y1:y2, x1:x2]
                color = classify_color(roi)
                color_votes[tid].append(color)

                if tid not in id_to_type:
                    id_to_type[tid] = VEHICLE_CLASSES.get(cls_id, "其他")

                final_color = CollectionsCounter(color_votes[tid]).most_common(1)[0][0]
                label = f"#{tid} {final_color} {id_to_type[tid]}"

                color_map = {"白色": (255, 255, 255), "黑色": (128, 128, 128),
                             "红色": (255, 0, 0), "其他": (255, 255, 0)}
                box_color_bgr = color_map.get(final_color, (0, 255, 0))
                label_color_rgb = (box_color_bgr[2], box_color_bgr[1], box_color_bgr[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color_bgr, 2)
                annotations.append((label, (x1, max(y1 - 20, 0)), label_color_rgb))

        # 批量绘制中文标注（整帧只做一次 PIL 转换）
        info = [
            (f"累计车辆: {len(color_votes)}", (10, 8), (0, 255, 255), 18),
            ("YOLOv8n + ByteTrack", (10, 34), (0, 255, 255), 13),
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

    # --- 汇总统计 ---
    type_stats = defaultdict(int)
    color_stats = defaultdict(int)

    for tid in color_votes:
        vtype = id_to_type.get(tid, "其他")
        final_color = CollectionsCounter(color_votes[tid]).most_common(1)[0][0]
        type_stats[vtype] += 1
        color_stats[final_color] += 1

    total_vehicles = len(color_votes)

    print(f"\n处理完成! 总耗时: {elapsed:.1f}s, 平均速度: {frame_idx/elapsed:.1f} fps")
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
