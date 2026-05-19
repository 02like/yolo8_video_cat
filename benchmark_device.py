"""设备性能基准测试 — 使用 intel:DEVICE 格式切换 OpenVINO 设备"""
import time, cv2
from ultralytics import YOLO

VIDEO = "4月22日.mp4"
WARMUP, TEST = 30, 300

# (device_arg, label)
TESTS = [
    (None,      "CPU (默认AUTO)"),
    ("intel:CPU",   "CPU"),
    ("intel:GPU",   "GPU (Arc iGPU)"),
    ("intel:NPU",   "NPU (AI Boost)"),
    ("intel:HETERO:GPU,CPU", "HETERO:GPU,CPU"),
    ("intel:MULTI:GPU,CPU",  "MULTI:GPU,CPU"),
    ("intel:MULTI:CPU,GPU",  "MULTI:CPU,GPU"),
]

print("=" * 60)
print("  OpenVINO 设备基准测试")
print(f"  ({WARMUP}帧预热 + {TEST}帧计时, 完整检测+跟踪管线)")
print("=" * 60)

scores = {}

for dev_arg, label in TESTS:
    print(f"\n  {label} ... ", end="", flush=True)
    try:
        model = YOLO("yolov8s_openvino_model/", task="detect", verbose=False)
        cap = cv2.VideoCapture(VIDEO)
        fi = 0
        total = WARMUP + TEST

        while fi < total:
            ret, frame = cap.read()
            if not ret:
                break
            kwargs = dict(persist=True, tracker="bytetrack.yaml",
                          conf=0.25, classes=[2,5,7], verbose=False)
            if dev_arg is not None:
                kwargs["device"] = dev_arg
            model.track(frame, **kwargs)
            fi += 1
            if fi == WARMUP:
                t0 = time.time()

        cap.release()
        elapsed = time.time() - t0
        fps = TEST / elapsed if elapsed > 0 else 0
        scores[label] = fps
        print(f"{fps:.1f} fps ({elapsed:.1f}s)")
    except Exception as e:
        print(f"FAILED: {str(e)[:150]}")

print(f"\n{'='*60}")
print("  最终排名")
print("=" * 60)
baseline = scores.get("CPU (默认AUTO)", scores.get("CPU", 1))
for label, fps in sorted(scores.items(), key=lambda x: -x[1]):
    bar = "#" * max(1, int(fps / 2))
    speedup = f"(x{fps/baseline:.1f})" if baseline > 0 else ""
    print(f"  {label:25s}  {fps:5.1f} fps  {speedup:8s}  {bar}")
print("=" * 60)
