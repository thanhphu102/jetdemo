#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv8 TensorRT detector (Python 3, standalone - no ROS).

Receives JPEG frames over UDP from camera_udp_forwarder.py (Py2), runs the engine,
and sends detection JSON over UDP to yolo_bridge_node.py (Py2), which republishes it
on /yolo_detections. The TensorRT engine wrapper, letterbox/preprocess and YOLOv8
postprocess are kept local here so the runtime path does not depend on notebooks
or standalone prototypes.

Run:
    python3 trt_udp_detector.py [--config config/yolo.yaml] [--mock]

--mock skips TensorRT entirely and emits a rotating fake detection so the
detector -> bridge -> /yolo_detections path can be verified without the engine.
"""
import argparse
import json
import os
import socket
import time

import cv2
import numpy as np
import yaml

# TensorRT / PyCUDA are imported lazily (only when not --mock) so this file runs
# on a dev box for plumbing tests.
trt = None
cuda = None


def _import_trt():
    global trt, cuda
    import tensorrt as _trt
    import pycuda.driver as _cuda
    import pycuda.autoinit  # noqa: F401  (initialises the CUDA context)
    trt, cuda = _trt, _cuda


# ============================================================
# PREPROCESS
# ============================================================
def letterbox(image, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / w, new_h / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, scale, pad_left, pad_top


def preprocess(frame, input_size=320, dtype=np.float32):
    img, scale, pad_left, pad_top = letterbox(frame, (input_size, input_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(dtype) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    return img, scale, pad_left, pad_top


# ============================================================
# TENSORRT ENGINE WRAPPER
# ============================================================
class TensorRTEngine:
    def __init__(self, engine_path, input_size):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Could not load TensorRT engine: %s" % engine_path)

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.input_binding_idx = None
        self.output_binding_idx = None

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.engine.get_binding_shape(i))
            if self.engine.binding_is_input(i):
                self.input_binding_idx = i
                self.input_name = name
                self.input_dtype = dtype
                self.input_shape = shape
            else:
                self.output_binding_idx = i
                self.output_name = name
                self.output_dtype = dtype
                self.output_shape = shape

        if self.input_binding_idx is None or self.output_binding_idx is None:
            raise RuntimeError("Engine must have exactly 1 input and 1 output.")

        if any(dim < 0 for dim in self.input_shape):
            self.context.set_binding_shape(self.input_binding_idx, (1, 3, input_size, input_size))
            self.input_shape = tuple(self.context.get_binding_shape(self.input_binding_idx))

        self.output_shape = tuple(self.context.get_binding_shape(self.output_binding_idx))
        self._allocate_buffers()

        print("[INFO] Engine loaded:", engine_path)
        print("[INFO] Input :", self.input_name, self.input_shape, self.input_dtype)
        print("[INFO] Output:", self.output_name, self.output_shape, self.output_dtype)

    def _volume(self, shape):
        vol = 1
        for s in shape:
            vol *= int(s)
        return vol

    def _allocate_buffers(self):
        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))
            size = self._volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(i):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def infer(self, input_tensor):
        input_tensor = input_tensor.astype(self.input_dtype, copy=False)
        input_tensor = np.ascontiguousarray(input_tensor)
        np.copyto(self.host_inputs[0], input_tensor.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        output = np.array(self.host_outputs[0], dtype=self.output_dtype)
        output = output.reshape(self.output_shape)
        return output


# ============================================================
# POSTPROCESS YOLOV8 RAW + NMS
# ============================================================
def _xywh_to_xyxy(cx, cy, w, h):
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def postprocess_yolov8_raw(output, class_names, orig_w, orig_h, scale, pad_left, pad_top,
                           conf_thres=0.35, iou_thres=0.45):
    pred = np.squeeze(output)
    if pred.ndim != 2:
        print("[WARN] output after squeeze is not 2D:", pred.shape)
        return []
    if pred.shape[0] < pred.shape[1]:   # YOLOv8 raw is (C, N) -> (N, C)
        pred = pred.T

    num_classes = len(class_names)
    expected_dim = 4 + num_classes
    if pred.shape[1] < expected_dim:
        print("[WARN] output shape does not match raw YOLOv8:", pred.shape,
              "expected last dim >=", expected_dim)
        return []

    boxes, scores, class_ids = [], [], []
    for row in pred:
        class_scores = row[4:4 + num_classes]
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])
        if conf < conf_thres:
            continue
        cx, cy, bw, bh = row[:4]
        x1, y1, x2, y2 = _xywh_to_xyxy(cx, cy, bw, bh)
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale
        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))
        w = x2 - x1
        h = y2 - y1
        if w <= 2 or h <= 2:
            continue
        boxes.append([int(x1), int(y1), int(w), int(h)])
        scores.append(conf)
        class_ids.append(class_id)

    if not boxes:
        return []
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_thres, nms_threshold=iou_thres)
    detections = []
    if len(indices) > 0:
        for i in np.array(indices).reshape(-1):
            x, y, w, h = boxes[i]
            detections.append({"class_id": class_ids[i],
                               "class_name": class_names[class_ids[i]],
                               "conf": round(scores[i], 4),
                               "box": [x, y, x + w, y + h]})
    return detections


# ============================================================
# CONFIG
# ============================================================
def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.normpath(os.path.join(here, "..", "config", "yolo.yaml"))
    p = argparse.ArgumentParser(description="YOLOv8 TensorRT UDP detector")
    p.add_argument("--config", default=default_cfg, help="path to yolo.yaml")
    p.add_argument("--mock", action="store_true", help="emit fake detections, no TensorRT")
    p.add_argument("--engine", help="override net.engine_path")
    p.add_argument("--frame-port", type=int, help="override transport.frame_port")
    p.add_argument("--det-port", type=int, help="override transport.det_port")
    p.add_argument("--conf", type=float, help="override net.conf_thres")
    return p.parse_args()


# ============================================================
# UDP HELPERS
# ============================================================
def latest_datagram(sock, bufsize=65535, timeout=1.0):
    """Block (up to timeout) for one datagram, then drain any backlog -> keep the newest."""
    sock.settimeout(timeout)
    try:
        data, _ = sock.recvfrom(bufsize)
    except socket.timeout:
        return None
    sock.setblocking(False)
    while True:
        try:
            data, _ = sock.recvfrom(bufsize)
        except (BlockingIOError, socket.error):
            break
    return data


# ============================================================
# MAIN
# ============================================================
def run_mock(class_names, send_sock, dst, rate_hz):
    print("[INFO] MOCK mode: emitting rotating fake detections to %s:%d" % dst)
    names = class_names or ["mock_class"]
    period = 1.0 / max(rate_hz, 1.0)
    i = 0
    while True:
        cid = i % len(names)
        msg = {"stamp": round(time.time(), 3),
               "detections": [{"class_id": cid, "class_name": names[cid],
                               "conf": 0.99, "box": [100, 80, 180, 160]}]}
        send_sock.sendto(json.dumps(msg).encode("utf-8"), dst)
        i += 1
        time.sleep(period)


def run_detector(cfg, frame_port, det_port, dst, class_names):
    net = cfg.get("net", {})
    input_size = int(net.get("input_size", 320))
    conf_thres = float(net.get("conf_thres", 0.35))
    iou_thres = float(net.get("iou_thres", 0.45))
    engine_path = net["engine_path"]
    debug_every = int(cfg.get("detector", {}).get("debug_every", 15))

    _import_trt()
    engine = TensorRTEngine(engine_path, input_size)

    # Validate class_names against the engine output dim: YOLOv8 raw is 4 + num_classes.
    out_dim = max(engine.output_shape[-1], engine.output_shape[-2])
    expected = 4 + len(class_names)
    if out_dim != expected:
        raise SystemExit(
            "[FATAL] class_names count mismatch: engine output dim=%d but 4 + len(class_names)=%d "
            "(class_names has %d). Fix net.class_names in yolo.yaml to the engine's trained order."
            % (out_dim, expected, len(class_names)))
    print("[INFO] class check OK: %d classes -> output dim %d" % (len(class_names), out_dim))

    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("", frame_port))
    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("[INFO] detector listening for frames on :%d, sending detections to %s:%d"
          % (frame_port, dst[0], dst[1]))

    n = 0
    prev = time.time()
    while True:
        data = latest_datagram(recv)
        if data is None:
            print("[WARN] no frames received in the last 1s")
            continue
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        orig_h, orig_w = frame.shape[:2]
        tensor, scale, pad_left, pad_top = preprocess(frame, input_size, engine.input_dtype)
        t0 = time.time()
        output = engine.infer(tensor)
        infer_ms = (time.time() - t0) * 1000.0
        dets = postprocess_yolov8_raw(output, class_names, orig_w, orig_h, scale, pad_left, pad_top,
                                      conf_thres, iou_thres)
        msg = {"stamp": round(time.time(), 3), "detections": dets}
        send.sendto(json.dumps(msg).encode("utf-8"), dst)

        n += 1
        if debug_every and n % debug_every == 0:
            now = time.time()
            fps = debug_every / max(now - prev, 1e-6)
            prev = now
            if dets:
                best = max(dets, key=lambda d: d["conf"])
                print("[DET] %s conf=%.2f box=%s fps=%.1f infer=%.1fms"
                      % (best["class_name"], best["conf"], best["box"], fps, infer_ms))
            else:
                print("[DET] none fps=%.1f infer=%.1fms" % (fps, infer_ms))


def main():
    args = build_args()
    cfg = load_config(args.config)
    net = cfg.get("net", {})
    tr = cfg.get("transport", {})
    if args.engine:
        net["engine_path"] = args.engine
    if args.conf is not None:
        net["conf_thres"] = args.conf

    udp_ip = tr.get("udp_ip", "127.0.0.1")
    frame_port = args.frame_port or int(tr.get("frame_port", 5006))
    det_port = args.det_port or int(tr.get("det_port", 5005))
    dst = (udp_ip, det_port)
    class_names = net.get("class_names", [])

    if args.mock:
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rate = float(cfg.get("frame", {}).get("rate_hz", 12))
        try:
            run_mock(class_names, send, dst, rate)
        except KeyboardInterrupt:
            print("\n[INFO] mock stopped.")
        return

    try:
        run_detector(cfg, frame_port, det_port, dst, class_names)
    except KeyboardInterrupt:
        print("\n[INFO] detector stopped.")


if __name__ == "__main__":
    main()
