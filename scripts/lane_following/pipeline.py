"""Pure lane-following pipeline ported from lane_tracking_camera_jetson_smart_city.ipynb.

Lane detection + heading + steering control only. NO ROS, NO intersection/turn/sim.
Functions are copied verbatim from the notebook (cell 1 core pipeline, cell 7 control)
so behaviour/tuning stays identical; see config/lane_following.yaml for parameters.
"""
import cv2
import numpy as np
import math


def build_roi_polygon(w, h, roi):
    cx = (0.5 + roi["center_shift"]) * w
    top_y = roi["top_y"] * h
    bot_y = roi["bottom_y"] * h
    tw = roi["top_width"] * w
    bw = roi["bottom_width"] * w
    return np.array([
        [cx - bw / 2.0, bot_y], [cx - tw / 2.0, top_y],
        [cx + tw / 2.0, top_y], [cx + bw / 2.0, bot_y],
    ], dtype=np.int32)


def roi_mask(shape_hw, polygon):
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def to_edges(frame, canny):
    k = int(canny["blur_kernel"])
    if k % 2 == 0:
        k += 1
    if k < 3:
        k = 3
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.Canny(blur, int(canny["low"]), int(canny["high"]))


def mask_edges(edges, mask):
    return cv2.bitwise_and(edges, mask)


def detect_lines(masked_edges, hough):
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, int(hough["threshold"]),
                            minLineLength=int(hough["min_line_length"]),
                            maxLineGap=int(hough["max_line_gap"]))
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)


def draw_lines(img, lines, color, thickness):
    for x1, y1, x2, y2 in lines:
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return img


def _fit_side(segs, w, h, roi):
    if len(segs) == 0:
        return None, 0.0
    xs, ys, weight = [], [], 0.0
    for x1, y1, x2, y2 in segs:
        xs += [x1, x2]
        ys += [y1, y2]
        weight += float(math.hypot(x2 - x1, y2 - y1))  # total segment length = line "solidity"
    coef = np.polyfit(np.array(ys, dtype=np.float64), np.array(xs, dtype=np.float64), 1)
    y_top = roi["top_y"] * h
    y_bot = roi["bottom_y"] * h
    line = np.array([coef[0] * y_bot + coef[1], y_bot,
                     coef[0] * y_top + coef[1], y_top], dtype=np.float32)
    return line, weight


def filter_lines(lines, w, h, flt, roi):
    cx = (0.5 + roi["center_shift"]) * w
    left, right = [], []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        slope = (y2 - y1) / float(x2 - x1)
        a = abs(slope)
        if a < flt["min_abs_slope"] or a > flt["max_abs_slope"]:
            continue
        midx = (x1 + x2) / 2.0
        if slope < 0 and midx < cx:
            left.append((x1, y1, x2, y2))
        elif slope > 0 and midx > cx:
            right.append((x1, y1, x2, y2))
    lline, lw = _fit_side(left, w, h, roi)
    rline, rw = _fit_side(right, w, h, roi)
    return lline, rline, lw, rw


def validate_lane(left, right, prev_center, w, val, lw=0.0, rw=0.0):
    # One line is enough to keep following (e.g. gaps in a dashed boundary).
    # Stopping for an intersection is handled by the caller (both lines gone).
    if left is None and right is None:
        return "invalid", "no lines", prev_center
    if left is not None and right is not None:
        lx, rx = float(left[0]), float(right[0])
        width = abs(rx - lx)
        raw_center = (lx + rx) / 2.0  # steering bias is applied in estimate_lane, not here
        if width < val["min_lane_width_frac"] * w or width > val["max_lane_width_frac"] * w:
            status, reason = "low_confidence", "lane width out of range"
        else:
            status, reason = "valid", ""
    else:
        raw_center = float((left if left is not None else right)[0])
        status, reason = "low_confidence", "single line"
    if prev_center is None:
        center = raw_center
    else:
        a = val["smooth_alpha"]
        center = a * raw_center + (1.0 - a) * prev_center
        if abs(raw_center - prev_center) > val["max_center_jump_frac"] * w and status == "valid":
            status, reason = "low_confidence", "center jump"
    return status, reason, center


def make_lane_state():
    return {"left": None, "right": None, "miss_left": 0, "miss_right": 0}


def _blend_line(prev, cur, alpha):
    cur = np.asarray(cur, dtype=np.float32)
    if prev is None:
        return cur
    return (alpha * cur + (1.0 - alpha) * prev).astype(np.float32)


def smooth_lanes(state, left, right, smoothing):
    alpha = smoothing["line_alpha"]
    hold = int(smoothing["max_missing_frames"])
    if left is not None:
        state["left"] = _blend_line(state["left"], left, alpha)
        state["miss_left"] = 0
    else:
        state["miss_left"] += 1
        if state["miss_left"] > hold:
            state["left"] = None
    if right is not None:
        state["right"] = _blend_line(state["right"], right, alpha)
        state["miss_right"] = 0
    else:
        state["miss_right"] += 1
        if state["miss_right"] > hold:
            state["right"] = None
    return state["left"], state["right"]


def lane_heading(left, right):
    if left is not None and right is not None:
        x_bot = (float(left[0]) + float(right[0])) / 2.0
        x_top = (float(left[2]) + float(right[2])) / 2.0
        y_bot = (float(left[1]) + float(right[1])) / 2.0
        y_top = (float(left[3]) + float(right[3])) / 2.0
    elif left is not None or right is not None:
        one = left if left is not None else right
        x_bot, y_bot, x_top, y_top = float(one[0]), float(one[1]), float(one[2]), float(one[3])
    else:
        return None
    dy = y_bot - y_top
    if dy <= 1e-6:
        return None
    angle = math.degrees(math.atan2(x_top - x_bot, dy))
    return (x_bot, y_bot), (x_top, y_top), angle


def draw_heading(img, heading):
    if heading is None:
        return img
    (cbx, cby), (ctx, cty), angle = heading
    cv2.arrowedLine(img, (int(cbx), int(cby)), (int(ctx), int(cty)),
                    (0, 165, 255), 3, tipLength=0.15)
    return img


STATUS_COLOR = {"valid": (0, 200, 0), "low_confidence": (0, 200, 200), "invalid": (0, 0, 220)}


def draw_overlay(frame, poly, raw_lines, left, right, lane_center, status, reason, cfg):
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.fillPoly(overlay, [poly], (60, 60, 0))
    out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
    cv2.polylines(out, [poly], True, (0, 255, 255), 1)
    if cfg["debug"]["show_raw_lines"] and len(raw_lines):
        draw_lines(out, raw_lines, (0, 0, 255), 1)
    for ln in (left, right):
        if ln is not None:
            x1, y1, x2, y2 = ln
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
    cam_x = w // 2
    cv2.line(out, (cam_x, 0), (cam_x, h), (255, 255, 255), 1)
    if lane_center is not None:
        cx = int(lane_center)
        cv2.circle(out, (cx, h - 20), 6, (255, 0, 255), -1)
        cv2.line(out, (cam_x, h - 20), (cx, h - 20), (255, 0, 255), 2)
    color = STATUS_COLOR.get(status, (200, 200, 200))
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    label = status if not reason else ("%s: %s" % (status, reason))
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def apply_steer_boost(steer, cfg):
    ctrl = cfg["control"]
    db = float(ctrl.get("steer_boost_deadband", 0.05))
    boost = float(ctrl.get("steer_min_boost", 0.0))
    if boost <= 0.0 or abs(steer) < db:
        return _clip(steer, -1.0, 1.0)
    return _clip(steer + (boost if steer > 0 else -boost), -1.0, 1.0)


def compute_drive(center_x, heading_deg, status, w, cfg):
    ctrl = cfg["control"]
    if center_x is None:
        return 0.0, 0.0
    off_n = _clip((center_x - w / 2.0) / (w / 2.0), -1.0, 1.0)
    head_n = _clip(heading_deg / float(ctrl["max_heading_deg"]), -1.0, 1.0)
    raw = ctrl["k_offset"] * off_n + ctrl["k_heading"] * head_n
    steer = ctrl["steer_sign"] * (ctrl["steer_gain"] * raw + ctrl["steer_bias"])
    steer = apply_steer_boost(_clip(steer, -1.0, 1.0), cfg)
    thr = ctrl["base_throttle"]
    if status == "low_confidence":
        thr *= ctrl["slow_factor"]
    if status == "invalid":
        thr = 0.0
    mt = ctrl.get("min_throttle", 0.0)
    if thr > 0.0:
        thr = max(thr, mt)
    thr = ctrl["throttle_sign"] * _clip(thr, -ctrl["max_throttle"], ctrl["max_throttle"])
    return steer, thr


def estimate_lane(left, right, w, h, cfg, lw=0.0, rw=0.0):
    ctrl = cfg["control"]
    lane_w = ctrl["expected_lane_width_frac"] * w
    margin = float(ctrl.get("single_line_margin_frac", 0.0)) * lane_w
    if left is not None and right is not None:
        lx, rx = float(left[0]), float(right[0])
        center = (lx + rx) / 2.0
        used = "both"
        # Auto anti-clipping: bias the steering center toward the more SOLID line
        # (more line pixels), ramped from a min imbalance so balanced lines give
        # no bias (avoids steering jitter). Keeps the car off the dashed line.
        off = float(ctrl.get("lane_center_offset_frac", 0.0))
        thr = float(ctrl.get("solid_detect_min_imbalance", 0.20))
        if off != 0.0 and (lw + rw) > 0.0:
            imb = abs(lw - rw) / (lw + rw)
            if imb >= thr:
                ramp = (imb - thr) / max(1e-6, 1.0 - thr)
                solid_sign = 1.0 if rw > lw else -1.0  # +1: right line is the solid one
                center += off * abs(rx - lx) * solid_sign * ramp
    elif left is not None:
        center = float(left[0]) + lane_w / 2.0 + margin  # half a lane + margin, away from this line
        used = "left"
    elif right is not None:
        center = float(right[0]) - lane_w / 2.0 - margin
        used = "right"
    else:
        return None
    hd = lane_heading(left, right)
    heading = 0.0 if hd is None else hd[2]
    return center, heading, used


def process_frame(frame, cfg, lane_state, prev_center):
    """Run the full lane-detection pipeline on one BGR frame.

    Returns a dict with the smoothed/raw lines, validation status, the steering
    estimate ``est = (center_x, heading_deg, used)`` (offset already applied in
    ``estimate_lane``), and ``lane_count`` = number of FRESH boundaries this frame.
    """
    th, tw = frame.shape[:2]
    poly = build_roi_polygon(tw, th, cfg["roi"])
    mask = roi_mask((th, tw), poly)
    edges = to_edges(frame, cfg["canny"])
    raw = detect_lines(mask_edges(edges, mask), cfg["hough"])
    fresh_left, fresh_right, lw, rw = filter_lines(raw, tw, th, cfg["filter"], cfg["roi"])
    lane_state["fresh_lw"], lane_state["fresh_rw"] = lw, rw
    left, right = smooth_lanes(lane_state, fresh_left, fresh_right, cfg["smoothing"])
    status, reason, prev_center = validate_lane(left, right, prev_center, tw, cfg["validation"], lw, rw)
    est = estimate_lane(left, right, tw, th, cfg, lw, rw)
    lane_count = int(fresh_left is not None) + int(fresh_right is not None)
    return {
        "raw": raw, "left": left, "right": right,
        "fresh_left": fresh_left, "fresh_right": fresh_right,
        "status": status, "reason": reason, "est": est,
        "lane_count": lane_count, "prev_center": prev_center,
        "edges": edges, "poly": poly,
    }
