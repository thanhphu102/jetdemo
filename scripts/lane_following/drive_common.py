"""Thin glue shared by the ROS driving nodes.

Kept deliberately ROS-light: ``straight_steer`` is pure (no ROS imports at load
time) so it can be exercised offline against ``pipeline.process_frame``. The
ROS-only bits (``Twist``, ``cv_bridge``) are imported lazily inside the helpers
that need them.
"""
from . import pipeline


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def cv_image(msg, bridge):
    """sensor_msgs/Image -> BGR numpy array."""
    return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def to_twist(steer, throttle, out_cfg):
    """Scale steer/throttle in [-1, 1] to a geometry_msgs/Twist.

    linear.x = throttle * max_linear ; angular.z = steer * max_angular.
    """
    from geometry_msgs.msg import Twist
    t = Twist()
    t.linear.x = float(throttle) * float(out_cfg.get("max_linear", 0.4))
    t.angular.z = float(steer) * float(out_cfg.get("max_angular", 1.0))
    return t


def _cross_throttle(cfg):
    """Base crossing throttle for the straight maneuver, slowed and sign/clip applied."""
    ctrl = cfg["control"]
    slow = float(cfg.get("drive", {}).get("straight_slow_factor", 0.6))
    thr = float(ctrl["base_throttle"]) * slow
    mt = float(ctrl.get("min_throttle", 0.0))
    if thr > 0.0:
        thr = max(thr, mt)
    return float(ctrl["throttle_sign"]) * _clip(thr, -float(ctrl["max_throttle"]),
                                                float(ctrl["max_throttle"]))


def straight_steer(result, w, cfg):
    """Heading-only steering for the STRAIGHT maneuver.

    The road is straight, so a single visible line defines its direction. We steer
    to hold the car parallel to that line (heading -> 0) WITHOUT the single-line
    half-lane lateral guess: pass ``center_x = w/2`` to ``compute_drive`` so the
    offset term zeroes and only ``k_heading`` acts. With no fresh line (``lane_count
    == 0``) we cross blind: zero steer, base throttle.

    ``result`` is the dict returned by ``pipeline.process_frame``.
    Returns ``(steer, throttle)`` both in [-1, 1].
    """
    thr = _cross_throttle(cfg)
    est = result.get("est")
    if result.get("lane_count", 0) <= 0 or est is None:
        return 0.0, thr
    heading_deg = est[1]
    steer, _ = pipeline.compute_drive(w / 2.0, heading_deg,
                                      result.get("status", "valid"), w, cfg)
    return steer, thr
