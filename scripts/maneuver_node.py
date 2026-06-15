#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Intersection maneuver driver (merged straight + IMU turns), request-driven.

Idle until a request arrives on `request_topic` (std_msgs/String = straight|left|
right). It then drives that maneuver, publishing Twist on `cmd_topic`, and on
completion publishes the finished action on `done_topic` and returns to idle.

  straight  vision: the road is straight, so a single visible line gives its
            direction. Steer heading-only to cross straight; finish when both
            lines return (`drive/straight_exit_frames`) or on timeout.
  left/right IMU: integrate /imu yaw to a target angle using the turn params in
            config/lane_following.yaml.

Topics (all configurable, absolute by default so the coordinator wiring is trivial)
  sub  request_topic (std_msgs/String)
  sub  ~image_topic  (sensor_msgs/Image)   used by the straight maneuver
  sub  imu_topic     (sensor_msgs/Imu)      used by the turn maneuvers
  pub  cmd_topic     (geometry_msgs/Twist)
  pub  done_topic    (std_msgs/String)      the action that just finished
"""
import math
import threading

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion

from lane_following import pipeline, drive_common


class ManeuverNode(object):
    def __init__(self):
        rospy.init_node("maneuver")
        self.cfg = rospy.get_param("~")
        drive = self.cfg.get("drive", {})
        self.out_cfg = self.cfg.get("output", {})
        self.turn = self.cfg.get("turn", {})

        self.image_topic = drive.get("image_topic", "/csi_cam_0/image_raw")
        self.straight_exit_frames = int(drive.get("straight_exit_frames", 4))
        self.straight_max_seconds = float(drive.get("straight_max_seconds", 6.0))
        self.rate_hz = float(self.turn.get("rate", 30.0))
        self.imu_topic = rospy.get_param("~imu_topic", "/imu")

        request_topic = rospy.get_param("~request_topic", "/maneuver/request")
        cmd_topic = rospy.get_param("~cmd_topic", "/maneuver/cmd")
        done_topic = rospy.get_param("~done_topic", "/maneuver/done")

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.frame_msg = None
        self.frame_seq = 0
        self.current_yaw = None
        self.pending = None
        self.busy = False

        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=1)
        self.done_pub = rospy.Publisher(done_topic, String, queue_size=5)
        rospy.Subscriber(request_topic, String, self.request_cb, queue_size=5)
        rospy.Subscriber(self.image_topic, Image, self.image_cb,
                         queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=10)
        rospy.on_shutdown(self.stop)
        rospy.loginfo("maneuver up: request=%s cmd=%s done=%s",
                      request_topic, cmd_topic, done_topic)

    # ---- inputs -----------------------------------------------------------
    def request_cb(self, msg):
        action = (msg.data or "").strip().lower()
        if self.busy:
            rospy.logwarn("maneuver busy, ignoring request '%s'", action)
            return
        if action not in ("straight", "left", "right"):
            rospy.logwarn("maneuver: unknown action '%s'", action)
            return
        self.pending = action

    def image_cb(self, msg):
        # Store the raw message; convert lazily only while a straight maneuver runs.
        with self.lock:
            self.frame_msg = msg
            self.frame_seq += 1

    def imu_cb(self, msg):
        q = msg.orientation
        _r, _p, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

    # ---- main loop --------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            action, self.pending = self.pending, None
            if action is not None:
                self.busy = True
                try:
                    if action == "straight":
                        self._do_straight()
                    else:
                        self._do_turn(action)
                finally:
                    self.stop()
                    self.busy = False
                    self.done_pub.publish(String(data=action))
                    rospy.loginfo("maneuver done: %s", action)
            rate.sleep()

    def _next_frame(self, last_seq):
        """Block until a frame newer than last_seq arrives; returns (bgr_frame, seq)."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                msg, seq = self.frame_msg, self.frame_seq
            if msg is not None and seq != last_seq:
                return drive_common.cv_image(msg, self.bridge), seq
            rate.sleep()
        return None, last_seq

    def _do_straight(self):
        rospy.loginfo("maneuver: crossing straight")
        lane_state = pipeline.make_lane_state()
        prev_center = None
        back = 0
        last_seq = -1
        t_start = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - t_start).to_sec() > self.straight_max_seconds:
                rospy.logwarn("maneuver: straight timeout")
                return
            frame, last_seq = self._next_frame(last_seq)
            if frame is None:
                return
            h, w = frame.shape[:2]
            result = pipeline.process_frame(frame, self.cfg, lane_state, prev_center)
            prev_center = result["prev_center"]
            back = back + 1 if result["lane_count"] >= 2 else 0
            if back >= self.straight_exit_frames:
                rospy.loginfo("maneuver: lane reacquired, straight complete")
                return
            steer, thr = drive_common.straight_steer(result, w, self.cfg)
            self.cmd_pub.publish(drive_common.to_twist(steer, thr, self.out_cfg))

    def _do_turn(self, direction):
        sign = 1.0 if direction == "left" else -1.0
        target = abs(float(self.turn.get("angle_deg", 90.0)))
        tol = abs(float(self.turn.get("yaw_tolerance_deg", 3.0)))
        slow_at = float(self.turn.get("slow_down_angle_deg", 18.0))
        turn_lin = float(self.turn.get("turn_linear", 0.14))
        turn_ang = float(self.turn.get("turn_angular", 0.75))
        slow_lin = float(self.turn.get("slow_linear", 0.09))
        slow_ang = float(self.turn.get("slow_angular", 0.55))
        max_time = float(self.turn.get("max_turn_time", 6.0))

        # wait for IMU
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown() and self.current_yaw is None:
            rate.sleep()
        if rospy.is_shutdown():
            return
        start_yaw = self.current_yaw
        rospy.loginfo("maneuver: turning %s %.0f deg", direction, target)

        t_start = rospy.Time.now()
        while not rospy.is_shutdown():
            if (rospy.Time.now() - t_start).to_sec() > max_time:
                rospy.logwarn("maneuver: turn timeout")
                return
            delta = math.atan2(math.sin(self.current_yaw - start_yaw),
                               math.cos(self.current_yaw - start_yaw))
            turned = math.degrees(sign * delta)
            remaining = target - turned
            if remaining <= tol:
                rospy.loginfo("maneuver: turn complete (turned %.1f deg)", turned)
                return
            cmd = Twist()
            if remaining <= slow_at:
                cmd.linear.x, cmd.angular.z = slow_lin, sign * slow_ang
            else:
                cmd.linear.x, cmd.angular.z = turn_lin, sign * turn_ang
            self.cmd_pub.publish(cmd)
            rate.sleep()

    def stop(self):
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass


if __name__ == "__main__":
    try:
        ManeuverNode().spin()
    except rospy.ROSInterruptException:
        pass
