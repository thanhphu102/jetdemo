#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lane-following driver (normal road mode).

Subscribes to the camera and runs the vision pipeline while enabled, steering to
keep the car in the middle of the lane (both lines visible). When the fresh line
PAIR has been missing for `drive/intersection_stop_frames` frames it declares an
intersection: it publishes `~event = "intersection"` and a zero Twist, then waits
for the coordinator to disable it and hand off to the maneuver node.

Topics
  sub  ~image_topic (sensor_msgs/Image, default /csi_cam_0/image_raw)
  sub  ~enable      (std_msgs/Bool)   gate from the coordinator
  pub  ~cmd         (geometry_msgs/Twist)
  pub  ~event       (std_msgs/String) "intersection" or "lane_ok"
  pub  ~debug_image (sensor_msgs/Image) when drive/publish_debug is true
"""
import rospy
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from lane_following import pipeline, drive_common


class LaneFollowingNode(object):
    def __init__(self):
        rospy.init_node("lane_following")
        self.cfg = rospy.get_param("~")
        drive = self.cfg.get("drive", {})
        self.out_cfg = self.cfg.get("output", {})
        self.image_topic = drive.get("image_topic", "/csi_cam_0/image_raw")
        self.stop_frames = int(drive.get("intersection_stop_frames", 5))
        self.lane_ok_frames = int(drive.get("lane_ok_frames", 3))
        self.publish_debug = bool(drive.get("publish_debug", False))

        self.bridge = CvBridge()
        self.lane_state = pipeline.make_lane_state()
        self.prev_center = None
        self.missing = 0
        self.lane_ok_count = 0
        self.lane_ok_sent = False
        self.at_intersection = False
        # Run standalone by default; the coordinator drives this explicitly.
        self.enabled = bool(rospy.get_param("~start_enabled", True))

        self.cmd_pub = rospy.Publisher("~cmd", Twist, queue_size=1)
        self.event_pub = rospy.Publisher("~event", String, queue_size=5)
        self.debug_pub = (rospy.Publisher("~debug_image", Image, queue_size=1)
                          if self.publish_debug else None)
        rospy.Subscriber("~enable", Bool, self.enable_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_cb,
                         queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self.stop)
        rospy.loginfo("lane_following up: image=%s stop_frames=%d",
                      self.image_topic, self.stop_frames)

    def enable_cb(self, msg):
        if msg.data and not self.enabled:
            # fresh start when re-enabled after a maneuver
            self.lane_state = pipeline.make_lane_state()
            self.prev_center = None
            self.missing = 0
            self.lane_ok_count = 0
            self.lane_ok_sent = False
            self.at_intersection = False
        self.enabled = bool(msg.data)
        if not self.enabled:
            self.stop()

    def image_cb(self, msg):
        if not self.enabled:
            return
        frame = drive_common.cv_image(msg, self.bridge)
        h, w = frame.shape[:2]
        result = pipeline.process_frame(frame, self.cfg, self.lane_state, self.prev_center)
        self.prev_center = result["prev_center"]

        lane_count = result["lane_count"]
        if lane_count >= 2:
            self.missing = 0
            self.lane_ok_count += 1
        else:
            self.missing += 1
            self.lane_ok_count = 0
            self.lane_ok_sent = False

        if self.missing >= self.stop_frames:
            if not self.at_intersection:
                self.at_intersection = True
                self.event_pub.publish(String(data="intersection"))
                rospy.loginfo("lane_following: intersection (pair lost %d frames)", self.missing)
            self.lane_ok_count = 0
            self.lane_ok_sent = False
            self.cmd_pub.publish(drive_common.to_twist(0.0, 0.0, self.out_cfg))
        else:
            self.at_intersection = False
            if self.lane_ok_count >= self.lane_ok_frames and not self.lane_ok_sent:
                self.event_pub.publish(String(data="lane_ok"))
                self.lane_ok_sent = True
                rospy.loginfo("lane_following: lane_ok (%d stable frames)", self.lane_ok_count)
            est = result["est"]
            if est is None:
                steer, thr = 0.0, 0.0
            else:
                center_x, heading_deg, _used = est
                steer, thr = pipeline.compute_drive(center_x, heading_deg,
                                                    result["status"], w, self.cfg)
            self.cmd_pub.publish(drive_common.to_twist(steer, thr, self.out_cfg))

        if self.debug_pub is not None:
            self._publish_debug(frame, result, msg.header)

    def _publish_debug(self, frame, result, header):
        est = result["est"]
        center = None if est is None else est[0]
        overlay = pipeline.draw_overlay(frame, result["poly"], result["raw"],
                                        result["left"], result["right"], center,
                                        result["status"], result["reason"], self.cfg)
        hd = pipeline.lane_heading(result["left"], result["right"])
        overlay = pipeline.draw_heading(overlay, hd)
        out = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        out.header = header
        self.debug_pub.publish(out)

    def stop(self):
        try:
            self.cmd_pub.publish(drive_common.to_twist(0.0, 0.0, self.out_cfg))
        except Exception:
            pass


if __name__ == "__main__":
    try:
        LaneFollowingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
