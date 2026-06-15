#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Forward camera frames to the Python 3 YOLO detector over UDP (ROS Python 2 node).

The Jetson CSI camera allows ~one consumer, and gscam already owns it
(/csi_cam_0/image_raw). This node subscribes to that topic, throttles to a modest
rate, JPEG-encodes each frame and sends it as a single UDP datagram to the Py3
detector (trt_udp_detector.py). No new runtime deps: cv2 + cv_bridge are already
used by the lane-following Py2 stack.
"""
import socket

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

UDP_SAFE_BYTES = 60000   # keep a JPEG comfortably inside one UDP datagram (<65507)


class CameraUdpForwarder(object):
    def __init__(self):
        rospy.init_node("camera_udp_forwarder")
        cfg = rospy.get_param("~")
        tr = cfg.get("transport", {})
        fr = cfg.get("frame", {})

        self.dst = (tr.get("udp_ip", "127.0.0.1"), int(tr.get("frame_port", 5006)))
        self.image_topic = fr.get("image_topic", "/csi_cam_0/image_raw")
        self.rate_hz = float(fr.get("rate_hz", 12))
        self.jpeg_quality = int(fr.get("jpeg_quality", 70))
        self.max_width = int(fr.get("max_width", 640))
        self.min_period = 1.0 / self.rate_hz if self.rate_hz > 0 else 0.0

        self.bridge = CvBridge()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_sent = rospy.Time(0)

        rospy.Subscriber(self.image_topic, Image, self.image_cb,
                         queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self.close)
        rospy.loginfo("camera_udp_forwarder: %s -> udp %s:%d @ %.1f Hz",
                      self.image_topic, self.dst[0], self.dst[1], self.rate_hz)

    def image_cb(self, msg):
        now = rospy.Time.now()
        if (now - self.last_sent).to_sec() < self.min_period:
            return
        self.last_sent = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = float(self.max_width) / w
            frame = cv2.resize(frame, (self.max_width, int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        data = buf.tobytes()
        if len(data) > UDP_SAFE_BYTES:
            rospy.logwarn_throttle(
                2.0, "frame JPEG %d B > %d B; lower jpeg_quality or max_width (dropped)",
                len(data), UDP_SAFE_BYTES)
            return
        try:
            self.sock.sendto(data, self.dst)
        except socket.error as e:
            rospy.logwarn_throttle(2.0, "UDP send failed: %s", e)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        CameraUdpForwarder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
