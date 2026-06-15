#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bridge YOLO detections from the Py3 detector into ROS (Python 2 node).

Binds a UDP socket, receives detection JSON datagrams from trt_udp_detector.py
(Py3) and republishes each as a std_msgs/String on /yolo_detections. Keeps the ROS
Python 2 environment free of any TensorRT / OpenCV dependency. The decision logic
(later) subscribes to /yolo_detections and parses the JSON.
"""
import socket

import rospy
from std_msgs.msg import String


class YoloBridge(object):
    def __init__(self):
        rospy.init_node("yolo_bridge")
        cfg = rospy.get_param("~")
        tr = cfg.get("transport", {})
        self.port = int(tr.get("det_port", 5005))
        self.topic = rospy.get_param("~topic", "/yolo_detections")

        self.pub = rospy.Publisher(self.topic, String, queue_size=10)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.port))
        self.sock.settimeout(0.5)        # wake periodically to honour shutdown
        rospy.on_shutdown(self.close)
        rospy.loginfo("yolo_bridge: udp :%d -> %s", self.port, self.topic)

    def spin(self):
        while not rospy.is_shutdown():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except socket.error:
                if rospy.is_shutdown():
                    break
                continue
            self.pub.publish(String(data=data.decode("utf-8", "replace")))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        YoloBridge().spin()
    except rospy.ROSInterruptException:
        pass
