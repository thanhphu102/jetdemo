#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Coordinator: the single owner of /cmd_vel.

Arbitrates between the lane-following driver (normal road mode) and the maneuver
driver (intersection crossing). It enables exactly one source at a time and
relays that source's latest Twist to /cmd_vel at a fixed rate.

FSM
  FOLLOW   : follower enabled; relay /lane_following/cmd.
             On /lane_following/event == "intersection" -> pop the next route
             action, disable the follower, publish /maneuver/request, go MANEUVER.
  MANEUVER : relay /maneuver/cmd.
             On /maneuver/done -> re-enable the follower, go FOLLOW.

Route is a list of actions (straight|left|right), cycled. Default ["straight"].
"""
import rospy
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist


class CoordinatorNode(object):
    def __init__(self):
        rospy.init_node("lane_coordinator")
        coord = rospy.get_param("~coordinator", {})
        self.route = coord.get("route", ["straight"]) or ["straight"]
        self.rate_hz = float(coord.get("rate", 30.0))
        self.cmd_timeout = float(coord.get("cmd_timeout", 0.5))

        self.cmd_vel_pub = rospy.Publisher(
            rospy.get_param("~cmd_vel_topic", "/cmd_vel"), Twist, queue_size=1)
        self.enable_pub = rospy.Publisher(
            rospy.get_param("~follow_enable_topic", "/lane_following/enable"),
            Bool, queue_size=1, latch=True)
        self.request_pub = rospy.Publisher(
            rospy.get_param("~maneuver_request_topic", "/maneuver/request"),
            String, queue_size=5)

        rospy.Subscriber(rospy.get_param("~follow_cmd_topic", "/lane_following/cmd"),
                         Twist, self._follow_cmd_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~maneuver_cmd_topic", "/maneuver/cmd"),
                         Twist, self._maneuver_cmd_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~follow_event_topic", "/lane_following/event"),
                         String, self._event_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param("~maneuver_done_topic", "/maneuver/done"),
                         String, self._done_cb, queue_size=5)

        self.mode = "FOLLOW"
        self.route_idx = 0
        self.follow_cmd = None
        self.maneuver_cmd = None
        self.follow_stamp = rospy.Time(0)
        self.maneuver_stamp = rospy.Time(0)

        rospy.on_shutdown(self.stop)
        rospy.sleep(0.3)             # let latched enable reach the follower
        self._set_follow(True)
        rospy.loginfo("coordinator up: route=%s", self.route)

    # ---- inputs -----------------------------------------------------------
    def _follow_cmd_cb(self, msg):
        self.follow_cmd = msg
        self.follow_stamp = rospy.Time.now()

    def _maneuver_cmd_cb(self, msg):
        self.maneuver_cmd = msg
        self.maneuver_stamp = rospy.Time.now()

    def _event_cb(self, msg):
        if self.mode != "FOLLOW" or (msg.data or "").strip() != "intersection":
            return
        action = self.route[self.route_idx % len(self.route)]
        self.route_idx += 1
        rospy.loginfo("coordinator: intersection -> maneuver '%s'", action)
        self._set_follow(False)
        self.maneuver_cmd = None
        self.mode = "MANEUVER"
        self.request_pub.publish(String(data=action))

    def _done_cb(self, msg):
        if self.mode != "MANEUVER":
            return
        rospy.loginfo("coordinator: maneuver '%s' done -> follow", (msg.data or "").strip())
        self.follow_cmd = None
        self.mode = "FOLLOW"
        self._set_follow(True)

    # ---- helpers ----------------------------------------------------------
    def _set_follow(self, on):
        self.enable_pub.publish(Bool(data=bool(on)))

    def _active(self):
        now = rospy.Time.now()
        if self.mode == "FOLLOW":
            cmd, stamp = self.follow_cmd, self.follow_stamp
        else:
            cmd, stamp = self.maneuver_cmd, self.maneuver_stamp
        if cmd is None or (now - stamp).to_sec() > self.cmd_timeout:
            return Twist()           # no fresh command -> stop
        return cmd

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self.cmd_vel_pub.publish(self._active())
            rate.sleep()

    def stop(self):
        try:
            self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass


if __name__ == "__main__":
    try:
        CoordinatorNode().spin()
    except rospy.ROSInterruptException:
        pass
