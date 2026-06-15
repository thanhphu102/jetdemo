#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step-by-step state machine controller for JetRacer.

The controller is the only publisher of /cmd_vel in step-drive mode. It enables
the lane follower, requests maneuvers, relays the active Twist source, and stops
cleanly after each commanded action.
"""
import json

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String


VALID_ACTIONS = ("straight", "left", "right")
MANEUVER_STATES = {
    "GO_STRAIGHT": "straight",
    "TURN_LEFT": "left",
    "TURN_RIGHT": "right",
}


class StateMachineControllerNode(object):
    def __init__(self):
        rospy.init_node("state_machine_controller")
        cfg = rospy.get_param("~")
        sm = cfg.get("state_machine", {})

        self.rate_hz = float(sm.get("rate", 30.0))
        self.cmd_timeout = float(sm.get("cmd_timeout", 0.5))
        self.recover_timeout = float(sm.get("recover_timeout", 5.0))
        self.lane_lost_timeout = float(sm.get("lane_lost_timeout", 4.0))
        self.no_dirs_policy = sm.get("no_dirs_policy", "stop")
        if self.no_dirs_policy not in ("stop", "straight"):
            rospy.logwarn("invalid no_dirs_policy '%s', using stop", self.no_dirs_policy)
            self.no_dirs_policy = "stop"

        self.sign_conf_min = float(sm.get("sign_conf_min", 0.45))
        self.sign_debounce_frames = max(1, int(sm.get("sign_debounce_frames", 3)))
        self.light_debounce_frames = max(1, int(sm.get("light_debounce_frames", 3)))
        self.class_map = sm.get("class_map", {
            "0_Go_straight": "straight",
            "1_Turn_left": "left",
            "2_Turn_right": "right",
            "3_Prohibited": "prohibited",
        })
        light_classes = sm.get("light_classes", {})
        self.red_classes = set(light_classes.get("red", ["4_Red_light"]))
        self.green_classes = set(light_classes.get("green", ["5_Green_light"]))

        self.cmd_vel_pub = rospy.Publisher(
            rospy.get_param("~cmd_vel_topic", "/cmd_vel"), Twist, queue_size=1)
        self.enable_pub = rospy.Publisher(
            rospy.get_param("~follow_enable_topic", "/lane_following/enable"),
            Bool, queue_size=1, latch=True)
        self.request_pub = rospy.Publisher(
            rospy.get_param("~maneuver_request_topic", "/maneuver/request"),
            String, queue_size=5)
        self.state_pub = rospy.Publisher(
            sm.get("state_topic", "/sm/state"), String, queue_size=1, latch=True)

        rospy.Subscriber(sm.get("command_topic", "/sm/command"),
                         String, self._command_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param("~follow_cmd_topic", "/lane_following/cmd"),
                         Twist, self._follow_cmd_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~follow_event_topic", "/lane_following/event"),
                         String, self._follow_event_cb, queue_size=5)
        rospy.Subscriber(rospy.get_param("~maneuver_cmd_topic", "/maneuver/cmd"),
                         Twist, self._maneuver_cmd_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~maneuver_done_topic", "/maneuver/done"),
                         String, self._maneuver_done_cb, queue_size=5)
        rospy.Subscriber(sm.get("yolo_topic", "/yolo_detections"),
                         String, self._yolo_cb, queue_size=5)
        rospy.Subscriber(sm.get("intersection_topic", "/perception/intersection"),
                         String, self._intersection_cb, queue_size=5)

        self.state = "IDLE"
        self.prev_state = "IDLE"
        self.error_reason = ""
        self.commanded_action = None
        self.pending_action = None
        self.in_intersection = False
        self.at_intersection = False
        self.dirs = None
        self.intersection_stamp = rospy.Time(0)
        self.follow_cmd = None
        self.follow_stamp = rospy.Time(0)
        self.maneuver_cmd = None
        self.maneuver_stamp = rospy.Time(0)
        self.follow_event = None
        self.lane_ok = False
        self.maneuver_done = None
        self.light = None
        self._sign_candidate = None
        self._sign_count = 0
        self._light_candidate = None
        self._light_count = 0
        self.state_enter = rospy.Time.now()
        self._stop_published_once = False

        rospy.on_shutdown(self.stop)
        rospy.sleep(0.3)
        self._set_follow(False)
        self.state_pub.publish(String(data=self.state))
        rospy.loginfo("state_machine_controller up")

    # ---- inputs -----------------------------------------------------------
    def _command_cb(self, msg):
        cmd = (msg.data or "").strip().lower()
        if cmd == "stop":
            self._transition("STOP")
            return
        if cmd == "reset":
            self.error_reason = ""
            self.commanded_action = None
            self.pending_action = None
            self.in_intersection = False
            self._transition("IDLE")
            return
        if cmd == "lane_follow":
            if self.state == "ERROR":
                rospy.logwarn("SMC: reset required before lane_follow")
                return
            self.commanded_action = None
            self._transition("LANE_FOLLOW_STEP")
            return

        action_map = {
            "go_straight": "straight",
            "turn_left": "left",
            "turn_right": "right",
        }
        if cmd not in action_map:
            rospy.logwarn("SMC: ignoring unknown command '%s'", cmd)
            return

        if self.state == "ERROR":
            rospy.logwarn("SMC: reset required before command '%s'", cmd)
            return

        self.commanded_action = action_map[cmd]
        if self.state in ("IDLE", "STOP", "ERROR"):
            self._transition("APPROACH_INTERSECTION")
        elif self.state == "LANE_FOLLOW_STEP":
            rospy.loginfo("SMC: queued commanded action '%s'", self.commanded_action)
        elif self.state == "APPROACH_INTERSECTION":
            rospy.loginfo("SMC: updated commanded action '%s'", self.commanded_action)
        else:
            rospy.logwarn("SMC: busy in %s, queued action '%s'",
                          self.state, self.commanded_action)

    def _follow_cmd_cb(self, msg):
        self.follow_cmd = msg
        self.follow_stamp = rospy.Time.now()

    def _maneuver_cmd_cb(self, msg):
        self.maneuver_cmd = msg
        self.maneuver_stamp = rospy.Time.now()

    def _follow_event_cb(self, msg):
        event = (msg.data or "").strip().lower()
        if event == "intersection":
            self.follow_event = event
            self.at_intersection = True
        elif event == "lane_ok":
            self.lane_ok = True

    def _maneuver_done_cb(self, msg):
        self.maneuver_done = (msg.data or "").strip().lower()

    def _intersection_cb(self, msg):
        try:
            data = json.loads(msg.data or "{}")
        except (TypeError, ValueError) as exc:
            rospy.logwarn("SMC: invalid intersection JSON: %s", exc)
            return
        self.at_intersection = bool(data.get("at_intersection", False))
        dirs = data.get("dirs", None)
        if isinstance(dirs, list):
            clean = []
            for item in dirs:
                value = str(item).strip().lower()
                if value in VALID_ACTIONS and value not in clean:
                    clean.append(value)
            self.dirs = clean
        elif dirs is None:
            self.dirs = None
        else:
            rospy.logwarn("SMC: ignoring invalid dirs field: %s", dirs)
        self.intersection_stamp = rospy.Time.now()

    def _yolo_cb(self, msg):
        try:
            data = json.loads(msg.data or "{}")
        except (TypeError, ValueError) as exc:
            rospy.logwarn("SMC: invalid YOLO JSON: %s", exc)
            return

        detections = data.get("detections", [])
        best_sign = None
        best_sign_conf = -1.0
        best_light = None
        best_light_conf = -1.0

        for det in detections:
            if not isinstance(det, dict):
                continue
            name = det.get("class_name", "")
            try:
                conf = float(det.get("conf", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < self.sign_conf_min:
                continue

            mapped = self.class_map.get(name)
            if mapped in ("straight", "left", "right", "prohibited"):
                if conf > best_sign_conf:
                    best_sign = mapped
                    best_sign_conf = conf
            if name in self.red_classes and conf > best_light_conf:
                best_light = "red"
                best_light_conf = conf
            elif name in self.green_classes and conf > best_light_conf:
                best_light = "green"
                best_light_conf = conf

        if best_sign is not None:
            self._debounce_sign(best_sign)
        self._debounce_light(best_light)

    # ---- state helpers ----------------------------------------------------
    def _transition(self, new_state, reason=""):
        if self.state == new_state:
            return
        old_state = self.state
        self.state = new_state
        self.state_enter = rospy.Time.now()
        self._stop_published_once = False
        rospy.loginfo("SMC: %s -> %s", old_state, new_state)

        if new_state == "IDLE":
            self._set_follow(False)
            self.commanded_action = None
            self.pending_action = None
            self.follow_event = None
            self.lane_ok = False
            self.maneuver_done = None
        elif new_state == "LANE_FOLLOW_STEP":
            self._set_follow(True)
            self.follow_cmd = None
            self.follow_event = None
            self.lane_ok = False
        elif new_state == "APPROACH_INTERSECTION":
            self._set_follow(True)
            self.follow_event = None
            self.lane_ok = False
        elif new_state in MANEUVER_STATES:
            action = MANEUVER_STATES[new_state]
            self.in_intersection = True
            self._set_follow(False)
            self.maneuver_cmd = None
            self.maneuver_done = None
            self.request_pub.publish(String(data=action))
            rospy.loginfo("SMC: requested maneuver '%s'", action)
        elif new_state == "RECOVER_LANE":
            self._set_follow(True)
            self.follow_cmd = None
            self.lane_ok = False
            self.follow_event = None
        elif new_state == "RED_LIGHT_STOP":
            self.prev_state = old_state
            self._set_follow(False)
        elif new_state == "STOP":
            self._set_follow(False)
            self.commanded_action = None
            self.pending_action = None
            self.follow_event = None
            self.lane_ok = False
            self.maneuver_done = None
            self.at_intersection = False
            self.dirs = None
        elif new_state == "ERROR":
            self._set_follow(False)
            self.error_reason = reason or self.error_reason or "unknown"
            rospy.logerr("SMC ERROR: %s", self.error_reason)

    def _debounce_sign(self, value):
        if value == self._sign_candidate:
            self._sign_count += 1
        else:
            self._sign_candidate = value
            self._sign_count = 1
        if self._sign_count >= self.sign_debounce_frames:
            if self.pending_action != value:
                rospy.loginfo("SMC: pending action from sign = %s", value)
            self.pending_action = value

    def _debounce_light(self, value):
        if value == self._light_candidate:
            self._light_count += 1
        else:
            self._light_candidate = value
            self._light_count = 1
        if self._light_count >= self.light_debounce_frames:
            if self.light != value:
                rospy.loginfo("SMC: traffic light = %s", value or "none")
            self.light = value

    def _set_follow(self, on):
        self.enable_pub.publish(Bool(data=bool(on)))

    def _state_age(self):
        return (rospy.Time.now() - self.state_enter).to_sec()

    def _cmd_fresh(self, stamp):
        return (rospy.Time.now() - stamp).to_sec() <= self.cmd_timeout

    def _source_cmd(self, source):
        if source == "follow":
            cmd, stamp = self.follow_cmd, self.follow_stamp
        else:
            cmd, stamp = self.maneuver_cmd, self.maneuver_stamp
        if cmd is None or not self._cmd_fresh(stamp):
            return Twist()
        return cmd

    def _maneuver_state_for(self, action):
        if action == "straight":
            return "GO_STRAIGHT"
        if action == "left":
            return "TURN_LEFT"
        if action == "right":
            return "TURN_RIGHT"
        return None

    def _intersection_ready(self):
        return self.at_intersection or self.follow_event == "intersection"

    def _choose_action(self):
        if self.commanded_action is not None:
            return self.commanded_action
        return self.pending_action

    def _validate_action(self, action):
        if action == "prohibited":
            return "prohibited"
        if action not in VALID_ACTIONS:
            return None
        if self.dirs is None:
            if self.no_dirs_policy == "straight":
                return "straight"
            return None
        if action in self.dirs:
            return action
        rospy.logwarn("SMC: action '%s' is not available in dirs=%s", action, self.dirs)
        return None

    def _prohibited_choice(self):
        if self.dirs is None:
            return None
        if "right" in self.dirs:
            return "right"
        if "left" in self.dirs:
            return "left"
        return None

    def _error(self, reason):
        self._transition("ERROR", reason)

    # ---- main loop --------------------------------------------------------
    def _step(self):
        if self.state == "IDLE":
            return

        if self.state == "STOP":
            if self._stop_published_once:
                self.in_intersection = False
                self._transition("IDLE")
            return

        if self.state == "ERROR":
            return

        if self.state == "RED_LIGHT_STOP":
            if self.light != "red":
                target = self.prev_state if self.prev_state else "LANE_FOLLOW_STEP"
                self._transition(target)
            return

        if self.state == "LANE_FOLLOW_STEP":
            if self.light == "red" and not self.in_intersection:
                self._transition("RED_LIGHT_STOP")
                return
            if self._intersection_ready():
                self._transition("APPROACH_INTERSECTION")
                return
            if self._state_age() > self.lane_lost_timeout:
                if self.follow_cmd is None or not self._cmd_fresh(self.follow_stamp):
                    self._error("lane follow command timeout")
            return

        if self.state == "APPROACH_INTERSECTION":
            if self.light == "red" and not self.in_intersection:
                self._transition("RED_LIGHT_STOP")
                return
            if not self._intersection_ready():
                return

            action = self._choose_action()
            if action is None:
                rospy.logwarn("SMC: no action available at intersection")
                self._transition("STOP")
                return

            action = self._validate_action(action)
            if action == "prohibited":
                self._transition("PROHIBITED_T_DECISION")
                return
            if action is None:
                self._transition("STOP")
                return

            self.commanded_action = None
            self.pending_action = None
            self._transition(self._maneuver_state_for(action))
            return

        if self.state == "PROHIBITED_T_DECISION":
            action = self._prohibited_choice()
            if action is None:
                rospy.logwarn("SMC: prohibited sign but no safe turn available")
                self._transition("STOP")
                return
            self.commanded_action = None
            self.pending_action = None
            self._transition(self._maneuver_state_for(action))
            return

        if self.state in MANEUVER_STATES:
            done = self.maneuver_done
            expected = MANEUVER_STATES[self.state]
            if done in VALID_ACTIONS:
                if done != expected:
                    rospy.logwarn("SMC: maneuver done '%s' while expecting '%s'",
                                  done, expected)
                self._transition("RECOVER_LANE")
            return

        if self.state == "RECOVER_LANE":
            if self.lane_ok:
                self.in_intersection = False
                self._transition("STOP")
                return
            if self._state_age() > self.recover_timeout:
                self._error("recover lane timeout")

    def _active_cmd(self):
        if self.state in ("LANE_FOLLOW_STEP", "APPROACH_INTERSECTION", "RECOVER_LANE"):
            return self._source_cmd("follow")
        if self.state in MANEUVER_STATES:
            return self._source_cmd("maneuver")
        return Twist()

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            self._step()
            cmd = self._active_cmd()
            self.cmd_vel_pub.publish(cmd)
            if self.state == "STOP":
                self._stop_published_once = True
            self.state_pub.publish(String(data=self.state))
            rate.sleep()

    def stop(self):
        try:
            self._set_follow(False)
            self.cmd_vel_pub.publish(Twist())
        except Exception:
            pass


if __name__ == "__main__":
    try:
        StateMachineControllerNode().spin()
    except rospy.ROSInterruptException:
        pass
