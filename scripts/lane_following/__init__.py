"""lane_following: shared ROS-free vision pipeline + small ROS drive helpers.

`pipeline` is a verbatim copy of the tuned notebook pipeline (lane detection,
heading, steering control). `drive_common` holds the thin glue the ROS nodes
share (image conversion, steer/throttle -> Twist scaling, heading-only steering
for the straight maneuver).
"""
