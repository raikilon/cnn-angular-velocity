#!/usr/bin/env python
# -*- coding:UTF-8 -*-(add)

import os
import os.path
import time

# OpenCV2 for saving an image
import cv2
import numpy as np
import rospy
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
# ROS Image message
from sensor_msgs.msg import Image


class ThymioController:

    def __init__(self):
        """Initialization."""
        self.start = time.time()
        self.bridge = CvBridge()
        self.image_count = 0
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.data = None

        if not os.path.exists(self.path + "/data"):
            os.makedirs(self.path + "/data")

        if not os.path.exists(self.path + "/data/imgs"):
            os.makedirs(self.path + "/data/imgs")

        # initialize the node
        rospy.init_node(
            'thymio_controller'
        )

        self.name = rospy.get_param('~robot_name')

        self.image = rospy.Subscriber(
            self.name + '/camera/image_raw',  # name of the topic
            Image,  # message type
            self.image_callback
        )

        # create velocity publisher
        self.velocity_publisher = rospy.Subscriber(
            self.name + '/cmd_vel',  # name of the topic
            Twist,  # message type
            self.save_vel  # queue size
        )

        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        # set node update frequency in Hz
        self.rate = rospy.Rate(10)

    def save_vel(self, mgs):
        self.angular_vel = mgs.angular.z

    def image_callback(self, msg):

        milsec = time.time() - self.start

        # Save picture and angular velocity every second
        if milsec > 1:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Save your OpenCV2 image as a jpeg
            cv2.imwrite(self.path + "/data/imgs/{}.jpeg".format(self.image_count), cv2_img)
            if self.data is not None:
                self.data = np.append(self.data, self.angular_vel)
            else:
                self.data = self.angular_vel

            self.start = time.time()
            self.image_count += 1

    def run(self):
        """Controls the Thymio."""

        while not rospy.is_shutdown():
            # sleep until next step
            self.rate.sleep()

    def stop(self):
        """Stops the robot."""

        self.velocity_publisher.publish(
            Twist()  # set velocities to 0
        )

        self.rate.sleep()


if __name__ == '__main__':
    controller = ThymioController()

    try:
        controller.run()
    except rospy.ROSInterruptException as e:
        # The data is saved if the user exit when in forward state with CTRL+C
        np.save(controller.path + "/data/angular_vel.npy", controller.data)
        pass
