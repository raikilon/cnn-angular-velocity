#!/usr/bin/env python
# -*- coding:UTF-8 -*-(add)

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from geometry_msgs.msg import Pose, Twist, Vector3
from tf.transformations import euler_from_quaternion
import numpy as np
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import os.path
import time


class ThymioController:
    FORWARD = 1
    ROTATING = 2
    ROTATING_ORTHOGONAL = 3
    DONE = 5
    count = 0

    def __init__(self):
        """Initialization."""
        self.ranges = {}
        self.angular_speed = 0.2
        self.speed = 0.2
        self.sign = 2
        self.min_wall_distance = 0.04
        self.status = ThymioController.FORWARD
        self.start = time.time()
        # Instantiate CvBridge
        self.bridge = CvBridge()
        self.image_count = 0
        self.sensors = []
        # initialize the node
        rospy.init_node(
            'thymio_controller' + str(ThymioController.count)  # name of the node
        )

        ThymioController.count = ThymioController.count + 1
        self.name = rospy.get_param('~robot_name')

        # log robot name to console
        rospy.loginfo('Controlling %s' % self.name)

        # create velocity publisher
        self.velocity_publisher = rospy.Publisher(
            self.name + '/cmd_vel',  # name of the topic
            Twist,  # message type
            queue_size=10  # queue size
        )

        self.prox_center_sub = rospy.Subscriber(
            self.name + '/proximity/left',  # name of the topic
            Range,  # message type
            self.sense_prox,  # function that hanldes incoming messages
            "left"
        )

        self.prox_center_sub = rospy.Subscriber(
            self.name + '/proximity/center_left',  # name of the topic
            Range,  # message type
            self.sense_prox,  # function that hanldes incoming messages
            "center_left"
        )

        self.prox_center_sub = rospy.Subscriber(
            self.name + '/proximity/center',  # name of the topic
            Range,  # message type
            self.sense_prox,  # function that hanldes incoming messages
            "center"
        )

        self.prox_center_sub = rospy.Subscriber(
            self.name + '/proximity/center_right',  # name of the topic
            Range,  # message type
            self.sense_prox,  # function that hanldes incoming messages
            "center_right"
        )

        self.prox_center_sub = rospy.Subscriber(
            self.name + '/proximity/right',  # name of the topic
            Range,  # message type
            self.sense_prox,  # function that hanldes incoming messages
            "right"
        )

        self.image = rospy.Subscriber(
            self.name + '/camera/image_raw',  # name of the topic
            Image,  # message type
            self.image_callback
        )

        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        # initialize pose to (X=0, Y=0, theta=0)
        self.pose = Pose()

        # initialize linear and angular velocities to 0
        self.velocity = Twist()

        # set node update frequency in Hz
        self.rate = rospy.Rate(10)

    def image_callback(self, msg):

        milsec = time.time() - self.start

        if milsec > 1:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Save your OpenCV2 image as a jpeg
            cv2.imwrite("data/imgs/{}.jpeg".format(self.image_count), cv2_img)
            if os.path.isfile('data/sensor_data.npy'):
                data = np.load("data/sensor_data.npy")
                data = np.append(data, self.ranges)
            else:
                data = self.ranges

            np.save("data/sensor_data.npy", data)
            self.start = time.time()
            self.image_count += 1

    def sense_prox(self, data, topic):
        """Updates robot pose and velocities, and logs pose to console."""
        sensor_range = data.range
        self.ranges[topic] = sensor_range
        if sensor_range < 0.2:
            if self.status == ThymioController.FORWARD:
                dif = self.ranges["left"] - self.ranges["right"]

                if np.isclose(0, dif, atol=0.005):
                    self.status = ThymioController.ROTATING_ORTHOGONAL
                else:
                    self.status = ThymioController.ROTATING
                velocity = self.get_control(0, 0)
                self.velocity_publisher.publish(velocity)

    def get_control(self, vel, ang):
        return Twist(
            linear=Vector3(
                vel,  # moves forward .2 m/s
                .0,
                .0,
            ),
            angular=Vector3(
                .0,
                .0,
                ang
            )
        )

    def run(self):
        """Controls the Thymio."""

        while not rospy.is_shutdown():
            if self.status == ThymioController.FORWARD:
                # decide control action
                velocity = self.get_control(self.speed, 0)

                # publish velocity message
                self.velocity_publisher.publish(velocity)
            elif self.status == ThymioController.ROTATING:
                dif = self.ranges["left"] - self.ranges["right"]

                if np.isclose(0, dif, atol=0.005):
                    self.velocity_publisher.publish(self.get_control(0, 0))
                    t0 = rospy.Time.now().to_sec()
                    current_angle = 0
                    angle = np.deg2rad(np.random.randint(5, 45))

                    while current_angle < angle:
                        self.velocity_publisher.publish(self.get_control(0, self.angular_speed * self.sign))
                        t1 = rospy.Time.now().to_sec()
                        current_angle = self.angular_speed * (t1 - t0)

                    self.sign = 2
                    self.status = ThymioController.FORWARD
                else:
                    if self.sign == 2:
                        self.sign = np.sign(dif)
                    # publish velocity message

                    self.velocity_publisher.publish(self.get_control(0, self.angular_speed * self.sign))

            # When Thymio is orthogonal to an object
            elif self.status == ThymioController.ROTATING_ORTHOGONAL:

                self.velocity_publisher.publish(self.get_control(0, 0))
                t0 = rospy.Time.now().to_sec()
                current_angle = 0
                angle = np.deg2rad(90 + np.random.randint(5, 45))
                rand_dir = 1 if np.random.random() < 0.5 else -1
                while current_angle < angle:
                    self.velocity_publisher.publish(self.get_control(0, self.angular_speed * rand_dir))
                    t1 = rospy.Time.now().to_sec()
                    current_angle = self.angular_speed * (t1 - t0)

                self.status = ThymioController.FORWARD

            elif self.status == ThymioController.DONE:
                self.velocity_publisher.publish(self.get_control(0, self.angular_speed))
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
        pass
