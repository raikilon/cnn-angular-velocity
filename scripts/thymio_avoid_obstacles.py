#!/usr/bin/env python
# -*- coding:UTF-8 -*-(add)

import time

import PIL.Image as PILImage
# OpenCV2 for saving an image
import cv2
import rospy
import torch
import torch.nn.functional
import torch.nn.parallel
import torch.optim
import torch.utils.data
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, Vector3
from model.cnn_regressor import CNNRegressor
# ROS Image message
from sensor_msgs.msg import Image
from torchvision import transforms
import os
import numpy as np


class ThymioController:
    def __init__(self):
        """Initialization."""
        # initialize linear speed
        self.speed = 0.2
        # sign by default -1 (left)
        self.sign = -1
        # initialize angular speed
        self.angular_speed = 0
        #  store start time
        self.start = time.time()
        # initialize CV bridge
        self.bridge = CvBridge()
        # path to current directory
        self.path = os.path.dirname(os.path.abspath(__file__))
        # initialize the node
        rospy.init_node(
            'thymio_controller1'
        )
        # Init CNN model
        self.model = CNNRegressor(2, False)
        checkpoint = torch.load(self.path + "/model{}.tar".format(rospy.get_param('~model')), map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        del checkpoint
        torch.cuda.empty_cache()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # get robot name from console
        self.name = rospy.get_param('~robot_name')

        # create velocity publisher
        self.velocity_publisher = rospy.Publisher(
            self.name + '/cmd_vel',  # name of the topic
            Twist,  # message type
            queue_size=10  # queue size
        )

        self.image = rospy.Subscriber(
            self.name + '/camera/image_raw',  # name of the topic
            Image,  # message type
            self.image_callback
        )

        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        # set node update frequency in Hz
        self.rate = rospy.Rate(10)

    def image_callback(self, msg):
        milsec = time.time() - self.start
        # do inference every 0.5 second
        if milsec > 0.5:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # convert color to RGB
            img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            # get image as PIL image
            im_pil = PILImage.fromarray(img)
            # transform PIL image to prepare for CNN
            image = self.transform(im_pil)
            with torch.no_grad():
                # get CNN output
                output = self.model(image.unsqueeze_(0))
                output = output.detach().cpu().numpy()[0]

                # Think that there is a centered object or a pitfall
                if output[1] > abs(output[0]):
                    # if output 1 larger than threshold 0.3
                    # turn with angular velocity set to output 1 scaled by constant 2 in direction given by sign of output 0
                    if abs(output[0]) > 0.3:
                        self.sign = - np.sign(output[0])
                    self.angular_speed = self.sign * output[1]
                # if T value (CNN output 0) bigger or equal C value (CNN output 1)
                else:
                    # turn with angular velocity set to output 
                    self.sign = -1
                    self.angular_speed = - output[0]

            self.start = time.time()

    # function to create Twist given linear and angular velocity
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

    # to run the Thymio and publish its velocity
    def run(self):
        """Controls the Thymio."""

        while not rospy.is_shutdown():
            velocity = self.get_control(self.speed, self.angular_speed)

            self.velocity_publisher.publish(velocity)

            self.rate.sleep()

    # to stop the Thymio
    def stop(self):
        """Stops the robot."""

        self.velocity_publisher.publish(
            Twist()  # set velocities to 0
        )

        self.rate.sleep()


if __name__ == '__main__':
    # create controller
    controller = ThymioController()

    try:
        # run controller
        controller.run()
    except rospy.ROSInterruptException as e:
        pass
