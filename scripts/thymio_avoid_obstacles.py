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

class ThymioController:
    def __init__(self):
        """Initialization."""
        self.speed = 0.2

        self.start = time.time()
        self.bridge = CvBridge()
        self.path = os.path.dirname(os.path.abspath(__file__))
        # Init CNN model
        self.model = CNNRegressor(2, False)
        checkpoint = torch.load(self.path+"/{}.tar".format(rospy.get_param('~model')), map_location='cpu')
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

        # initialize the node
        rospy.init_node(
            'thymio_controller1'
        )

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
        # do inference every second
        if milsec > 1:
            # Convert your ROS Image message to OpenCV2
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            im_pil = PILImage.fromarray(img)
            image = self.transform(im_pil)
            with torch.no_grad():
                output = self.model(image.unsqueeze_(0))
                output = output.detach().cpu().numpy()[0]

                # Think that there is a centered object or a pitfall
                if output[1] > abs(output[0]):
                    self.angular_speed = output[1]
                else:
                    self.angular_speed = - output[0]

            self.start = time.time()

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
            velocity = self.get_control(self.speed, self.angular_speed)

            self.velocity_publisher.publish(velocity)

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
