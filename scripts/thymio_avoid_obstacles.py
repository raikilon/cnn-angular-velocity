#!/usr/bin/env python
# -*- coding:UTF-8 -*-(add)

import rospy
from geometry_msgs.msg import Pose, Twist, Vector3
import numpy as np
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge
# OpenCV2 for saving an image
import cv2
import time
import torch
import torch.nn.parallel
import torch.nn.functional
import torch.optim
import torch.utils.data
from torchvision import transforms
import PIL.Image as PILImage
from model.cnn_regressor import CNNRegressor


class ThymioController:
    FORWARD = 1
    ROTATING = 2
    count = 0

    def __init__(self):
        """Initialization."""
        self.angular_speed = 0
        self.speed = 0.2
        self.sign = 2
        self.status = ThymioController.FORWARD
        self.start = time.time()
        # Instantiate CvBridge
        self.bridge = CvBridge()

        self.current_angle = 0
        self.angle = np.deg2rad(20)
        self.sign = 0

        # Init CNN model
        self.model = CNNRegressor(2, False)
        checkpoint = torch.load("best_model.pth.tar", map_location='cpu')
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
            img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            im_pil = PILImage.fromarray(img)
            image = self.transform(im_pil)
            with torch.no_grad():
                output = self.model(image.unsqueeze_(0))
                output = output.detach().cpu().numpy()[0]

                # Think that there is a centered object
                if output[1] > abs(output[0]):
                    print("Center")
                    self.angular_speed = output[1]
                else:
                    self.angular_speed = - output[0]
                    if output[0] > 0:
                        print("LEFT")
                    else:
                        print("RIGHT")

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
            if self.status == ThymioController.FORWARD:
                # decide control action
                velocity = self.get_control(self.speed, self.angular_speed)

                # publish velocity message
                self.velocity_publisher.publish(velocity)

                t0 = rospy.Time.now().to_sec()
            elif self.status == ThymioController.ROTATING:

                if self.current_angle < self.angle:
                    self.velocity_publisher.publish(self.get_control(self.speed, self.angular_speed * self.sign))
                    t1 = rospy.Time.now().to_sec()
                    self.current_angle = self.angular_speed * (t1 - t0)
                else:
                    self.status = ThymioController.FORWARD
                    self.current_angle = 0
                    self.sign = 0

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
