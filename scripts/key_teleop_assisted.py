#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013 PAL Robotics SL.
# Released under the BSD License.
#
# Authors:
#   * Siegfried-A. Gevatter

import curses
import math

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
import os

# as in standard teleoperation source code
class Velocity(object):

    def __init__(self, min_velocity, max_velocity, num_steps):
        assert min_velocity > 0 and max_velocity > 0 and num_steps > 0
        self._min = min_velocity
        self._max = max_velocity
        self._num_steps = num_steps
        if self._num_steps > 1:
            self._step_incr = (max_velocity - min_velocity) / (self._num_steps - 1)
        else:
            # If num_steps is one, we always use the minimum velocity.
            self._step_incr = 0

    def __call__(self, value, step):
        """
        Takes a value in the range [0, 1] and the step and returns the
        velocity (usually m/s or rad/s).
        """
        if step == 0:
            return 0

        max_value = self._min + self._step_incr * (step - 1)
        return value * max_value

# as in standard teleoperation source code
class TextWindow():
    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError, 'lineno out of bounds'
        height, width = self._screen.getmaxyx()
        y = (height / self._num_lines) * lineno
        x = 10
        for text in message.split('\n'):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()

# This class we used for our semi-automatic assisted teleoperation
class KeyTeleop():
    _interface = None

    _linear = None
    _angular = None

    def __init__(self, interface):
        self._interface = interface
        # thymio publisher
        self._pub_cmd = rospy.Publisher('/thymio10/cmd_vel', Twist)
        # rate set to 10hz
        self._hz = rospy.get_param('~hz', 10)
        # sum_step counter from openspurce implementation of KeyTeleop. 
        # Initially set to for, we only allow 1 command click while performing an action
        self._num_steps = rospy.get_param('~turbo/steps', 1)
        # min and max value for forward velocity
        forward_min = rospy.get_param('~turbo/linear_forward_min', 0.2)
        forward_max = rospy.get_param('~turbo/linear_forward_max', 1.5)
        self._forward = Velocity(forward_min, forward_max, self._num_steps)
        # min and max value for backward velocity
        backward_min = rospy.get_param('~turbo/linear_backward_min', 0.3)
        backward_max = rospy.get_param('~turbo/linear_backward_max', 1.0)
        self._backward = Velocity(backward_min, backward_max, self._num_steps)
        # min and max value for angular velocity
        angular_min = rospy.get_param('~turbo/angular_min', 0.8)
        angular_max = rospy.get_param('~turbo/angular_max', 3.0)
        self._rotation = Velocity(angular_min, angular_max, self._num_steps)
        # message in terminal while teleoperation is active (assistant passive)
        self.message = 'Use arrow keys to move, space to stop, q to exit.'
        self.prev_message = self.message

        ### Begin: fields for automatic obstacle bypass ###
        # initialize status to FORWARD for Thymio state machine
        self.status = SimpleKeyTeleop.FORWARD
        self.start = time.time()
        #initialize assistant_step value to non zero
        self.assistant_step = 11
        # Instantiate CvBridge
        self.bridge = CvBridge()
        #path to current directory
        self.path = os.path.dirname(os.path.abspath(__file__))

        # Initalize CNN model
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
        # default sign is -1 (left)
        self.sign = -1
        # log robot name to console
        rospy.loginfo('Controlling %s' % self.name)

        # subscriber to raw camera messages
        self.image = rospy.Subscriber(
            self.name + '/camera/image_raw',  # name of the topic
            Image,  # message type
            self.image_callback
        )

        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        ## End: fields for automatic obstacle bypass ###

    # called on each incoming raw camera frame
    def image_callback(self, msg):

        # get time past till last start time count
        milsec = time.time() - self.start
        # if more than 0.4 seconds passed, act as follows
        if milsec > 0.4:
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
                output = output.detach().numpy()[0]
                # act if either C or T gives a stronger signal than threshold 0.4
                if np.max(abs(output)) > 0.4:
                    # if assistant_step non zero, make collision prevention assistant take over th control
                    if self.assistant_step != 0:
                        # remember the linear velocity before the assistant kicked in
                        self.prev_linear = self._forward._min
                        # remember the terminal message while teleoperation was active
                        self.prev_message = self.message
                        # set terminal message to 'Collision prevention assistant taking control'
                        self.message = 'Collision prevention assistant taking control'
                    # automatic steering by assistant using CNN output values
                    self.assistant_steer(output)
                    # set asssitant_step to 0
                    self.assistant_step = 0

                # if assistant_step zero, control Thymio by teleoperation
                elif self.assistant_step == 0:
                    # set angular velocity to 0
                    self._angular = 0
                    # set forward velcity back to velocity before assistant took control
                    self._forward._min = self.prev_linear
                    # set terminal message to message before assistant took control
                    self.message = self.prev_message
                    # set assistant step to non zero
                    self.assistant_step = 1
            self.start = time.time()

            self._publish()

    def assistant_steer(self, output):
        # if CNN output 1 bigger than CNN output 0
        # think that there is a centered object or a pitfall
        if output[1] > abs(output[0]):
            # if output 1 larger than threshold 0.3
            # turn with angular velocity set to output 1 scaled by constant 2 in direction given by sign of output 0
            # print(output[0])
            if abs(output[0]) > 0.3:
                self.sign = - np.sign(output[0])
            self._angular = self.sign * output[1] * 2

         # if T value (CNN output 0) bigger or equal C value (CNN output 1)
        else:
            # turn with angular velocity set to output 0
            self.sign = -1
            self._angular =  - output[0]

    # as in standard teleoperation source code
    def run(self):
        self._linear = 0
        self._angular = 0
        self.prev_linear = self._linear

        rate = rospy.Rate(self._hz)
        while True:
            keycode = self._interface.read_key()
            if keycode:
                if self._key_pressed(keycode):
                    self._publish()
            else:
                self._publish()
                rate.sleep()

    # as in standard teleoperation source code
    def _get_twist(self, linear, angular):
        twist = Twist()
        if linear >= 0:
            twist.linear.x = self._forward(1.0, linear)
        else:
            twist.linear.x = self._backward(-1.0, -linear)
        twist.angular.z = self._rotation(math.copysign(1, angular), abs(angular))
        return twist

    # as in standard teleoperation source code
    def _key_pressed(self, keycode):
        movement_bindings = {
            curses.KEY_UP: (1, 0),
            curses.KEY_DOWN: (-1, 0),
            curses.KEY_LEFT: (0, 1),
            curses.KEY_RIGHT: (0, -1),
        }
        speed_bindings = {
            ord(' '): (0, 0),
        }
        if keycode in movement_bindings:
            acc = movement_bindings[keycode]
            ok = False
            if acc[0]:
                linear = self._linear + acc[0]
                if abs(linear) <= self._num_steps:
                    self._linear = linear
                    ok = True
            if acc[1]:
                angular = self._angular + acc[1]
                if abs(angular) <= self._num_steps:
                    self._angular = angular
                    ok = True
            if not ok:
                self._interface.beep()
        elif keycode in speed_bindings:
            acc = speed_bindings[keycode]
            # Note: bounds aren't enforced here!
            if acc[0] is not None:
                self._linear = acc[0]
            if acc[1] is not None:
                self._angular = acc[1]

        elif keycode == ord('q'):
            rospy.signal_shutdown('Bye')
        else:
            return False

        return True

    # publish as in standard teleoperation source code
    def _publish(self):
        self._interface.clear()
        self._interface.write_line(2, 'Linear: %d, Angular: %d' % (self._linear, self._angular))
        self._interface.write_line(5, self.message)
        self._interface.refresh()

        twist = self._get_twist(self._linear, self._angular)
        self._pub_cmd.publish(twist)

    def stop(self):
        """Stops the robot."""

        self.velocity_publisher.publish(
            Twist()  # set velocities to 0
        )

        self.rate.sleep()

# from standard teleoperation source code, not used in the assisted Thymio implementation
# could be used if one wants to use simpled steering of the Thymio
class SimpleKeyTeleop():
    FORWARD = 1
    ROTATING = 2
    count = 0

    def __init__(self, interface):
        self._interface = interface
        self._pub_cmd = rospy.Publisher('/thymio10/cmd_vel', Twist)

        self._hz = rospy.get_param('~hz', 10)

        self._forward_rate = rospy.get_param('~forward_rate', 0.2)
        self._backward_rate = rospy.get_param('~backward_rate', 0.2)
        self._rotation_rate = rospy.get_param('~rotation_rate', 0.2)
        self._last_pressed = {}
        self._angular = 0
        self._linear = 0

        ### begin: fields for automatic obstacle bypass ###
        self.status = SimpleKeyTeleop.FORWARD
        self.start = time.time()
        # Instantiate CvBridge
        self.bridge = CvBridge()

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

        # SimpleKeyTeleop.count = SimpleKeyTeleop.count + 1
        self.name = rospy.get_param('~robot_name')

        # log robot name to console
        rospy.loginfo('Controlling %s' % self.name)

        self.image = rospy.Subscriber(
            self.name + '/camera/image_raw',  # name of the topic
            Image,  # message type
            self.image_callback
        )

        # tell ros to call stop when the program is terminated
        rospy.on_shutdown(self.stop)

        ## end: fields for automatic obstacle bypass ###

    movement_bindings = {
        curses.KEY_UP: (1, 0),
        curses.KEY_DOWN: (-1, 0),
        curses.KEY_LEFT: (0, 1),
        curses.KEY_RIGHT: (0, -1),
    }

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
                output = output.detach().numpy()[0]
                if np.max(output) > 0.1:
                    # Think that there is a centered object
                    if output[1] > abs(output[0]):
                        # print("Center")
                        # get best direction to go away from centered object
                        sign = - np.sign(output[0])
                        self._angular = sign * output[1]
                    else:
                        self._angular = - output[0]
                        # if output[0] > 0:
                        #     print("LEFT")
                        # else:
                        #     print("RIGHT")

            self.start = time.time()

            # velocity = self.get_control(self._linear, self._angular)

            self._publish()

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
        rate = rospy.Rate(self._hz)
        self._running = True
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
            self._set_velocity()
            self._publish()
            rate.sleep()

    def _get_twist(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        return twist

    def _set_velocity(self):
        now = rospy.get_time()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < 0.4:
                keys.append(a)
        linear = 0.0
        angular = 0.0
        for k in keys:
            l, a = self.movement_bindings[k]
            linear += l
            angular += a
        if linear > 0:
            linear = linear * self._forward_rate
        else:
            linear = linear * self._backward_rate
        angular = angular * self._rotation_rate
        self._angular = angular
        self._linear = linear

    def _key_pressed(self, keycode):
        if keycode == ord('q'):
            self._running = False
            rospy.signal_shutdown('Bye')
        elif keycode in self.movement_bindings:
            self._last_pressed[keycode] = rospy.get_time()

    def _publish(self):
        self._interface.clear()
        self._interface.write_line(2, 'Linear: %f, Angular: %f' % (self._linear, self._angular))
        self._interface.write_line(5, 'Use arrow keys to move, q to exit.')
        self._interface.refresh()

        twist = self._get_twist(self._linear, self._angular)
        self._pub_cmd.publish(twist)

    def stop(self):
        """Stops the robot."""

        self.velocity_publisher.publish(
            Twist()  # set velocities to 0
        )

        self.rate.sleep()


def main(stdscr):
    rospy.init_node('key_teleop')
    app = KeyTeleop(TextWindow(stdscr))
    app.run()


if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except rospy.ROSInterruptException:
        pass
