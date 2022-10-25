import sys
import termios
import tty

import select
import rclpy
from rclpy.node import Node
import numpy as np
import cv2 as cv
from enum import Enum
from geometry_msgs.msg import Vector3, Twist
from sensor_msgs.msg import Image

class State(Enum):
    INIT_FINDING_PERSON = 0
    PERSON_FOUND = 1
    FOLLOWING_BALL = 2
    FINDING_PERSON = 3
    FOLLOWING_PERSON = 4
    REACHED_PERSON = 5

class FetchNode(Node):
    key_to_vel = {
        "q": Twist(
            linear=Vector3(x=0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=1.0)
        ),
        "w": Twist(
            linear=Vector3(x=0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0)
        ),
        "e": Twist(
            linear=Vector3(x=0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=-1.0)
        ),
        "a": Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=1.0)
        ),
        "s": Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0)
        ),
        "d": Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=-1.0)
        ),
        "z": Twist(
            linear=Vector3(x=-0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=-1.0)
        ),
        "x": Twist(
            linear=Vector3(x=-0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0)
        ),
        "c": Twist(
            linear=Vector3(x=-0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=1.0)
        ),
    }

    def __init__(self):
        super.__init__('fetch_node')
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.run_loop)
        self.state = State.FINDING_PERSON
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.cam_sub = self.create_subscription(Image, "camera/image_raw", self.process_image, 10)
        self.key = None
        self.settings = termios.tcgetattr(sys.stdin)
        self.image = None
        self.reference_image = None
        self.reference_kps = None
        self.reference_descs = None
        self.initialize_cv_algorithms()

    def initialize_cv_algorithms(self):
        self.orb = cv.ORB_create()
        index_params = dict(
            algoirthm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        self.flann = cv.FlannBasedMatcher(index_params, {})

    def process_image(self, msg: Image):
        self.image = msg.data

    def get_key(self):
        """
        Function to monitor the keyboard and extract any inputs from the user.
        """
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run_loop(self):
        if self.state == State.INIT_FINDING_PERSON:
            try:
                while True:
                    self.key = self.get_key()

                    if self.key == "\x03":
                        self.vel_pub.publish(self.key_to_vel["s"])
                        raise KeyboardInterrupt
                    if self.key in self.key_to_vel.keys():
                        self.vel_pub.publish(self.key_to_vel[self.key])
            except KeyboardInterrupt:
                self.state = State.PERSON_FOUND
        elif self.state == State.PERSON_FOUND:
            self.reference_image = self.image
            self.reference_kps, self.reference_descs = self.orb.detectAndCompute(self.reference_image, None)
            self.state = State.FOLLOWING_BALL
        elif self.state == State.FINDING_PERSON:
            pass
        elif self.state == State.FOLLOWING_PERSON:
            pass
        elif self.state == State.PERSON_FOUND:
            pass

            

    def calculate_keypoints(self, image):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = FetchNode()
    rclpy.spin(node)
    rclpy.shutdown()