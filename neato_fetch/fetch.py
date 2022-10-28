import sys
import termios
import tty

import select

from torch import fake_quantize_per_tensor_affine
import rclpy
from rclpy.node import Node
import numpy as np
import cv2 as cv
from enum import Enum
from geometry_msgs.msg import Vector3, Twist
from sensor_msgs.msg import Image
from neato2_interfaces.msg import Bump
from cv_bridge import CvBridge

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

    NUM_MATCHES_THRESHOLD = 10
    GOOD_MATCH_THRESHOLD = 0.75

    def __init__(self):
        super().__init__('fetch_node')
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.run_loop)
        self.state = State.PERSON_FOUND
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.cam_sub = self.create_subscription(Image, "camera/image_raw", self.process_image, 10)
        self.bump_sub = self.create_subscription(Bump, "bump", self.process_bump, 10)
        self.debug_img_pub = self.create_publisher(Image, "cv_debug", 10)
        self.key = None
        self.settings = termios.tcgetattr(sys.stdin)
        self.image = None
        self.reference_image = None
        self.reference_kps = None
        self.reference_descs = None
        self.bump = False
        self.bridge = CvBridge()
        self.initialize_cv_algorithms()

    def process_bump(self, msg: Bump):
        if msg.left_front or msg.right_front or msg.left_side or msg.right_side:
            self.bump = True
        else:
            self.bump = False

    def initialize_cv_algorithms(self):
        self.orb = cv.ORB_create()
        index_params = dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        self.flann = cv.FlannBasedMatcher(index_params, {})

    def process_image(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def get_key(self):
        """
        Function to monitor the keyboard and extract any inputs from the user.
        """
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def record_reference_img(self):
        print(np.shape(self.image))
        self.reference_image = self.image
        gray_img = cv.cvtColor(self.reference_image, cv.COLOR_BGR2GRAY)
        self.reference_kps, self.reference_descs = self.orb.detectAndCompute(gray_img, None)
        
    def drive_to_person(self):
        try:
            while True:
                self.key = self.get_key()
                if self.key == "\x03":
                    self.vel_pub.publish(self.key_to_vel["s"])
                    raise KeyboardInterrupt
                if self.key in self.key_to_vel.keys():
                    print(self.key_to_vel[self.key])
                    self.vel_pub.publish(self.key_to_vel[self.key])
        except KeyboardInterrupt:
            return

    def look_for_person(self):
        gray_img = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        curr_kps, curr_descs = self.orb.detectAndCompute(gray_img, None)

        matches = self.flann.knnMatch(self.reference_descs, curr_descs, 2)

        num_matches = 0

        matchesMask = [[0,0] for i in range(len(matches))]

        if len(np.shape(matches)) != 2:
            return
        for i, (m, n) in enumerate(matches):
            if m.distance < self.GOOD_MATCH_THRESHOLD * n.distance:
                num_matches += 1
                matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = cv.DrawMatchesFlags_DEFAULT)  
        corrected_img = cv.drawMatchesKnn(self.reference_image, self.reference_kps, self.image, curr_kps, matches, None, **draw_params)
        cv.imshow('test', corrected_img)
        cv.waitKey(1)
        # if num_matches > self.NUM_MATCHES_THRESHOLD:
        #     self.vel_pub.publish(self.key_to_vel["s"])
            # self.state = State.FOLLOWING_PERSON
        # else:
            # self.vel_pub.publish(self.key_to_vel["d"])
        print(num_matches)

    def drive_back_to_person(self):
        self.vel_pub.publish(self.key_to_vel["w"])
        return


    def run_loop(self):
        print(self.state)
        if self.state == State.INIT_FINDING_PERSON:
            self.drive_to_person()
            self.state = State.PERSON_FOUND
        elif self.state == State.PERSON_FOUND:
            self.record_reference_img()
            # self.state = State.FOLLOWING_BALL
            self.state = State.FINDING_PERSON
        elif self.state == State.FOLLOWING_BALL:
            pass
        elif self.state == State.FINDING_PERSON:
            self.look_for_person()
        elif self.state == State.FOLLOWING_PERSON:
            self.drive_back_to_person()
        elif self.state == State.PERSON_FOUND:
            pass

    def calculate_keypoints(self, image):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = FetchNode()
    rclpy.spin(node)
    rclpy.shutdown()