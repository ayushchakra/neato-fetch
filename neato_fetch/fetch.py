import sys
import termios
import tty

import select
import time

import rclpy
from rclpy.node import Node
import numpy as np
import cv2 as cv
from enum import Enum
from geometry_msgs.msg import Vector3, Twist
from sensor_msgs.msg import Image, LaserScan
from neato2_interfaces.msg import Bump
from cv_bridge import CvBridge

class NeatoState(Enum):
    """
    This class defines the different states that the Neato can be in while
    completing its fetch action.
    """
    DRIVE_NEATO_START = 0
    ANALYZE_REF_IMAGE = 1
    TRACK_BALL = 2
    TRACK_PERSON = 3
    CELEBRATION = 4
    FETCH_DONE = 5

class DrawBoxState(Enum):
    """
    This class defines the different states within the NeatoState.ANALYZE_REF_IMAGE
    state and is used to determine which corner of each bounding box is currently
    being inputted by the user. 
    """
    GET_PERSON_CORNER_ONE = 0
    GET_PERSON_CORNER_TWO = 1
    GET_BALL_CORNER_ONE = 2
    GET_BALL_CORNER_TWO = 3

class FetchNode(Node):
    """
    This is the main node that dictates the workflow of the neato's fetch behavior.
    The Node handles the different state transitions through the NeatoState
    finite state machine and defines the proper actions for each state.
    """

    # This dictionary is used by the teleop_to_start function to allow for
    # users to remotely drive the Neato to the intended start location.
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

    # This defines the threshold for a good match based on the distance between
    # the two nearest neighbors to the queried point.
    GOOD_MATCH_THRESHOLD = .9

    # This defines the proportional control applied to the angular velocity
    # of the Neato as it approaches the person and the ball.
    P_CONSTANT = 500

    # This defines the angular range in front of the Neato to analyze in order to
    # determine if a specific destination has been reached
    SCAN_BOUND = 30

    def __init__(self):
        """
        This is the constructor for the FetchNeato Node, which defines the node's
        publishers and subscribers, along with declaring class variables relevant
        to each of the neato's states.
        """
        super().__init__('fetch_node')

        # Instantiates the initial state of the neato
        self.neatoState = NeatoState.DRIVE_NEATO_START
        self.drawBoxState = DrawBoxState.GET_BALL_CORNER_ONE

        # Instantiates a timer to continuously call run_loop
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.run_loop)

        # Creates message publishers and subscribers to send a receive data
        # from the Neato
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.cam_sub = self.create_subscription(Image, "camera/image_raw", self.process_image, 10)
        self.bump_sub = self.create_subscription(Bump, "bump", self.process_bump, 10)
        self.scan_sub = self.create_subscription(LaserScan, "scan", self.process_scan, 10)
        self.bump = False
        self.scan = False

        # Defines variables to capture the user's keyboard input for teleop
        self.key = None
        self.settings = termios.tcgetattr(sys.stdin)

        # Creates instance variables for tracking, analyzing, and displaying image
        # features
        self.image = None
        self.reference_image = None
        self.ball_kps = []
        self.ball_descs = []
        self.person_kps = []
        self.person_descs = []
        self.bridge = CvBridge()
        self.initialize_cv_algorithms()
        cv.namedWindow("Reference Image")
        cv.setMouseCallback("Reference Image", self.process_mouse_click)
        self.ball_corner_one = None
        self.ball_corner_two = None
        self.person_corner_one = None
        self.person_corner_two = None

    def process_scan(self, msg: LaserScan):
        """
        This is the callback for when a new laser scan is available. The function
        processes scan values from -SCAN_BOUND to SCAN_BOUND and determines whether
        any are close enough to consider the Neato as reaching its goal.
        """
        # Filters out scans outside of the neato's laser scan range
        filtered_dist = [x for x in msg.ranges[-self.SCAN_BOUND:] + msg.ranges[:self.SCAN_BOUND] if x != 0.0]

        if len(filtered_dist) > 0 and np.min(filtered_dist) < .4:
            self.scan = True
        else:
            self.scan = False

    def process_bump(self, msg: Bump):
        """
        This is the callback for the bump sensor. If any of the bumps are triggered,
        self.bump is set to True.
        """
        if msg.left_front == 1 or msg.right_front == 1 or msg.left_side == 1 or msg.right_side == 1:
            self.bump = True
        else:
            self.bump = False

    def process_image(self, msg: Image):
        """
        This is the callback for the image topic, where each received image is
        converted to an OpenCV image that can be further processed.
        """
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def initialize_cv_algorithms(self):
        """
        This function initialized the ORB image descriptor algorithm and the FLANN
        descriptor matching algorithm, both of which are defined by OpenCV>
        """
        self.orb = cv.ORB_create()
        index_params = dict(
            algorithm=6,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        self.flann = cv.FlannBasedMatcher(index_params, {})

    def get_key(self):
        """
        Function to monitor the keyboard and extract any inputs from the user.
        """
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def get_kps_descs(self, image):
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kps, descs = self.orb.detectAndCompute(gray_img, None)
        return kps, descs

    def get_ref_img_kps_descs(self):
        kps, descs = self.get_kps_descs(self.reference_image)

        for kp, desc in zip(kps, descs):
            if kp.pt[0] > self.person_corner_one[0] and kp.pt[0] < self.person_corner_two[0] and\
                kp.pt[1] > self.person_corner_one[1] and kp.pt[1] < self.person_corner_two[1]:
                self.person_kps.append(kp)
                self.person_descs.append(desc)
            if kp.pt[0] > self.ball_corner_one[0] and kp.pt[0] < self.ball_corner_two[0] and\
                kp.pt[1] > self.ball_corner_one[1] and kp.pt[1] < self.ball_corner_two[1]:
                self.ball_kps.append(kp)
                self.ball_descs.append(desc)
        self.person_kps = np.array(self.person_kps)
        self.person_descs = np.array(self.person_descs)
        self.ball_kps = np.array(self.ball_kps)
        self.ball_descs = np.array(self.ball_descs)
        self.neatoState = NeatoState.TRACK_BALL
        
    def teleop_to_start(self):
        try:
            while True:
                self.key = self.get_key()
                if self.key == "\x03":
                    self.vel_pub.publish(self.key_to_vel["s"])
                    raise KeyboardInterrupt
                if self.key in self.key_to_vel.keys():
                    self.vel_pub.publish(self.key_to_vel[self.key])
        except KeyboardInterrupt:
            self.neatoState = NeatoState.ANALYZE_REF_IMAGE
            return

    def drive_to_object(self, curr_center_x):
        self.vel_pub.publish(Twist(linear=Vector3(x=0.2), angular=Vector3(z=(512-curr_center_x)/self.P_CONSTANT)))

    def celebration(self):          
        self.vel_pub.publish(self.key_to_vel["x"])
        time.sleep(1)
        self.vel_pub.publish(self.key_to_vel["a"])
        time.sleep(0.5)
        self.vel_pub.publish(self.key_to_vel["d"])
        time.sleep(0.5)
        self.vel_pub.publish(self.key_to_vel["w"])
        time.sleep(1)
        if self.bump:
            self.vel_pub.publish(self.key_to_vel["s"])
            self.neatoState = NeatoState.FETCH_DONE

    def process_mouse_click(self, event, x, y, flag, im):
        if self.neatoState == NeatoState.ANALYZE_REF_IMAGE and event == cv.EVENT_FLAG_LBUTTON:
            if self.drawBoxState == DrawBoxState.GET_BALL_CORNER_ONE:
                self.ball_corner_one = (x,y)
                self.drawBoxState = DrawBoxState.GET_BALL_CORNER_TWO
            elif self.drawBoxState == DrawBoxState.GET_BALL_CORNER_TWO:
                self.ball_corner_two = (x,y)
                self.drawBoxState = DrawBoxState.GET_PERSON_CORNER_ONE
            elif self.drawBoxState == DrawBoxState.GET_PERSON_CORNER_ONE:
                self.person_corner_one = (x,y)
                self.drawBoxState = DrawBoxState.GET_PERSON_CORNER_TWO
            elif self.drawBoxState == DrawBoxState.GET_PERSON_CORNER_TWO:
                self.person_corner_two = (x,y)

    def draw_bounding_boxes(self):
        if self.reference_image is None:
            self.reference_image = self.image
        else:
            cv.imshow("Reference Image", self.reference_image)
            cv.waitKey(5)
        
    def get_matches(self, ref_kps, ref_descs):
        ref_kp_matches = []
        curr_kp_matches = []

        try:
            curr_kps, curr_descs = self.get_kps_descs(self.image)
            matches = self.flann.knnMatch(ref_descs, curr_descs, 2)
            matchesMask = [[0,0] for i in range(len(matches))]
            for i, (dmatch_one, dmatch_two) in enumerate(matches):
                if dmatch_one.distance < self.GOOD_MATCH_THRESHOLD * dmatch_two.distance:
                    ref_kp_matches.append(ref_kps[dmatch_one.queryIdx])
                    curr_kp_matches.append(curr_kps[dmatch_one.trainIdx])
                    matchesMask[i] = [1,0]
            draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
            corrected_img = cv.drawMatchesKnn(self.reference_image,ref_kps,self.image,curr_kps,matches,None,**draw_params)
            cv.imshow("Detected Matches", corrected_img)
            cv.waitKey(5)
        except:
            pass
        return ref_kp_matches, curr_kp_matches

    def drive_to_ball(self):
        matched_ref_kps, matched_curr_kps = self.get_matches(self.ball_kps, self.ball_descs)
        if len(matched_curr_kps) == 0:
            return
        avg_ref_kp_x = sum([kp.pt[0] for kp in matched_ref_kps])/len(matched_ref_kps)
        avg_curr_kp_x = sum([kp.pt[0] for kp in matched_curr_kps])/len(matched_curr_kps)
        self.drive_to_object(avg_ref_kp_x, avg_curr_kp_x)
        if self.scan:
            self.vel_pub.publish(self.key_to_vel["d"])
            time.sleep(3.14)
            self.neatoState = NeatoState.TRACK_PERSON

    def drive_to_person(self):
        matched_ref_kps, matched_curr_kps = self.get_matches(self.person_kps, self.person_descs)
        if len(matched_curr_kps) == 0:
            return
        avg_ref_kp_x = sum([kp.pt[0] for kp in matched_ref_kps])/len(matched_ref_kps)
        avg_curr_kp_x = sum([kp.pt[0] for kp in matched_curr_kps])/len(matched_curr_kps)
        self.drive_to_object(avg_ref_kp_x, avg_curr_kp_x)
        if self.bump:
            self.neatoState = NeatoState.CELEBRATION

    def run_loop(self):
        print(self.neatoState)
        if self.neatoState == NeatoState.DRIVE_NEATO_START:
            self.teleop_to_start()
        elif self.neatoState == NeatoState.ANALYZE_REF_IMAGE:
            self.draw_bounding_boxes()
            if self.person_corner_two is not None:
                self.get_ref_img_kps_descs()
        elif self.neatoState == NeatoState.TRACK_BALL:
            self.drive_to_ball()
        elif self.neatoState == NeatoState.TRACK_PERSON:
            self.drive_to_person()
        elif self.neatoState == NeatoState.CELEBRATION:
            self.celebration()

def main(args=None):
    rclpy.init(args=args)
    node = FetchNode()
    rclpy.spin(node)
    rclpy.shutdown()