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

    # This is the center x pixel position of the camera frame.
    IMAGE_CENTER = 512

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
        """
        This function returns the keypoints and image descriptors (obtained
        using ORB) of the inputted image.
        """
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        kps, descs = self.orb.detectAndCompute(gray_img, None)
        return kps, descs

    def get_ref_img_kps_descs(self):
        """
        This function obtains and filters the keypoints and image descriptors
        of the stored reference image (which contains both the picture of the
        ball and the person).
        """
        # Obtains all keypoints and image descriptors from the reference image.
        kps, descs = self.get_kps_descs(self.reference_image)

        for kp, desc in zip(kps, descs):
            # If the keypoint is within the bounding box of the person, append
            # the keypoint and descriptor to the stored list of person keypoints
            # and image descriptors.
            if kp.pt[0] > self.person_corner_one[0] and kp.pt[0] < self.person_corner_two[0] and\
                kp.pt[1] > self.person_corner_one[1] and kp.pt[1] < self.person_corner_two[1]:
                self.person_kps.append(kp)
                self.person_descs.append(desc)
            # If the keypoint is within the bounding box of the ball, append
            # the keypoint and descriptor to the stored list of ball keypoints
            # and image descriptors.
            if kp.pt[0] > self.ball_corner_one[0] and kp.pt[0] < self.ball_corner_two[0] and\
                kp.pt[1] > self.ball_corner_one[1] and kp.pt[1] < self.ball_corner_two[1]:
                self.ball_kps.append(kp)
                self.ball_descs.append(desc)
        # Convert the arrays into numpy arrays
        self.person_kps = np.array(self.person_kps)
        self.person_descs = np.array(self.person_descs)
        self.ball_kps = np.array(self.ball_kps)
        self.ball_descs = np.array(self.ball_descs)
        
        # Change the Neato's state to tracking the ball.
        self.neatoState = NeatoState.TRACK_BALL
        
    def teleop_to_start(self):
        """
        This function listens to the user's keyboard inputs and publishes
        velocity commands corresponding to those keys until a keyboard
        interrupt is triggered.
        """
        try:
            while True:
                self.key = self.get_key()
                # This corresponds to Ctrl+C on most keyboards, which will raise
                # a keyboard interrupt.
                if self.key == "\x03":
                    self.vel_pub.publish(self.key_to_vel["s"])
                    raise KeyboardInterrupt
                if self.key in self.key_to_vel.keys():
                    self.vel_pub.publish(self.key_to_vel[self.key])
        except KeyboardInterrupt:
            # Change the neato's state to analyzing the reference image once
            # a keyboard interrupt has been initiated.
            self.neatoState = NeatoState.ANALYZE_REF_IMAGE
            return

    def drive_to_object(self, curr_center_x):
        """
        This function sends angular and linear velocities to the Neato based
        on the difference of the center pixel of the screen and the average
        center of all the matched keypoints.
        """
        self.vel_pub.publish(Twist(linear=Vector3(x=0.2), angular=Vector3(z=(self.IMAGE_CENTER-curr_center_x)/self.P_CONSTANT)))

    def celebration(self):
        """
        This function defines the celebration feature of the Neato, which
        continuously does a dance until the bump sensor is pressed by the
        person it navigated to.
        """   
        self.vel_pub.publish(self.key_to_vel["x"])
        time.sleep(1)
        self.vel_pub.publish(self.key_to_vel["a"])
        time.sleep(0.5)
        self.vel_pub.publish(self.key_to_vel["d"])
        time.sleep(0.5)
        self.vel_pub.publish(self.key_to_vel["w"])
        time.sleep(1)
        if self.bump:
            # Once the bump sensor has been pressed, stop the Neato and change
            # its state to FETCH_DONE.s
            self.vel_pub.publish(self.key_to_vel["s"])
            self.neatoState = NeatoState.FETCH_DONE

    def process_mouse_click(self, event, x, y):
        """
        This function handles the process for obtaining the bounding box around
        the person and the ball by storing mouse click positions.
        """
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
        """
        This function stores the reference image and displays it to the user,
        allowing them to select the bounding box by clicking on it.
        """
        if self.reference_image is None:
            self.reference_image = self.image
        else:
            cv.imshow("Reference Image", self.reference_image)
            cv.waitKey(5)
        
    def get_matches(self, ref_kps, ref_descs):
        """
        This function calculates matches between the keypoints of the current
        image and a specific area within the reference image (either the ball
        or the person).
        """
        ref_kp_matches = []
        curr_kp_matches = []

        try:
            # Get the keypoints and image descriptors of the current frame
            curr_kps, curr_descs = self.get_kps_descs(self.image)

            # Get matches between the current image and inputted reference
            # using FLANN, with 2 nearest neighbors.
            matches = self.flann.knnMatch(ref_descs, curr_descs, 2)
            for (dmatch_one, dmatch_two) in matches:
                # If the two matches are within a certain distance of each other
                # (not conflicting heavily), then append the closest match to
                # the list of matches
                if dmatch_one.distance < self.GOOD_MATCH_THRESHOLD * dmatch_two.distance:
                    ref_kp_matches.append(ref_kps[dmatch_one.queryIdx])
                    curr_kp_matches.append(curr_kps[dmatch_one.trainIdx])
        except:
            pass
        return ref_kp_matches, curr_kp_matches

    def drive_to_ball(self):
        """
        This function commands the Neato to drive towards the ball based
        on the matched keypoints between the current frame and the ones
        stored around the ball in the reference image.
        """
        # Obtain matches between the ball and the current frame
        _, matched_curr_kps = self.get_matches(self.ball_kps, self.ball_descs)
        
        # If no matches are found, exit this function.
        if len(matched_curr_kps) == 0:
            return

        # Calculate the average x position of the matched keypoints
        avg_curr_kp_x = sum([kp.pt[0] for kp in matched_curr_kps])/len(matched_curr_kps)

        # Drive to the object based on the current average keypoint position
        self.drive_to_object(avg_curr_kp_x)

        # If there is an object directly in front of the Neato (assumed to be
        # the ball), then turn 180 degrees and begin tracking the person.
        if self.scan:
            self.vel_pub.publish(self.key_to_vel["d"])
            time.sleep(np.pi)
            self.neatoState = NeatoState.TRACK_PERSON

    def drive_to_person(self):
        """
        This function commands the Neato to drive towards the person based on
        the matches between the stored references of the person and the current
        image.
        """
        # Obtain matches between the referenced person and the current frame
        _, matched_curr_kps = self.get_matches(self.person_kps, self.person_descs)

        # If no matches are found, exit this function
        if len(matched_curr_kps) == 0:
            return

        # Calculate the average x position of matches and drive towards that
        # average position
        avg_curr_kp_x = sum([kp.pt[0] for kp in matched_curr_kps])/len(matched_curr_kps)
        self.drive_to_object(avg_curr_kp_x)

        # If the bump sensor is triggered (assumed to be bumping the person),
        # then change states to CELEBRATION
        if self.bump:
            self.neatoState = NeatoState.CELEBRATION

    def run_loop(self):
        """
        This is the main loop of the FetchNode. Based on the current
        state of the Neato, the corresponding function is called, commanding
        the current action of the Neato.
        """
        # This is the teleop start of the Neato, which allows the user to palce
        # the Neato in the desired start position, facing the ball and the person.
        if self.neatoState == NeatoState.DRIVE_NEATO_START:
            self.teleop_to_start()
        # Once in the desired start position, the current frame is analyzed for
        # a user-inputted bounding box around the ball and the person, which
        # are both analyzed for keypoints and image descriptors.
        elif self.neatoState == NeatoState.ANALYZE_REF_IMAGE:
            self.draw_bounding_boxes()
            if self.person_corner_two is not None:
                self.get_ref_img_kps_descs()
        # The Neato is then directed to drive towards the ball by obtianing
        # matches between the current frame and the ball keypoints.
        elif self.neatoState == NeatoState.TRACK_BALL:
            self.drive_to_ball()
        # After the Neato kicks the ball, it begins return to the person, which
        # is done by obtaining matches between the current frame and the person
        # keypoints.
        elif self.neatoState == NeatoState.TRACK_PERSON:
            self.drive_to_person()
        # Once the Neato successfully reaches the person, it begins celebrating.
        elif self.neatoState == NeatoState.CELEBRATION:
            self.celebration()

def main(args=None):
    """
    This is the main function of this script, which establishes rclpy, instantiates
    the FetchNode, triggered the entire fetch workflow on the Neato.
    """
    rclpy.init(args=args)
    node = FetchNode()
    rclpy.spin(node)
    rclpy.shutdown()