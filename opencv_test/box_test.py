import pdb
import cv2 as cv
from enum import Enum

import numpy as np

class curr_state(Enum):
    wait_for_frame = 0
    get_frame = 1
    get_first_corner = 2
    get_second_corner = 3
    calculate_kps = 4
    show_matches = 5

class analyzeFrame():
    def __init__(self):
        self.state = curr_state.wait_for_frame
        self.record_reference = False
        self.reference_image = None
        self.first_corner = None
        self.second_corner = None


def set_box(event, x, y, flag, im):
    if event == cv.EVENT_FLAG_LBUTTON:
        if tracker.state == curr_state.wait_for_frame:
            tracker.state = curr_state.get_frame
        elif tracker.state == curr_state.get_first_corner:
            tracker.first_corner = (x,y)
            tracker.state = curr_state.get_second_corner
        elif tracker.state == curr_state.get_second_corner:
            tracker.second_corner = (x,y)
            tracker.state = curr_state.calculate_kps

def get_kps_descs(image):
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        reference_kps, reference_descs = orb.detectAndCompute(gray_img, None)
        return reference_kps, reference_descs

if __name__ == "__main__":
    tracker = analyzeFrame()

    cv.namedWindow("test")
    cv.setMouseCallback("test", set_box)

    cv.namedWindow("Test")
    cv.setMouseCallback("Test", set_box)

    cap = cv.VideoCapture(0)
    while True:
        if tracker.state == curr_state.wait_for_frame:
            ret, frame = cap.read()
            cv.imshow("test", frame)
        elif tracker.state == curr_state.get_frame:
            ret, frame = cap.read()
            tracker.reference_image = frame
            tracker.state = curr_state.get_first_corner
            cv.imshow("test", tracker.reference_image)
        elif tracker.state == curr_state.get_first_corner :
            cv.imshow("test", tracker.reference_image)
        elif tracker.state == curr_state.get_second_corner:
            cv.imshow("test", tracker.reference_image)
        elif tracker.state == curr_state.calculate_kps:
            cv.imshow("test", tracker.reference_image)
            kps, descs = get_kps_descs(tracker.reference_image)
            relevant_kps = []
            relevant_descs = []
            for kp, desc in zip(kps, descs):
                if kp.pt[0] > tracker.first_corner[0] and kp.pt[0] < tracker.second_corner[0] and\
                    kp.pt[1] > tracker.first_corner[1] and kp.pt[1] < tracker.second_corner[1]:
                    relevant_kps.append(kp)
                    relevant_descs.append(desc)
                    print("here")
            relevant_kps = np.array(relevant_kps)
            relevant_descs = np.array(relevant_descs)
            tracker.state = curr_state.show_matches
        elif tracker.state == curr_state.show_matches:
            ret, frame = cap.read()
            curr_kps, curr_descs = get_kps_descs(frame)
            index_params = dict(
                algorithm=6,
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
            flann = cv.FlannBasedMatcher(index_params, {})
            matches = flann.knnMatch(relevant_descs, curr_descs, 2)
            
            num_matches = 0

            matchesMask = [[0,0] for i in range(len(matches))]
            if len(np.shape(matches)) != 2:
                continue
            for i, (m, n) in enumerate(matches):
                if m.distance < .8 * n.distance:
                    num_matches += 1
                    matchesMask[i]=[1,0]

            draw_params = dict(matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = cv.DrawMatchesFlags_DEFAULT)  
            corrected_img = cv.drawMatchesKnn(tracker.reference_image, relevant_kps, frame, curr_kps, matches, None, **draw_params)
            cv.imshow('test', corrected_img)

        cv.waitKey(5)

