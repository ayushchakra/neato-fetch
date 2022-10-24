import cv2
import numpy as np

cam = cv2.VideoCapture(0)
result, base_image = cam.read()
gray_img = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

base_kps, base_descs = orb.detectAndCompute(base_image, None)

index_params = dict(algorithm=6,
    table_number=6,
    key_size=12,
    multi_probe_level=1)

flann = cv2.FlannBasedMatcher(index_params, {})

# cv2.imshow('kp', kp_img)
# cv2.waitKey()

while True:
    try:
        _, image = cam.read()
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        curr_kps, curr_descs = orb.detectAndCompute(image, None)
        matches = flann.knnMatch(base_descs, curr_descs, 2)

        matchesMask = [[0,0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.675 * n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
        corrected_img = cv2.drawMatchesKnn(base_image,base_kps,image,curr_kps,matches,None,**draw_params)
    except:
        continue

    cv2.imshow('test', corrected_img)
    if cv2.waitKey(1) & 0xFF ==ord('e'):
        break