import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
import time
import math
import os

SRC = np.float32(
    [[127 * 2, 315 * 2],
     [239 * 2, 441 * 2],
     [585 * 2, 375 * 2],
     [328 * 2, 305 * 2]])

CAM_MATRIX = np.matrix([[923.6709132611408, 0, 660.3716073305085], [0, 925.6373437421516, 495.2039455113797],
                        [0, 0, 1]])
DIST_COEF = np.array([-0.2947018330961229, 0.09105224521150024, 0.0001143430530863253, 0.0003862123859247846, 0])


def bird_view(image):
    img_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(SRC, DST)
    return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)


if __name__ == '__main__':
    cap = cv2.VideoCapture('output.avi')
    while True:
        ret, frame = cap.read()
        height, width, depth = frame.shape
        kernel = np.ones((20, 100), np.uint8)

        DST = np.float32(
            [[(width - 716) / 2, 0],
             [(width - 716) / 2, height],
             [width - 282, height],
             [width - 282, 0]]
        )

        if not ret:
            break

        undistorted_image = cv2.undistort(frame, CAM_MATRIX, DIST_COEF)

        viewed = bird_view(undistorted_image)

        region_of_interest_vertices = [
            (128, height - 128),
            (128, 128),
            (width - 288, 128),
            (width - 288, height - 128),
        ]
        region_of_interest_vertices = np.array([region_of_interest_vertices], dtype=np.int)
        mask = np.zeros_like(viewed)

        channels = viewed.shape[2]

        match_mask = (255,) * channels
        cv2.fillPoly(mask, region_of_interest_vertices, match_mask)

        masked = cv2.bitwise_and(viewed, mask)

        blur = cv2.GaussianBlur(masked, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        ranged = cv2.inRange(hsv, (5, 5, 150), (255, 255, 255))

        dilated_view = cv2.dilate(ranged, kernel, iterations=1)
        eroded_view = cv2.erode(dilated_view, kernel, iterations=1)

        ret, thresh = cv2.threshold(eroded_view, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        contours_sorted_by_area = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        contours_coords = []

        for j, contour_sort in enumerate(contours_sorted_by_area):
            if 0 in contour_sort or cv2.contourArea(contour_sort) < 100:
                continue
            x, y, w, h = cv2.boundingRect(contour_sort)
            contours_coords.append([x, y, w, h])

        contours_sorted_by_x = sorted(contours_coords, key=lambda x: x[1])

        if len(contours_sorted_by_x) < 2:
            continue

        x1, y1, w1, h1 = contours_sorted_by_x[-1]
        x2, y2, w2, h2 = contours_sorted_by_x[-2]

        angle = int(math.atan((y2 - y1 + 1) / (x2 - x1 + 1)) * 180 / math.pi)

        if abs(angle) > 80 and abs(angle) < 100 and (y1 - y2) > 150:
            cv2.line(viewed, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(viewed, (x1, y1), 5, (0, 0, 255), 3)
            cv2.circle(viewed, (x2, y2), 5, (0, 0, 255), 3)

        viewed = cv2.resize(viewed, dsize=(int(height / 2), int(width / 2)))

        cv2.imshow('viewed', viewed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
