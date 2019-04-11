import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
import time
import math
import os

SRC = np.float32(
    [[127 * 2, 315 * 2],
     [247 * 2, 449 * 2],
     [600 * 2, 378 * 2],
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
    # cap = cv2.VideoCapture('2/2.mp4')

    # count = 0
    # total = 269
    while True:
        ret, frame = cap.read()
        start = time.time()
        height, width, depth = frame.shape

        DST = np.float32(
            [[(width - 716) / 2, 0],
             [(width - 716) / 2, height],
             [width - 282, height],
             [width - 282, 0]]
        )

        if not ret:
            break

        undistorted_image = cv2.undistort(frame, CAM_MATRIX, DIST_COEF)

        viewed_x = bird_view(undistorted_image)
        viewed = cv2.resize(viewed_x, dsize=(int(width / 2), int(height / 2)), interpolation=cv2.INTER_LINEAR)

        viewed = cv2.GaussianBlur(viewed, ksize=(5, 5), sigmaX=0)

        # img = cv2.resize(viewed, dsize=(512, 640), interpolation=cv2.INTER_LINEAR)

        gray_y = cv2.cvtColor(viewed, cv2.COLOR_BGR2HSV)
        gray_y = gray_y[:, :, 2]
        gray_x = gray_y.copy()

        line_y = np.zeros((11, 11), dtype=np.uint8)
        line_y[5, ...] = 1
        line_x = np.transpose(line_y)

        y = cv2.morphologyEx(gray_y, cv2.MORPH_OPEN, line_y, iterations=3)
        x = cv2.morphologyEx(gray_x, cv2.MORPH_OPEN, line_x, iterations=3)

        gray_y -= y
        gray_x -= x

        kernel_x = np.ones((5, 20), dtype=np.uint8)
        kernel_y = np.ones((10, 1), dtype=np.uint8)

        dilated_view_x = cv2.dilate(gray_x, kernel_x, iterations=2)
        kernel_x = np.ones((5, 40), dtype=np.uint8)
        eroded_view_x = cv2.erode(dilated_view_x, kernel_x, iterations=2)

        eroded_view_y = cv2.erode(gray_y, kernel_y, iterations=2)

        threshold_x = eroded_view_x / 255
        threshold_y = eroded_view_y / 255

        threshold_y[threshold_y >= 0.2] = 1
        threshold_y[threshold_y < 0.2] = 0

        threshold_x[threshold_x >= 0.2] = 1
        threshold_x[threshold_x < 0.2] = 0

        arr_x = threshold_x > 0
        final_x = remove_small_objects(arr_x, min_size=512, connectivity=1)
        final_x = np.array(final_x * 255, dtype=np.uint8)

        arr_y = threshold_y > 0
        final_y = remove_small_objects(arr_y, min_size=512, connectivity=1)
        final_y = np.array(final_y * 255, dtype=np.uint8)

        cv2.imshow('final_x', final_x)
        cv2.imshow('final_y', final_y)
        cv2.imshow('viewed', viewed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        arr = threshold_x > 0
        final = remove_small_objects(arr, min_size=512, connectivity=1)
        final = np.array(final * 255, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(final, 1, 2)
        contours_sorted_by_area = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        contours_vertical = []
        for j, contour_sort in enumerate(contours_sorted_by_area):
            if 0 in contour_sort or cv2.contourArea(contour_sort) < 250:
                continue
            x, y, w, h = cv2.boundingRect(contour_sort)
            contours_vertical.append([x, y, w, h])

        contours_sorted_by_x = sorted(contours_vertical, key=lambda x: x[1])

        if len(contours_sorted_by_x) > 2:
            x1, y1, w1, h1 = contours_sorted_by_x[-1]
            x2, y2, w2, h2 = contours_sorted_by_x[-2]

            angle = abs(int(math.atan((y2 - y1) / (x2 - x1 + 0.001)) * 180 / math.pi))
            if 80 < angle < 100:
                cv2.line(viewed, (x1, y1), (x2, y2 + h2), (0, 0, 255), 2)
                cv2.circle(viewed, (x1, y1), 5, (0, 0, 255), 3)
                cv2.circle(viewed, (x2, y2 + h2), 5, (0, 0, 255), 3)

            cv2.circle(viewed, (x1, y1), 5, (0, 0, 255), 3)
            cv2.circle(viewed, (x2, y2 + h2), 5, (0, 0, 255), 3)

        end = time.time()
        ms = end - start

        cv2.putText(final, f'time spent {ms}', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

        cv2.circle(viewed, (162, 470), 5, (0, 0, 255), 3)
        cv2.circle(frame, (508, 838), 5, (0, 0, 255), 3)

        # cv2.imshow('frame', frame)
        # cv2.imshow('threshold_x', threshold_x)
        # cv2.imshow('threshold_y', threshold_y)
        cv2.imshow('viewed', viewed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
