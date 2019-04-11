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
        threshold_x = remove_small_objects(arr_x, min_size=512, connectivity=1)
        threshold_x = np.array(threshold_x * 255, dtype=np.uint8)

        arr_y = threshold_y > 0
        threshold_y = remove_small_objects(arr_y, min_size=512, connectivity=1)
        threshold_y = np.array(threshold_y * 255, dtype=np.uint8)

        contours, hierarchy = cv2.findContours(threshold_x, 1, 2)
        contours_sorted_by_area_x = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        contours_coords_x = []

        for j, contour_sort in enumerate(contours_sorted_by_area_x):
            if 0 in contour_sort or cv2.contourArea(contour_sort) < 250:
                continue
            x, y, w, h = cv2.boundingRect(contour_sort)
            contours_coords_x.append([x, y, w, h])

        contours_x_sorted_by_x = sorted(contours_coords_x, key=lambda x: x[1])

        contours, hierarchy = cv2.findContours(threshold_y, 1, 2)
        contours_sorted_by_area_y = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        contours_coords_y = []

        for j, contour_sort in enumerate(contours_sorted_by_area_y):
            if 0 in contour_sort or cv2.contourArea(contour_sort) < 250:
                continue
            x, y, w, h = cv2.boundingRect(contour_sort)
            contours_coords_y.append([x, y, w, h])

        contours_y_sorted_by_y = sorted(contours_coords_y, key=lambda x: x[1])

        if len(contours_x_sorted_by_x) > 1 and len(contours_sorted_by_area_y) > 0:
            x1, y1, w1, h1 = contours_x_sorted_by_x[-1]
            x2, y2, w2, h2 = contours_x_sorted_by_x[-2]

            angle_in_deg_1 = math.atan(w1 / h1) * 180 / math.pi
            angle_in_deg_2 = math.atan(w2 / h2) * 180 / math.pi

            angle_delta = abs(angle_in_deg_2 - angle_in_deg_1)

            if 0 < angle_delta < 5:
                middle_point_x = int((x2 + x1) / 2)
                middle_point_y = int((y2 + h2 + y1) / 2)

                for contour_y in contours_y_sorted_by_y:
                    x_y_1 = contour_y[0]
                    y_y_1 = contour_y[1]
                    w_y_1 = contour_y[2]
                    h_y_1 = contour_y[3]

                    if y_y_1 < middle_point_y < y_y_1 + h_y_1:
                        angle_in_deg_y = math.atan(1 / contour_y[2]) * 180 / math.pi
                        angle_in_deg_x = math.atan(w2 / h2) * 180 / math.pi

                        angle_delta = round(angle_in_deg_x - angle_in_deg_y)

                        if 80 < angle_delta < 90:
                            cv2.line(viewed, (middle_point_x, middle_point_y),
                                     (int(x_y_1 + w_y_1 / 2), int(y_y_1 + h_y_1 / 2)),
                                     (0, 0, 255), 2)
                            cv2.circle(viewed, (middle_point_x, middle_point_y), 5, (0, 0, 255), 3)
                            cv2.line(viewed, (x1, y1), (x2, y2 + h2), (0, 0, 255), 2)
                            cv2.circle(viewed, (x1, y1), 5, (0, 0, 255), 3)
                            cv2.circle(viewed, (x2, y2 + h2), 5, (0, 0, 255), 3)

        end = time.time()
        ms = round(end - start, 4)

        cv2.putText(viewed, f'time {ms} ms', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))

        cv2.circle(viewed, (162, 470), 5, (0, 0, 255), 3)
        cv2.circle(frame, (508, 838), 5, (0, 0, 255), 3)

        viewed = cv2.resize(viewed, dsize=(width, height), interpolation=cv2.INTER_AREA)
        img_size = (viewed.shape[1], viewed.shape[0])
        M = cv2.getPerspectiveTransform(DST, SRC)
        result = cv2.warpPerspective(viewed, M, img_size, flags=cv2.INTER_LINEAR)

        # result = cv2.bitwise_or(result, frame)

        # cv2.imshow('viewed', viewed)
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
