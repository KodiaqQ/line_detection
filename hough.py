import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
import time
import math

# fig, axes = plt.subplots(6, 4)
#
# image = cv2.imread('lane2.jpg', cv2.IMREAD_GRAYSCALE)
# feature extraction
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
#
# axes[0, 0].imshow(image)
# axes[0, 1].imshow(image[:, :, 0])
# axes[0, 2].imshow(image[:, :, 1])
# axes[0, 3].imshow(image[:, :, 2])
#
# axes[1, 0].imshow(hsv)
# axes[1, 1].imshow(hsv[:, :, 0])
# axes[1, 2].imshow(hsv[:, :, 1])
# axes[1, 3].imshow(hsv[:, :, 2])
#
# axes[2, 0].imshow(hls)
# axes[2, 1].imshow(hls[:, :, 0])
# axes[2, 2].imshow(hls[:, :, 1])
# axes[2, 3].imshow(hls[:, :, 2])
#
# image = cv2.medianBlur(image, ksize=5)
# hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#
# axes[3, 0].imshow(image)
# axes[3, 1].imshow(image[:, :, 0])
# axes[3, 2].imshow(image[:, :, 1])
# axes[3, 3].imshow(image[:, :, 2])
#
# axes[4, 0].imshow(hsv)
# axes[4, 1].imshow(hsv[:, :, 0])
# axes[4, 2].imshow(hsv[:, :, 1])
# axes[4, 3].imshow(hsv[:, :, 2])
#
# axes[5, 0].imshow(hls)
# axes[5, 1].imshow(hls[:, :, 0])
# axes[5, 2].imshow(hls[:, :, 1])
# axes[5, 3].imshow(hls[:, :, 2])
#
# plt.show()

cap = cv2.VideoCapture('auto.mp4')

while True:

    ret, frame = cap.read()
    if not ret:
        break
    height, width, depth = frame.shape
    region_of_interest_vertices = [
        (0, height),
        (0, height / 2),
        (width / 3, height / 2),
        (width, height),
    ]
    region_of_interest_vertices = np.array([region_of_interest_vertices], dtype=np.int)
    mask = np.zeros_like(frame)

    channels = frame.shape[2]

    match_mask = (255,) * channels
    cv2.fillPoly(mask, region_of_interest_vertices, match_mask)

    masked = cv2.bitwise_and(frame, mask)

    # filtering
    image = cv2.GaussianBlur(masked, (5, 5), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image, (5, 5, 150), (255, 255, 255))
    kernel = np.ones((20, 20), np.uint8)  # vertical
    d_im = cv2.dilate(image, kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1)
    # kernel = np.ones((20, 1), np.uint8)
    # e_im = cv2.erode(e_im, kernel, iterations=1)
    ret, thresh = cv2.threshold(e_im, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for contour in cntsSorted[:4]:
        if cv2.contourArea(contour) < 150:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = int(math.atan((y - (y + w)) / ((x + h) - x)) * 180 / math.pi)
        if abs(angle) > 70 and abs(angle) < 90:
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

        # cv2.line(frame, (x, y - h), (x + w, y), (0, 0, 255), 2)

        # cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

    # edges = cv2.Canny(e_im, 50, 150, apertureSize=3)
    #
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
    #                         lines=np.array([]),
    #                         minLineLength=40,
    #                         maxLineGap=25)
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # if lines is not None:
    #     for line in lines:
    # for rho, theta in line:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    # for x1, y1, x2, y2 in line:
    #     cv2.circle(frame, (x1, y1), 10, (0, 0, 255), 3)
    #     cv2.circle(frame, (x2, y2), 10, (0, 255, 0), 3)
    # cv2.line(original, (x1, y1), (x2, y2), (0, 0, 255), 2)

    frame = cv2.resize(frame, dsize=(round(frame.shape[1] / 2), round(frame.shape[0] / 2)))
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
