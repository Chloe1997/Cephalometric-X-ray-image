import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from xgboost import cv

image_path = "Pre_X-ray.bmp"

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.show()


# find the circle landmarker
def landmark_detection(image_path):
    im_gray = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    thresh = 220
    im_bi = np.where(im_gray < thresh, 0, im_gray)
    im_aq = cv2.equalizeHist(im_bi)
    kernel = np.ones((1, 1), np.uint8)
    erosion = cv2.erode(im_aq, kernel, iterations=1)
    dilate = cv2.dilate(erosion, kernel, iterations=2)
    image = dilate
    # Method 1
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows,
                                   param1=10, param2=30,
                                   minRadius=0, maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            size = i[2]
            return center, size

# find center position
center, size = landmark_detection(image_path)
print(size,"size")


def edge_detection_pre(file_path):
    im_gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    raw_image = im_gray.copy()
    ret, th1 = cv2.threshold(im_gray, 70, 255, cv2.THRESH_BINARY)
    im_gray[th1 == 0] = 0

    # find jaw contour
    thresh = 50
    im_bi = np.where(im_gray < thresh, 0, im_gray)
    im_aq = cv2.equalizeHist(im_bi)
    jaw = im_aq

    # find teeth location
    thresh = 220
    im_bi = np.where(im_gray < thresh, 0, im_gray)
    im_tooth = cv2.equalizeHist(im_bi)
    kernel = np.ones((5, 5), np.uint8)
    im_tooth = cv2.erode(im_tooth, kernel, iterations=1)
    im_tooth = cv2.dilate(im_tooth, kernel, iterations=6)

    return raw_image, jaw, im_tooth


(x, y) = center
raw_img, image, tooth = edge_detection_pre(image_path)
Verti2 = np.concatenate((raw_img, image), axis=1)
show_image(Verti2)
show_image(tooth)

# find jaw contour
copy_image = np.zeros_like(image)
copy_image[y+size:, x+size:] = image[y+size:, x+size:]
copy_image = cv2.equalizeHist(copy_image)
cnts, _ = cv2.findContours(copy_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filter_cnt = []
for cnt in cnts:
    if cv2.contourArea(cnt) > 80 and len(filter_cnt) == 0:
        filter_cnt.append(cnt)
    elif cv2.contourArea(cnt) < 80:
        pass
    elif cv2.contourArea(cnt) > cv2.contourArea(filter_cnt[0]):
        filter_cnt.pop(0)
        filter_cnt.append(cnt)


img = copy_image.copy()
mask = np.zeros(img.shape, dtype='uint8')  #依Contours圖形建立mask
cv2.drawContours(mask, filter_cnt, -1, 255, -1)  # 255        →白色, -1→塗滿
# show_image(mask)
roi_mask = cv2.bitwise_and(img, img, mask=mask)
print('roi_mask')
# show_image(roi_mask)

# print jaw boundary
jaw_img = raw_img.copy()
jaw_img = cv2.cvtColor(jaw_img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(jaw_img, filter_cnt, -1, (0, 255, 0), 2)
# print('jaw_img')
# show_image(jaw_img)


# find constraint jaw line
tooth = np.where(roi_mask < 230, 0, roi_mask)
print('highlight')
show_image(tooth)
kernel = np.ones((5, 5), np.uint8)
tooth = cv2.dilate(tooth, kernel, iterations=4)
tooth = np.where(tooth<240,0,tooth)
show_image(tooth)
rows, cols = tooth.shape[:2]
cnt_t, _ = cv2.findContours(tooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
[vx, vy, x, y] = cv2.fitLine(cnt_t[0],cv2.DIST_L2,0,0.01,0.01)
def line(x_position):
    if (vy/vx)*x-y < 0:
        return True
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)

# below tooth mask 
points = np.array( [[cols-1, righty], [0, lefty], [0,rows-1], [cols-1,rows-1]])
mask = np.zeros(img.shape, dtype='uint8')
# cv2.fillPoly(mask, [points], 255)
cv2.drawContours(mask, [points],-1,255,-1)
show_image(mask)
output = cv2.bitwise_and(jaw_img, jaw_img, mask=mask)
show_image(jaw_img)
print('output', np.shape(output))
show_image(output)


raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
cv2.circle(raw_img, center, size, (255, 0, 0), 2)
cv2.putText(raw_img, str(center), (center[0] + 20, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

output = cv2.bitwise_or(output,raw_img)

show_image(output)


