import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "Pre_X-ray.bmp"


def plot_histo(gray):
    # print(np.unique(gray))
    # # 計算直方圖每個 bin 的數值
    # # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    # #
    # # print(len(hist.ravel()))
    # # 畫出直方圖
    # plt.hist(gray.ravel(), 256, [0, 256])
    # plt.show()
    # reads an input image
    img = gray

    # find frequency of pixels in range 0-255
    histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(histr)
    histr = np.squeeze(histr,axis=1)[1:-1]
    print(histr)
    # show the plotting graph of an image
    plt.hist(histr, bins=np.arange(1,256))
    plt.title("histogram")
    plt.show()

def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def read_image(file_path):
    im_gray = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    # plot_histo(im_gray)
    thresh = 220
    im_bi = np.where(im_gray < thresh, 0, im_gray)
    # show_image(im_bi)
    im_aq = cv2.equalizeHist(im_bi)
    # show_image(im_aq)

    kernel = np.ones((1, 1), np.uint8)
    erosion = cv2.erode(im_aq, kernel, iterations=1)
    dilate = cv2.dilate(erosion, kernel, iterations=2)
    show_image(dilate)

    # dilate = im_aq
    return im_gray, dilate

# Method 1
im_gray, image = read_image(image_path)
Verti2 = np.concatenate((im_gray, image), axis=1)
show_image(Verti2)
rows = image.shape[0]
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows,
                               param1=10, param2=30,
                               minRadius=0, maxRadius=30)
print(circles)
src = image.copy()
src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle center
        cv2.circle(src, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(src, center, radius, (255, 0, 0), 2)
        cv2.putText(src, str(center), (i[0]+20, i[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)
show_image(src)



# Method 2
# Set our filtering parameters
# Initialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
# Set Area filtering parameters
params.filterByArea = True
params.minArea = 20
# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.8

# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.7
# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.6
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs
keypoints = detector.detect(image)
# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
im = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
x, y, size = keypoints[0].pt[0], keypoints[0].pt[1], keypoints[0].size
print(x,y,size)
x,y = int(x), int(y)
center = (x,y)
blobs = cv2.drawKeypoints(im, keypoints, blank, (255, 0, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# circle center
cv2.circle(blobs, center, 1, (255, 255, 0), 1)
cv2.putText(blobs, str(center), (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

show_image(blobs)
show_image(im)

