import cv2
import numpy as np

# The classic get contours, with customized parameters to fit our use case of document scanning
def getContours(img,imgContour):

    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    max_area = 0
    biggest_dpc = None
    
    for cnt in contours:
    
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 :
            area = cv2.contourArea(approx)
            if area > 500 and area > max_area:
                max_area = area
                biggest_dpc = approx

            
    if biggest_dpc is not None:
        cv2.drawContours(imgContour, [biggest_dpc], -1, (0, 255, 0), 5)

    return biggest_dpc

#################################################################

# reordring the points of the contour to be in the order of top-left, top-right, bottom-right, bottom-left
def reorder(biggest):
    biggest = biggest.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = biggest.sum(1)
    new_points[0] = biggest[np.argmin(add)] # top-left point has the smallest sum
    new_points[3] = biggest[np.argmax(add)] # bottom-right point has the largest sum
    diff = np.diff(biggest, axis=1)
    new_points[1] = biggest[np.argmin(diff)] # top-right point has the smallest difference
    new_points[2] = biggest[np.argmax(diff)] # bottom-left point has the largest difference
    return new_points


##################################################################

# applying the perspective transform to get a top-down view of the document
def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [PROC_WIDTH, 0], [0, PROC_HEIGHT], [PROC_WIDTH, PROC_HEIGHT]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (PROC_WIDTH, PROC_HEIGHT))
    return imgOutput



##################################################################


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

PROC_WIDTH = 450
PROC_HEIGHT = 360


while True:
    # setting up the frames
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (PROC_WIDTH, PROC_HEIGHT))
    imgcontour = frame.copy()
    grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # setting up the kernel for dilation and erosion
    kernal = np.ones((5,5),np.uint8)

    # Gaussian blur to reduce noise before edge detection
    blurred_frame = cv2.GaussianBlur(grey_frame, (5, 5), 0)

    edges = cv2.Canny(grey_frame, 130, 50) # edge detection
    edges_blurred = cv2.Canny(blurred_frame, 130, 50) # edge detection on blurred image to reduce noise
    dilated_image = cv2.dilate(edges_blurred, kernal, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernal, iterations=1) # the final result that we will use for contour detection and document scanning

    biggest_dpc = getContours(eroded_image,imgcontour)

    # handling the case when no document is found
    warped_window_open = False
    if biggest_dpc is not None:
        warped_image = getWarp(frame, biggest_dpc)
        gaus = cv2.adaptiveThreshold(cv2.cvtColor(warped_image,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,99,2)
        cv2.imshow('Warped Image', warped_image)
        cv2.imshow('Adaptive Gaussian Threshold', gaus)
        warped_window_open = True

    else:
        if warped_window_open:
            cv2.destroyWindow('Warped Image')
            warped_window_open = False


    eroded_image = np.stack((eroded_image,)*3, axis=-1) # Convert single channel to 3 channels for stacking

    # stacing images for better visualization(original image, contour detection, and warped output)
    
    stacked_img = np.hstack((frame, eroded_image, imgcontour))
    
    cv2.imshow('Stacked Images', stacked_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
