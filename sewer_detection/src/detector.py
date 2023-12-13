#!/usr/bin/env python3
from sensor_msgs.msg import Image
import rospy
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge,CvBridgeError
import cv2
from std_msgs.msg import  Header
import numpy as np
import time
import psutil
import os
from pyAAMED import pyAAMED


from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Sewer_manhole_detection:

    def __init__(self ):
       
        # OpenCV bridge
        self.bridge = CvBridge()     
        self.image_sub = rospy.Subscriber('tracking', Image, self.get_center)
        self.min_area  = 2000  # unit in pixel
        self.max_area  = 20000
        self.min_aspect_ratio  = 0.5
        self.max_aspect_ratio  = 2
    
    def ellipse_detection(self, image):  

        '''
        filtered_cnts=[]
        aamed = pyAAMED(600, 700)

        aamed.setParameters(3.1415926/3, 3.4, 0.5)
        res = aamed.run_AAMED(image)
        for ellipse in res:
            area=np.pi*(ellipse[2]/2) *(ellipse[3]/2)
            if self.min_area < area < self.max_area:
                aspect_ratio = float(max(ellipse[3],ellipse[2])) / min(ellipse[3], ellipse[2])
                # if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio :
                filtered_cnts.append(ellipse)   
        return filtered_cnts
        '''

        #image_gray = color.rgb2gray(image)
        image_gray = image
        edges = canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
        print("edges")
        # Display all ellipses
       
        # Perform a Hough Transform
        # The accuracy corresponds to the bin size of the histogram for minor axis lengths.
        # A higher `accuracy` value will lead to more ellipses being found, at the
        # cost of a lower precision on the minor axis length estimation.
        # A higher `threshold` will lead to less ellipses being found, filtering out those
        # with fewer edge points (as found above by the Canny detector) on their perimeter.
        result = hough_ellipse(image_gray, accuracy=20, threshold=250, min_size=100, max_size=120)
        print("DOP!")
        '''
        result.sort(order='accumulator')

        # Estimated parameters for the ellipse
        best = list(result[-1])
        yc, xc, a, b = (int(round(x)) for x in best[1:5])
        orientation = best[5]

        # Draw the ellipse on the original image
        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        image_rgb[cy, cx] = (0, 0, 255)
        # Draw the edge (white) and the resulting ellipse (red)
        edges = color.gray2rgb(img_as_ubyte(edges))
        edges[cy, cx] = (250, 0, 0)

        fig2, (ax1, ax2) = plt.subplots(
            ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True
        )

        ax1.set_title('Original picture')
        ax1.imshow(image_rgb)

        ax2.set_title('Edge (white) and result (red)')
        ax2.imshow(edges)
        '''
        #plt.show()
        


    def get_center(self,image):

       
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image,'bgr8')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate the new camera matrix and obtain the maps for undistortion and rectification
            #new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D,self.DIM, np.eye(3))
            #map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_K, self.DIM, cv2.CV_32FC1)
            # Undistort the whole image
            #undistorted_image = cv2.remap(cv_image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
           
        except CvBridgeError as e:
            print(e) 
        
        # Apply GaussianBlur to reduce noise and help with edge detection
        gray_blurred = cv2.GaussianBlur(cv_image, (5, 5), 0)
    
        # Use the HoughCircles function to detect ellipses
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Draw the detected ellipses on the original image
            for i in circles[0, :]:
                # Draw the outer circle
                cv2.circle(cv_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(cv_image, (i[0], i[1]), 2, (0, 0, 255), 3)

            # Display the result
            cv2.imshow('Ellipses Detected', cv_image)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        else:
            print("No ellipses detected.")

        

if __name__ == '__main__':

    rospy.init_node('Sewer_manhole_detection_node') 
    node = Sewer_manhole_detection()
    rospy.spin()
        


