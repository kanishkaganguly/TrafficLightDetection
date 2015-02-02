import numpy as np
import cv2

def main():
	red_val = 0
	green_val = 65

	# Make a window for the video feed  
	#cv2.namedWindow('frame',cv2.CV_WINDOW_AUTOSIZE)

	# Make the trackbar used for HSV masking    
	#cv2.createTrackbar('HSV','frame',0,255,getVal)

	# Capture Video from Webcam
	cap = cv2.VideoCapture(1)

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Name the variable used for mask bounds
	    #j = cv2.getTrackbarPos('HSV','frame')

	    # Convert BGR to HSV
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    	# define range of RED color in HSV
		red_lower = np.array([red_val-10,100,100])
		red_upper = np.array([red_val+10,255,255])

    	# define range of GREEN color in HSV
		green_lower = np.array([green_val-10,100,100])
		green_upper = np.array([green_val+10,255,255])

	    # Threshold the HSV image to get only selected color
		red_mask = cv2.inRange(hsv, red_lower, red_upper)
		green_mask = cv2.inRange(hsv, green_lower, green_upper)

		# Bitwise-AND mask the original image
		red_res = cv2.bitwise_and(frame,frame, mask= red_mask)
		green_res = cv2.bitwise_and(frame,frame, mask= green_mask)

		# Structuring Element
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	    
		# Morphological Closing
		red_closing = cv2.morphologyEx(red_res,cv2.MORPH_CLOSE,kernel)
		green_closing = cv2.morphologyEx(green_res,cv2.MORPH_CLOSE,kernel)

		#Convert to Black and White image
		red_gray = cv2.cvtColor(red_closing, cv2.COLOR_BGR2GRAY)
		green_gray = cv2.cvtColor(green_closing, cv2.COLOR_BGR2GRAY)
		(thresh1, red_bw) = cv2.threshold(red_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		(thresh2, green_bw) = cv2.threshold(green_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

		# Count pixel changes
		red_black = cv2.countNonZero(red_bw)
		if red_black > 20000:
			print "RED"

		green_black = cv2.countNonZero(green_bw)
		if green_black > 18000:
			print "GREEN"

		# Display the resulting frame
		#both = np.hstack((red_bw,green_bw))
		#cv2.imshow('RED_GREEN', both)

	    # Press q to quit
		#if cv2.waitKey(3) & 0xFF == ord('q'):
		#	break

	# When everything is done, release the capture
	cap.release()
	#cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
