import cv2
import numpy as np
import vector
import matplotlib.pyplot as plt
from numpy.linalg import inv
#from msvcrt import getch

nwindows = 1
margin=110 
minpix=50
ym_per_pix = 30/720 
xm_per_pix = 3.7/700 

def calculateCurvature(yRange, left_fit_cr):
	"""
	Returns the curvature of the polynomial `fit` on the y range `yRange`.
	"""
	
	return ((1 + (2*left_fit_cr[0]*yRange*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

def correct_dist(initial_img):
	k = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
		 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
	k = np.array(k)
	# Distortion Matrix
	dist = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
	dist = np.array(dist)
	img_2 = cv2.undistort(initial_img, k, dist, None, k)

	return img_2


def image_prep(img):

	crop_img = img[420:720, 40:1280, :]  # To get the region of interest
	gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
	#mat_h = np.array([[-2.50638675e+00, -4.84647803e+00, 1.44429332e+03], [-1.24287961e+00, -2.34766149e+01, 1.81515663e+03], [-2.19653696e-03, -3.38621948e-02, 1.00000000e+00]])

	#print(mat_h)

	top_img = cv2.warpPerspective(gray_image, H_matrix, (300, 600))
	cv2.imshow('top_img',top_img)
	undist_img = correct_dist(crop_img)
	cv2.imshow('un-distored image', undist_img)

	hsl_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HLS)
	cv2.imshow('hsl', hsl_img)
	# print(hsl_img.shape)

	# TO seperate out Yellow color
	lower_mask_yellow = np.array([20, 120, 80], dtype='uint8')
	upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
	mask_yellow = cv2.inRange(hsl_img, lower_mask_yellow, upper_mask_yellow)

	yellow_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)
	cv2.imshow('yellow', yellow_detect)

	# To seperate out White color
	lower_mask_white = np.array([0, 200, 0], dtype='uint8')
	upper_mask_white = np.array([255, 255, 255], dtype='uint8')
	mask_white = cv2.inRange(hsl_img, lower_mask_white, upper_mask_white)

	white_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)
	cv2.imshow('white', white_detect)

	# Combine both
	lanes = cv2.bitwise_or(yellow_detect, white_detect)
	cv2.imshow('lanes', lanes)

	new_lanes = cv2.cvtColor(lanes, cv2.COLOR_HLS2BGR)
	cv2.imshow('new_lanes', new_lanes)
	final = cv2.cvtColor(new_lanes, cv2.COLOR_BGR2GRAY)

	cv2.imshow('final_11', final)

	img_blur = cv2.bilateralFilter(final, 9, 120, 100)
	cv2.imshow('blur', img_blur)


	img_edge = cv2.Canny(img_blur, 100, 200)
	cv2.imshow('canny', img_edge)



	#cv2.circle(undist_img, (516, 50), 5, (0, 0, 255), -1)  # LU
	#cv2.circle(undist_img, (686, 41), 5, (0, 0, 255), -1)  # RU
	#cv2.circle(undist_img, (1068, 253), 5, (0, 0, 255), -1)  # RB
	#cv2.circle(undist_img, (251, 259), 5, (0, 0, 255), -1)  # LB
	#cv2.imshow('controu', undist_img)


	# mat_h = np.array([[-2.50638675e+00, -4.84647803e+00, 1.44429332e+03], [-1.24287961e+00, -2.34766149e+01, 1.81515663e+03], [-2.19653696e-03, -3.38621948e-02, 1.00000000e+00]])

	# #print(mat_h)

	
	new_img = cv2.warpPerspective(img_edge, H_matrix, (300, 600))
	histogram = np.sum(new_img, axis=0)
	out_img = np.dstack((new_img,new_img,new_img))*255
	window_img = np.zeros_like(out_img)

	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint


	# Set height of windows
	window_height = np.int(new_img.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = new_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	
	# left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = new_img.shape[0] - (window+1)*window_height
		win_y_high = new_img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
	
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

#	left_XY=zip(leftx,lefty)
#	print(left_XY.shape)
	# Fit a second order polynomial to each
	
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	# Fit a second order polynomial to each
	# left_fit_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
	# right_fit_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
	ploty = np.linspace(0, new_img.shape[0]-1, new_img.shape[0] )


	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	cv2.imshow('hist',out_img)

	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	#y_eval = np.max(ploty)
	# Fit new polynomials to x,y in world space
	#left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	#right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
	# Calculate the new radii of curvature
	#leftCurvature = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	#rightCurvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
	# Now our radius of curvature is in meters
	#print(left_curverad, 'm', right_curverad, 'm')
	#print('Left : {:.2f} m, Right : {:.2f} m'.format(leftCurvature, rightCurvature))

	
	
	
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)
	# plt.show()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                              ploty])))])
	pts = np.hstack((left_line_window1, left_line_window2))
	pts = np.array(pts, dtype=np.int32)
	#right_line_window1 = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
	#right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
     #                         ploty])))])
	#right_line_pts = np.hstack((right_line_window1, right_line_window2))

#	print(left_lane_inds)
	#np.reshape(left_lane_inds,(left_lane_inds.shape[0])
	#np.reshape(right_lane_inds,(right_lane_inds.shape[0])
#	right_lane_inds.reshape(,-1)
#	print(left_lane_inds.shape)
	#left_lane_inds = [left_lane_inds, right_lane_inds]
	#pts = np.hstack((left_lane_inds, right_lane_inds))
	color_warp = np.zeros_like(img).astype(np.uint8)
	cv2.fillPoly(color_warp, pts, (0,255, 0))
	cv2.imshow('green', color_warp)
	#cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	#cv2.imshow('result', new_img)	
	newwarp = cv2.warpPerspective(color_warp, inv(H_matrix), (crop_img.shape[1], crop_img.shape[0]))
	result = cv2.addWeighted(crop_img, 1, newwarp, 0.3, 0)
	cv2.imshow('result-image',result)

	# oblique_img = cv2.warpPerspective(result, inv(mat_h), (crop_img.shape[1],crop_img.shape[0]))
	# f_image = cv2.bitwise_or(oblique_img,crop_img)
	# cv2.imshow('oblique',f_image)
	#ax.plot(left_fitx, ploty, color='yellow')
	#ax.plot(right_fitx, ploty, color='yellow')
	# cv2.imshow('new', new_img)
	#print(crop_img.shape)
	#cv2.imshow('img', crop_img)




	"""lines = cv2.HoughLinesP(img_edge, 0.5, np.pi/180, 20, None, 180, 120)

	Lhs = np.zeros((2, 2), dtype=np.float32)
	Rhs = np.zeros((2, 1), dtype=np.float32)
	x_max = 0
	x_min = 2555
	for line in lines:
		for x1, y1, x2, y2 in line:
			# Find the norm (the distances between the two points)
			normal = np.array([[-(y2 - y1)], [x2 - x1]], dtype=np.float32)  # question about this implementation
			normal = normal / np.linalg.norm(normal)

			pt = np.array([[x1], [y1]], dtype=np.float32)

			outer = np.matmul(normal, normal.T)

			Lhs += outer
			Rhs += np.matmul(outer, pt)  # use matmul for matrix multiply and not dot product

			cv2.line(crop_img, (x1, y1), (x2, y2), (255, 0, 0), thickness=4)

			x_iter_max = max(x1, x2)
			x_iter_min = min(x1, x2)
			x_max = max(x_max, x_iter_max)
			x_min = min(x_min, x_iter_min)

	width = x_max - x_min
	print('width : ', width)
	# Calculate Vanishing Point
	vp = np.matmul(np.linalg.inv(Lhs), Rhs)

	print('vp is : ', vp)
	plt.plot(vp[0], vp[1], 'c^')
	plt.imshow(crop_img)
	plt.show() """
	



	#cv2.waitKey(0)


	#hist, bins = np.histogram(img_edge.ravel(), 256, [0, 256])
	#cv2.waitKey(0)


src = np.array([[500, 50], [686, 41], [1078, 253], [231, 259]], dtype="float32")
dst = np.array([[50, 0], [250, 0], [250, 500], [0, 500]], dtype="float32")
##############  Homography  #########
H_matrix = cv2.getPerspectiveTransform(src, dst)
Hinv = inv(H_matrix)


# Code Starts to run from here
cap = cv2.VideoCapture('project_video.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
# print(fps)

while cap.isOpened():
	success, frame = cap.read()
	if success is False:
		break
	initial_image = frame
	# print(initial_image.shape)
	image_prep(initial_image)

	if cv2.waitKey(15) & 0xff == 27:  # To get the correct frame rate
		cv2.destroyAllWindows()
		break
cap.release()