import numpy as np 
import cv2
import glob
import sys


'''
This function calibrates the camera. Make sure there are at least 10 pictures for an ok calibration.
Define the number of rows and colums and their dimension.
Define a working directory (ex workingdirectory = "./directory")
And give the image type.

The code will run through all the images that are in the working folder and find the chessboard corners.
Every image from which the corners are found are saved and will be used for finding the camera matrix and distortion matrix.
The distortion and camera matric will be saved in the working folder as a txt file
'''

def calibrate(nRows, nCols, dimension, Workingdirectory, imageType):
	#termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

	#prepare object points, like (0,0,0), (1,0,0), (2,0,0), ... , (6,5,0)
	objp = np.zeros((nRows*nCols,3), np.float32)
	objp[:,:2] = np.mgrid[0:nCols,0:nRows].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images
	objpoints = [] # 3d point in real wordl space
	imgpoints = [] # 2d points in image plane


	#find image files
	filename = Workingdirectory + "/*." + imageType
	images   = glob.glob(filename)

	print (len(images), "images found")
	if len(images) < 10:
		print ("Not enough images were found: at least 10!!")
		sys.exit()

	else:
		nPatternFound = 0
		imgNotGood = images[1]

		for fname in images:
			#if 'calibresult' in fname: continue
			# -- read the file and convert in greyscale
			img = cv2.imread(fname)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

			print ("Reading image", fname)

			# Find the chess Board corners
			ret, corners = cv2.findChessboardCorners(gray, (nCols,nRows),None)
			# If found, add object points, image points (after refining them)
			if ret == True:
				print ("Pattern found! Press ESC to skip or ENTER to accept")
				# Sometimes, corners fail with not good quality pictures so:
				corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

				#Draw and display corners
				cv2.drawChessboardCorners(img,(nCols,nRows),corners2,ret)
				cv2.namedWindow('img', cv2.WINDOW_NORMAL)
				cv2.resizeWindow('img',800,800)
				cv2.imshow('img',img)
				#cv2.waitKey(0)
				k = cv2.waitKey(0) & 0xFF
				if k == 27: #-- that is ESC Button
					print ("Image Skipped")
					imgNotGood = fname
					continue

				print ("Image Accepted")
				nPatternFound += 1
				objpoints.append(objp)
				imgpoints.append(corners2)

				#cv2.waitkey(0)
			else:
				imgNotGood = fname
				print("Image was not good")

	cv2.destroyAllWindows()

	if (nPatternFound > 1):
		print ("Found %d good images" % (nPatternFound))
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

		# Undistort an image
		img = cv2.imread(imgNotGood)
		h,  w = img.shape[:2]
		print ("Image to undistort: ", imgNotGood)
		newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

		# undistort
		mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
		dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

		#crop image
		x,y,w,h = roi
		dst = dst[y:y+h, x:x+w]
		print ("ROI: ", x, y, w, h)

		cv2.imwrite(Workingdirectory + "/calibresult.png", dst)
		print ("Calibrated picture saved as calibresult.png")
		print ("Calibration Matrix: ")
		print (mtx)
		print ("Disortion: ", dist)

		#--------- Save result
		filename = Workingdirectory + "/cameraMatrix.txt"
		np.savetxt(filename, mtx, delimiter=',')
		filename = Workingdirectory + "/cameraDistortion.txt"
		np.savetxt(filename, dist, delimiter=',')

		mean_error = 0
		for i in range(len(objpoints)):
			imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
			error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
			mean_error += error

		print ("total error: ", mean_error/len(objpoints))

	else:
		print ("In order to calibrate you need at least 9 good pictures... try again")
