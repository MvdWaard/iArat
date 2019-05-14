import cv2
import numpy as np


'''
Find the corners of the chessboard which is 6x9 when given 2 images.
Make sure to use the correct number of rows and columns

'''

def chess_corners(img1, img2, chessRow, chessCol, loadfolder1, loadfolder2):
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# #SET WORKSPACE AND LOAD INITIAL MATRICES AND VECTORS
	K1 = np.loadtxt(loadfolder1+"/cameraMatrix.txt", delimiter=",")
	K2 = np.loadtxt(loadfolder2+"/cameraMatrix.txt", delimiter=",")

	K1[0][0], K1[1][1], K1[0][2], K1[1][2] = K1[0][0], K1[1][1], K1[0][2], K1[1][2]
	K2[0][0], K2[1][1], K2[0][2], K2[1][2] = K2[0][0], K2[1][1], K2[0][2], K2[1][2]
	dist1 = np.loadtxt(loadfolder1+"/cameradistortion.txt", delimiter=",")
	dist2 = np.loadtxt(loadfolder2+"/cameradistortion.txt", delimiter=",")
	P = np.hstack([K1[0, 0], K1[0, 2], K1[1, 1], K1[1, 2], K2[0, 0], K2[0, 2], K2[1, 1], K2[1, 2], dist1, dist2])

	#SHOW IMAGES
	# cv2.namedWindow('img1', cv2.WINDOW_NORMAL),cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('img1', 800, 800), cv2.resizeWindow('img2', 800, 800)
	# cv2.imshow("img1", img1), cv2.imshow("img2", img2)
	# #cv2.waitKey(0)
	# cv2.destroyAllWindows()

	#FIND CHESS BOARD CORNERS IN IMAGES
	imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	found1, corners1 = cv2.findChessboardCorners(imgGray1, (chessCol, chessRow), None)
	found2, corners2 = cv2.findChessboardCorners(imgGray2, (chessCol, chessRow), None)

	#REFINE CORNERS
	if found1 is True and found2 is True:
		cornersRefined1 = cv2.cornerSubPix(imgGray1, corners1, (11, 11), (-1, -1), criteria)
		cornersRefined2 = cv2.cornerSubPix(imgGray2, corners2, (11, 11), (-1, -1), criteria)
		img1 = cv2.drawChessboardCorners(img1, (chessCol, chessRow), cornersRefined1, found1)
		img2 = cv2.drawChessboardCorners(img2, (chessCol, chessRow), cornersRefined2, found2)   
		cv2.namedWindow('img1', cv2.WINDOW_NORMAL), cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('img1', 960, 540)
		cv2.resizeWindow('img2', 960, 540)
		cv2.imshow('img1', img1) # display labelled images
		cv2.imshow('img2', img2)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print('Please use another image')

	srcPts = cornersRefined1
	dstPts = cornersRefined2

	return srcPts, dstPts, P