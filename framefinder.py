import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import random
from mpl_toolkits.mplot3d import Axes3D
import sys

import iArat.utils

'''
This code uses a calibration video and a calibration image to find a good match of calibration images.
This is done by extracting random frames from the video and reconstructiong the chessboard.
The code asks for optimization.
At the end, the reconstructed chessboard is shown and may be accepted or not.
'''

def findframes(Cal_img1_location, Cal_vid_location, Cal_vid, chessRow, chessCol, Nr_of_frames,loadfolder1, loadfolder2):

	img1 = cv2.imread(Cal_img1_location)
	vidcap = cv2.VideoCapture(Cal_vid_location + "\\" + Cal_vid)

	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("Length of the video has "+str(length)+" frames!")


	optimize = input("Use optimization? [y/n]: ")

	frame_array = random.sample(range(length), Nr_of_frames)
	for x in frame_array:
		ex_frame = x

		if ex_frame > length:
			print("Choose a number under the %d for extraction!" % length)
			#Extracting frames from the video to create the tracking lines:
		else:
			Extract = os.path.isfile(Cal_vid_location + "\\Cal_%d.jpg"% ex_frame)
			if Extract == False:
				print('Extracting the frame')
				vidcap = cv2.VideoCapture(Cal_vid_location + "\\" + Cal_vid)
				success,image = vidcap.read()
				count = 0
				while success:
					if count == ex_frame:
						cv2.imwrite(Cal_vid_location + "\\Cal_%d.jpg" % count, image)     # save frame as JPEG file      
						print('Frame extracted!')
					success,image = vidcap.read()
					count += 1



		img2 = cv2.imread(Cal_vid_location + "\\Cal_%d.jpg" % ex_frame)

		srcPts, dstPts, P = iArat.utils.chess_corners(img1, img2, chessRow, chessCol,loadfolder1, loadfolder2)

		if optimize == 'y':
			print("Optimizing")
			res = scipy.optimize.minimize(iArat.utils.fit_err, P, args=(srcPts, dstPts, chessRow, chessCol), method='Nelder-Mead')
			c, normal, XYZ, resid, proj1, proj2 = iArat.utils.reconAndFit(res.x, srcPts, dstPts, chessRow, chessCol)

		else:
			XYZ, proj1, proj2 = iArat.utils.find4Dcorners(srcPts,dstPts,P)

		x3D, y3D, z3D = XYZ.T

		#ROTATE THE CHESSBOARD TO THE AXIS WE WANT
		origin = len(x3D)-chessCol
		originX = x3D[origin]
		originY = y3D[origin]
		originZ = z3D[origin]

		x3D -= originX
		y3D -= originY
		z3D -= originZ

		#FINDING ROTATION MATRIX
		CBx = np.array([x3D[-1], y3D[-1], z3D[-1]])
		CBx /= np.linalg.norm(CBx)
		CBy = np.array([x3D[0],  y3D[0],  z3D[0]])
		CBy /= np.linalg.norm(CBy)
		CBz = np.cross(CBx, CBy)
		R = np.vstack([CBx[None], CBy[None], CBz[None]]).T

		#ROTATING
		p3D = np.concatenate([x3D[:,None], y3D[:,None], z3D[:,None]], axis=1)
		p3DR = np.matmul(p3D[:,None], R).squeeze()

		fig = plt.figure()
		ax = plt.gca(projection='3d')
		ax.plot([0,], [0,], [0,], 'ro')
		ax.plot(p3DR[:,0], p3DR[:,1], p3DR[:,2], 'c^')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		iArat.utils.set_axes_equal(ax)
		plt.show()
		Accept = input("Want to accept this image? [y/n]: ")
		if Accept == 'n': #-- that is a Button
			print("Frame not accepted")
			continue
		if Accept == 'y':
			print ("Frame Accepted")
			break

	if Accept== 'y':
		print("Chosen frame to be good is %d" %ex_frame)
	else:
		print("No good frames were found within the %d extracted frames. Increase Nr_of_frames or try again" % Nr_of_frames)