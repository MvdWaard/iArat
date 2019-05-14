import cv2
import numpy as np
import sys

import iArat.utils



def unpackP(p):
	'''
	Unpacks P to re create the K and distortion matrixes.

	'''
	K1 = np.zeros([3, 3])
	K2 = np.zeros([3, 3])
	K1[0,0] = p[0]
	K1[0,2] = p[1]
	K1[1,1] = p[2]
	K1[1,2] = p[3]
	K1[2,2] = 1
	K2[0,0] = p[4]
	K2[0,2] = p[5]
	K2[1,1] = p[6]
	K2[1,2] = p[7]
	K2[2,2] = 1
	dist1 = p[8:13]
	dist2 = p[13:]

	return K1, K2, dist1, dist2

def fitPlaneLTSQ(XYZ):
	'''
	Finds the error between the fitted plane and the chessboard
	'''
	rows, cols = XYZ.shape
	G = np.ones((rows, 3))
	G[:, 0] = XYZ[:, 0]  #X
	G[:, 1] = XYZ[:, 1]  #Y
	Z = XYZ[:, 2]
	#Return the least-squares solution to a linear matrix equation
	(a, b, c), resid, rank, s = np.linalg.lstsq(G, Z) 
	normal = (a, b, -1)
	nn = np.linalg.norm(normal)
	normal = normal / nn

	return (c, normal, resid)

def reconAndFit(p, srcPts, dstPts, chessRow, chessCol):
	'''
	Finds the corners, fits a plane and find residuals to see how good the fit is.
	The weight of the residuals can be changed to find a better optimization.
	Resid1 = Fitting to the plane
	Resid2 = the distance error between the corners
	Resid3 = the length error between the first and last column
	Resid4 = the length error between the first and last row
	resid5 = the error in scale between the row and column length
	resid6 - resid9 = the error in angle of the corners of the chessboard
	'''
	ChessCol = 6
	XYZ, proj1, proj2 = iArat.utils.find4Dcorners(srcPts,dstPts,p)
	c, normal, resid1 = iArat.utils.fitPlaneLTSQ(XYZ)

	x3D, y3D, z3D = XYZ.T
	x3D = np.asarray(x3D)
	y3D = np.asarray(y3D)
	z3D = np.asarray(z3D)

	# find an aspect ratio for grid
	x3Dm = x3D.reshape(chessRow, chessCol)
	y3Dm = y3D.reshape(chessRow, chessCol)
	z3Dm = z3D.reshape(chessRow, chessCol)

	p3Dm = np.concatenate([x3Dm[...,None], y3Dm[...,None], z3Dm[...,None]], axis=2)

	edist = lambda x, y: np.sqrt(np.sum((x-y)**2))

	# cols
	distsC = np.zeros((chessRow, chessCol - 1))
	for i in range(chessRow):
		for j in range(chessCol - 1):
			distsC[i,j] = edist(p3Dm[i,j], p3Dm[i,j+1])
	#rows
	distsR = np.zeros((chessRow - 1, chessCol))
	for i in range(chessCol):
		for j in range(chessRow-1):
			distsR[j,i] = edist(p3Dm[j,i], p3Dm[j+1,i])

	#Means
	resid2 = (1 - (distsR.mean()/distsC.mean()))**2
	#print(distsR.mean()), print(distsC.mean())
	TdistC = distsC.sum(axis=1)
	TdistR = distsR.sum(axis=0)

	#Rows and Colums lengths
	resid3 = (1-(TdistC[0]/TdistC[-chessRow]))**2
	resid4 = (1-(TdistR[0]/TdistR[-chessCol]))**2

	#Scale
	#resid5 = ((chessRow/chessCol)-(TdistR.mean()/TdistC.mean()))**2
	resid5 = ((chessRow/chessCol)-(TdistR[0]/TdistC[0]))**2

	#Angles
	def FindAngle(XYZ, Position, v1, v2):
		v1_u = XYZ[v1]-XYZ[Position]
		v1_u /=np.linalg.norm(XYZ[v1]-XYZ[Position])
		v2_u = XYZ[v2]-XYZ[Position]
		v2_u /= np.linalg.norm(XYZ[v2]-XYZ[Position])
		Ang_rad = np.arccos(np.clip(np.dot(v1_u,v2_u), -1.0,1.0))
		return Ang_rad

	Ang_rad1 = FindAngle(XYZ, (len(x3D)-chessCol), 0, -1)            #Angle at XYZ[48]
	Ang_rad2 = FindAngle(XYZ, 0, (len(x3D)-chessCol), (chessCol-1))  #Angle at XYZ[0]
	Ang_rad3 = FindAngle(XYZ, (chessCol-1), 0, -1)                   #Angle at XYZ[5]
	Ang_rad4 = FindAngle(XYZ, -1, (len(x3D)-chessCol), (chessCol-1)) #Angle at XYZ[-1]

	resid6 = (1-1.57/Ang_rad1)**2 # Smalles residu when the angle is 90 degrees (=1.57 radians)
	#resid6 = (1-(Ang_rad1/Ang_rad2/Ang_rad3/Ang_rad4/1.57))**2

	#print(resid6, a, b)
	resid7 = (1-1.57/Ang_rad2)**2
	resid8 = (1-1.57/Ang_rad3)**2
	resid9 = (1-1.57/Ang_rad4)**2
	                     
	#Total resid just give a factor to the ones you want to use:
	resid = 0*resid1+0*resid2+0*resid3+0*resid4+2*resid5+10*resid6+0*resid7+0*resid8+0*resid9


	sys.stdout.write("#")
	sys.stdout.flush()

	return c, normal, XYZ, resid, proj1, proj2

def fit_err(p, srcPts, dstPts, chessRow, chessCol):
	'''
	uses the recon and fit function to find the optimal K and dist matrixes
	'''
	_1, _2, _3, resid, _4, _5 = iArat.utils.reconAndFit(p, srcPts, dstPts, chessRow, chessCol)

	return resid

def smooth(y, filterorder):
	'''
	Simple moving average filter that is used for smoothening the data
	'''
	box = np.ones(filterorder)/filterorder
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth
