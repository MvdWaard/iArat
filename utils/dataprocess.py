import cv2
import numpy as np

import iArat.utils



def find4Dpoints(x_s,y_s,x_i,y_i,proj1,proj2):
	'''
	Find the 4D points of the CSV Data
	Returns the 3D coordinates which are in the homogeneous 
	'''
	s = np.array([np.asarray(x_s,dtype=np.float32),np.asarray(y_s,dtype=np.float32)])
	i = np.array([np.asarray(x_i,dtype=np.float32),np.asarray(y_i,dtype=np.float32)])
	P4D = cv2.triangulatePoints(proj1, proj2, s, i) 

	x3D = []
	y3D = []
	z3D = []
	for i in range(len(P4D[1])):
	    x3D.append((P4D[0][i] / P4D[3][i]))
	    y3D.append((P4D[1][i] / P4D[3][i]))
	    z3D.append((P4D[2][i] / P4D[3][i]))
	XYZ = np.vstack([x3D, y3D, z3D]).transpose()
	return XYZ

def find4Dcorners(srcPts,dstPts,P):
	'''
	Finds the 4D positions of the chessboard corners.
	Returns the 3D coordinates and the projections
	'''
	K1, K2, dist1, dist2 = iArat.utils.unpackP(P)
	srcPtsNorm = cv2.undistortPoints(srcPts, K1, dist1)
	dstPtsNorm = cv2.undistortPoints(dstPts, K2, dist2)
	# essential matrix, rotation matrix and translation vector
	E, mask = cv2.findEssentialMat(srcPtsNorm, dstPtsNorm, focal=1.00, pp=(0., 0.), method=cv2.RANSAC, prob=0.999)
	M, R, t, mask = cv2.recoverPose(E, srcPtsNorm, dstPtsNorm)

	#TRIANGULATE POINTS
	srcPtsTrans = (srcPts.reshape(-1, 2)).transpose(1, 0)
	dstPtsTrans = (dstPts.reshape(-1, 2)).transpose(1, 0)
	proj1 = np.column_stack((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
	proj2 = np.column_stack((R, t)) # 3x4 projection matrix of cameras
	proj1 = np.dot(K1, proj1)
	proj2 = np.dot(K2, proj2)
	# 4xN array of reconstructed points in homogeneous coordinates
	points4D = cv2.triangulatePoints(proj1, proj2, srcPtsTrans, dstPtsTrans) 

	#FIND COORDINATES FOR SCATTER PLOT
	x3D = []
	y3D = []
	z3D = []
	for i in range(len(srcPts)):
	    x3D.append((points4D[0][i] / points4D[3][i]))
	    y3D.append((points4D[1][i] / points4D[3][i]))
	    z3D.append((points4D[2][i] / points4D[3][i]))

	XYZ = np.vstack([x3D, y3D, z3D]).transpose()
	return XYZ, proj1, proj2

def PosVelAcc(XYZR):
    '''
    Calculates the position, velocity and acceleration. 
    Uses a simple moving average filter to smoothen the data to get rid of high peaks after differentiation
    Filter order is hardcoded, but can be changed
    '''
    filterorder=4
    pos_x = iArat.utils.smooth(XYZR[:,0],filterorder)
    pos_y = iArat.utils.smooth(XYZR[:,1],filterorder)
    pos_z = iArat.utils.smooth(XYZR[:,2],filterorder)
    pos = np.column_stack([pos_x,pos_y,pos_z])
    pos=pos[2:,:]

    vel_x = iArat.utils.smooth(np.diff(pos_x[2:]),filterorder)
    vel_y = iArat.utils.smooth(np.diff(pos_y[2:]),filterorder)
    vel_z = iArat.utils.smooth(np.diff(pos_z[2:]),filterorder)
    vel = np.column_stack([vel_x,vel_y,vel_z])
    vel = np.insert(vel,0,0,axis=0)

    acc_x = iArat.utils.smooth(np.diff(vel_x),filterorder)
    acc_y = iArat.utils.smooth(np.diff(vel_y),filterorder)
    acc_z = iArat.utils.smooth(np.diff(vel_z),filterorder)
    acc = np.column_stack([acc_x,acc_y,acc_z])
    acc = np.insert(acc,0,0,axis=0)
    acc = np.insert(acc,0,0,axis=0)

    PVA = np.column_stack([pos,vel,acc])

    return PVA