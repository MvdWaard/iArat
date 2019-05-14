import cv2
import csv
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import math
import pylab as py
from mpl_toolkits.mplot3d import Axes3D
import sys

import iArat
import iArat.utils

'''
This code processes the data.
First it finds the chessboard corners and with that the rotation matrix.
Then the CSV data is reconstructed into 3D data and plotted.
At the end the kinematics are plotted. The positions of all the joints will be plotted.
For now the position, velocity and acceleration are commented out, but can be plotted if wanted.

'''


def data_3D(CSV_data_sam, CSV_data_ipad, Img1_location, Img2_location, chessRow, chessCol, loadfolder1, loadfolder2):

    # Read the CSV file
    with open(CSV_data_sam) as data:
    	reader = csv.reader(data)
    	data_list_s = list(reader)
    with open(CSV_data_ipad) as data:
    	reader = csv.reader(data)
    	data_list_i = list(reader)


    data_T_s = list(map(list,zip(*data_list_s)))
    data_T_i = list(map(list,zip(*data_list_i)))

    # The CVS data is in pixels from a 1440:810 video. When tracking 
    # the 1920:1080 image, the data should be scaled:
    res = 1920/1920

    #Creating floats of the CSV data (Going from 5 to end bc data starts at 4th and first 2 frames can be black giving weird positions)
    frames_s    = [float(i) for i in data_T_s[0][5:]]
    x_hand_s    = [float(i)*res for i in data_T_s[1][5:]]
    y_hand_s    = [float(i)*res for i in data_T_s[2][5:]]
    llh_hand_s  = [float(i) for i in data_T_s[3][5:]]
    x_wrist_s   = [float(i)*res for i in data_T_s[4][5:]]
    y_wrist_s   = [float(i)*res for i in data_T_s[5][5:]]
    llh_wrist_s = [float(i) for i in data_T_s[6][5:]]
    x_elbow_s   = [float(i)*res for i in data_T_s[7][5:]]
    y_elbow_s   = [float(i)*res for i in data_T_s[8][5:]]
    llh_elbow_s = [float(i) for i in data_T_s[9][5:]]
    x_shoulder_s= [float(i)*res for i in data_T_s[10][5:]]
    y_shoulder_s= [float(i)*res for i in data_T_s[11][5:]]
    llh_shoulder_s=[float(i) for i in data_T_s[12][5:]]

    frames_i    = [float(i) for i in data_T_i[0][5:]]
    x_hand_i    = [float(i)*res for i in data_T_i[1][5:]]
    y_hand_i    = [float(i)*res for i in data_T_i[2][5:]]
    llh_hand_i  = [float(i) for i in data_T_i[3][5:]]
    x_wrist_i   = [float(i)*res for i in data_T_i[4][5:]]
    y_wrist_i   = [float(i)*res for i in data_T_i[5][5:]]
    llh_wrist_i = [float(i) for i in data_T_i[6][5:]]
    x_elbow_i   = [float(i)*res for i in data_T_i[7][5:]]
    y_elbow_i   = [float(i)*res for i in data_T_i[8][5:]]
    llh_elbow_i = [float(i) for i in data_T_i[9][5:]]
    x_shoulder_i= [float(i)*res for i in data_T_i[10][5:]]
    y_shoulder_i= [float(i)*res for i in data_T_i[11][5:]]
    llh_shoulder_i=[float(i) for i in data_T_i[12][5:]]

    img1 = cv2.imread(Img1_location)
    img2 = cv2.imread(Img2_location)


    srcPts, dstPts, P = iArat.utils.chess_corners(img1, img2, chessRow, chessCol,loadfolder1, loadfolder2)

    optimize = input("Use optimization? [y/n]: ")

    if optimize == 'y':
        print("Optimizing")
        # loops over <reconAndFit> so that K and dist best match the plane from <fitPlaneLTSQ>
        resi = scipy.optimize.minimize(iArat.utils.fit_err, P, args=(srcPts, dstPts, chessRow, chessCol), method='Nelder-Mead')
        # get final points for plot
        c, normal, XYZ, resid, proj1, proj2 = iArat.utils.reconAndFit(resi.x, srcPts, dstPts, chessRow, chessCol)

    if optimize == 'n':
        XYZ, proj1, proj2 = iArat.utils.find4Dcorners(srcPts,dstPts,P)
        c, normal, resid1 = iArat.utils.fitPlaneLTSQ(XYZ)


    # # loops over <reconAndFit> so that K and dist best match the plane from <fitPlaneLTSQ>
    # res = scipy.optimize.minimize(iArat.utils.fit_err, P, args=(srcPts, dstPts, chessRow, chessCol), method='Nelder-Mead')
    # # get final points for plot
    # c, normal, XYZ, resid, proj1, proj2 = iArat.utils.reconAndFit(res.x, srcPts, dstPts, chessRow, chessCol) 

    x3D, y3D, z3D = XYZ.T

    #CREATE POINTS FOR PLANE
    point = np.array([0.0, 0.0, c])
    d = -point.dot(normal)
    xx, yy = np.asarray(x3D), np.asarray(y3D)
    z = (-normal[0]*xx - normal[1]*yy - d)*1. / normal[2]


    #TRANSLATING TO ORIGIN
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


    #SHOW ROT RESULTS
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot([0,], [0,], [0,], 'ro')
    ax.plot(p3DR[:,0], p3DR[:,1], p3DR[:,2], 'c^')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    iArat.utils.set_axes_equal(ax)
    plt.show()



    #Find the 4D location of the tracked values from DLC
    XYZ_hand = iArat.utils.find4Dpoints(x_hand_s,y_hand_s,x_hand_i,y_hand_i,proj1,proj2)
    XYZ_wrist = iArat.utils.find4Dpoints(x_wrist_s,y_wrist_s,x_wrist_i,y_wrist_i,proj1,proj2)
    XYZ_elbow = iArat.utils.find4Dpoints(x_elbow_s,y_elbow_s,x_elbow_i,y_elbow_i,proj1,proj2)
    XYZ_shoulder = iArat.utils.find4Dpoints(x_shoulder_s,y_shoulder_s,x_shoulder_i,y_shoulder_i,proj1,proj2)
    
    #Translate to make 0,0 the first position of the shoulder
    XYZ_hand -= XYZ_shoulder[0]
    XYZ_wrist -= XYZ_shoulder[0]
    XYZ_elbow -= XYZ_shoulder[0]
    XYZ_shoulder -= XYZ_shoulder[0]
    
    #Rotate
    XYZR_hand = np.matmul(XYZ_hand[:,None], R).squeeze()
    XYZR_wrist = np.matmul(XYZ_wrist[:,None], R).squeeze()
    XYZR_elbow = np.matmul(XYZ_elbow[:,None], R).squeeze()
    XYZR_shoulder = np.matmul(XYZ_shoulder[:,None], R).squeeze()

    #Flip in Z-axis (because the pixels in Y go from up to down)
    XYZR_hand[:,2] *=  -1
    XYZR_wrist[:,2] *= -1
    XYZR_elbow[:,2] *= -1
    XYZR_shoulder[:,2] *= -1

    #Set beginning and ending of the showed tracked line (duration)
    beg1 = 0
    end1 = len(XYZR_hand)


    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot(XYZR_hand[beg1:end1,0], XYZR_hand[beg1:end1,1], XYZR_hand[beg1:end1,2], 'r')
    ax.plot(XYZR_wrist[beg1:end1,0], XYZR_wrist[beg1:end1,1], XYZR_wrist[beg1:end1,2], 'b')
    ax.plot(XYZR_elbow[beg1:end1,0], XYZR_elbow[beg1:end1,1], XYZR_elbow[beg1:end1,2], 'g')
    ax.plot(XYZR_shoulder[beg1:end1,0], XYZR_shoulder[beg1:end1,1], XYZR_shoulder[beg1:end1,2], 'c')
    ax.scatter(p3DR[:,0], p3DR[:,1], p3DR[:,2], 'c^', s=2)

    # ax.plot(XYZ_hand[:,0], XYZ_hand[:,1], XYZ_hand[:,2], 'ro')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    iArat.utils.set_axes_equal(ax)
    plt.show()


    PVA_hand = iArat.utils.PosVelAcc(XYZR_hand)
    PVA_wrist = iArat.utils.PosVelAcc(XYZR_wrist)
    PVA_elbow = iArat.utils.PosVelAcc(XYZR_elbow)
    PVA_shoulder = iArat.utils.PosVelAcc(XYZR_shoulder)

    #Set beginning and ending of the plots
    beg2 = 10
    end2 = len(PVA_hand)-40

    #Create time vector
    freq = 1/25
    t = np.arange(0,(end2-beg2)*freq,freq)

    #Plot the position of the hand, wrist, elbow and shoulder in a 3x4 plot
    f,ax = plt.subplots(3,4)
    ax[0,0].plot(t,PVA_hand[beg2:end2,1],'r')
    ax[0,0].set_title('Hand')
    ax[0,0].set_ylabel('y-pos')
    ax[1,0].plot(t,-PVA_hand[beg2:end2,0],'r')
    ax[1,0].set_ylabel('x-pos')
    ax[2,0].plot(t,PVA_hand[beg2:end2,2],'r')
    ax[2,0].set_ylabel('z-pos')

    ax[0,1].plot(t,PVA_wrist[beg2:end2,1],'b')
    ax[0,1].set_title('Wrist')
    ax[1,1].plot(t,-PVA_wrist[beg2:end2,0],'b')
    ax[2,1].plot(t,PVA_wrist[beg2:end2,2],'b')

    ax[0,2].plot(t,PVA_elbow[beg2:end2,1],'g')
    ax[0,2].set_title('Elbow')
    ax[1,2].plot(t,PVA_elbow[beg2:end2,0],'g')
    ax[2,2].plot(t,PVA_elbow[beg2:end2,2],'g')

    ax[0,3].plot(t,PVA_shoulder[beg2:end2,1],'c')
    ax[0,3].set_title('Shoulder')
    ax[1,3].plot(t,PVA_shoulder[beg2:end2,0],'c')
    ax[2,3].plot(t,PVA_shoulder[beg2:end2,2],'c')




    #This plots the position, velocity and acceleration of the hand only in a 3x3 plot
    # figure = plt.figure()
    # plt.subplot(334)
    # plt.plot(t,PVA_hand[beg2:end2,0])
    # plt.subplot(331)
    # plt.plot(t,PVA_hand[beg2:end2,1])
    # plt.subplot(337)
    # plt.plot(t,PVA_hand[beg2:end2,2])
    # plt.subplot(335)
    # plt.plot(t,PVA_wrist[beg2:end2,0])
    # plt.subplot(332)
    # plt.plot(t,PVA_wrist[beg2:end2,1])
    # plt.subplot(338)
    # plt.plot(t,PVA_wrist[beg2:end2,2])
    # plt.subplot(336)
    # plt.plot(t,PVA_elbow[beg2:end2,0])
    # plt.subplot(333)
    # plt.plot(t,PVA_elbow[beg2:end2,1])
    # plt.subplot(339)
    # plt.plot(t,PVA_elbow[beg2:end2,2])


    plt.show()
