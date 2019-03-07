import os
import csv
import cv2
import sys
import matplotlib.pyplot as plt


'''
This function uses the CSV data from deeplabcut to create an image that shows the tracked positions of a certain amount of frames
Input the CSV data location and the location of the corresponding video. Then choose the length of the tracking and which frame.

If the tracking is off, check the res (is hard coded). If the DLC video is scaled before training, the CVS data is also scaled as it is in pixels.
When then drawing the data back on the unscaled video gives wrong results. 


'''


def line_tracked_image(CSV_data, Tracked_vid,working_dir, track_length, tracked_frame):
	# Read the CSV file
	with open(CSV_data) as data:
		reader = csv.reader(data)
		data_list = list(reader)

	#Create a list of the CSV File
	data_T = list(map(list,zip(*data_list)))

	# The CVS data is in pixels from a 1440:810 video. When tracking 
	# the 1920:1080 image, the data should be scaled:
	res = 1080/810

	#Creating floats of the CSV data
	frames    = [float(i) for i in data_T[0][3:]]
	x_hand    = [float(i)*res for i in data_T[1][3:]]
	y_hand    = [float(i)*res for i in data_T[2][3:]]
	llh_hand  = [float(i) for i in data_T[3][3:]]
	x_wrist   = [float(i)*res for i in data_T[4][3:]]
	y_wrist   = [float(i)*res for i in data_T[5][3:]]
	llh_wrist = [float(i) for i in data_T[6][3:]]
	x_elbow   = [float(i)*res for i in data_T[7][3:]]
	y_elbow   = [float(i)*res for i in data_T[8][3:]]
	llh_elbow = [float(i) for i in data_T[9][3:]]
	x_shoulder= [float(i)*res for i in data_T[10][3:]]
	y_shoulder= [float(i)*res for i in data_T[11][3:]]
	llh_shoulder=[float(i) for i in data_T[12][3:]]


	vidcap = cv2.VideoCapture(Tracked_vid)
	length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("Length of the video has "+str(length)+" frames!")
	if tracked_frame > length:
		print("Choose a number under the %d for extraction!" % length)
		#Extracting frames from the video to create the tracking lines:
	else:
		Extract = os.path.isfile("frame%d.jpg"% tracked_frame)
		if Extract == False:
			print('Extracting the frame')
			vidcap = cv2.VideoCapture(Tracked_vid)
			success,image = vidcap.read()
			count = 0
			while success:
				if count == tracked_frame:
					cv2.imwrite(working_dir+"/frame%d.jpg" % count, image)     # save frame as JPEG file      
					print('Frame extracted!')
				success,image = vidcap.read()
				count += 1




	#Reading image:
	img1 = cv2.imread("frame%d.jpg" % tracked_frame)
	RGB_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

	#Plotting the tracking lines:
	plt.imshow(RGB_img)
	if tracked_frame < 2:
		print("Take a frame that is later than the 2nd one")
	if tracked_frame < track_length+2:
		plt.plot(x_hand[3:tracked_frame], y_hand[3:tracked_frame],'r',
			x_wrist[3:tracked_frame],y_wrist[3:tracked_frame],'b',
			x_elbow[3:tracked_frame],y_elbow[3:tracked_frame],'g',
			x_shoulder[3:tracked_frame],y_shoulder[3:tracked_frame],'c')
	else:
		plt.plot(x_hand[tracked_frame-track_length:tracked_frame+1], y_hand[tracked_frame-track_length:tracked_frame+1],'r',
			x_wrist[tracked_frame-track_length:tracked_frame+1],y_wrist[tracked_frame-track_length:tracked_frame+1],'b',
			x_elbow[tracked_frame-track_length:tracked_frame+1],y_elbow[tracked_frame-track_length:tracked_frame+1],'g',
			x_shoulder[tracked_frame-track_length:tracked_frame+1],y_shoulder[tracked_frame-track_length:tracked_frame+1],'c')

	plt.plot(x_hand[tracked_frame+1],y_hand[tracked_frame-1],'or',
		x_wrist[tracked_frame+1],y_wrist[tracked_frame-1],'ob',
		x_elbow[tracked_frame+1],y_elbow[tracked_frame-1],'og',
		x_shoulder[tracked_frame+1],y_shoulder[tracked_frame-1],'oc',)
	plt.show()