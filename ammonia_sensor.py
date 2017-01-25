#!/usr/bin/env python

import numpy as np
import argparse
import sys
import time
from scipy import interpolate
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import cv2
import os

os.system('uvcdynctrl -L ~/ammonia_sensor/ammonia_camera_config.gpfl')

now = datetime.now()
filename = now.strftime("%Y-%m-%d-%H-%M-%S")

camera_port = 0

ramp_frames = 30

camera = cv2.VideoCapture(camera_port)

def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im
 

# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
for i in xrange(ramp_frames):
 temp = get_image()
print("Taking image...")
# Take the actual image we want to keep
camera_capture = get_image()

file = "~/ammonia_sensor/images/" + filename + ".png"
# A nice feature of the imwrite method is that it will automatically choose the
# correct format based on the file extension you provide. Convenient!
cv2.imwrite(file, camera_capture)

 
# You'll want to release the camera, otherwise you won't be able to create a new
# capture object until your script exits
del(camera)

user_name = os.popen('whoami').read().strip('\n')
results = open("/home/"+str(user_name)+"/ammonia_sensor/csv_files/" + filename + ".csv", 'wb')
write = csv.writer(results, dialect='excel')
header = "Average BGR Values"
write.writerow([header])
write.writerow(['Cage Number','Sample','B','G','R'])

print '\n'
print 'Output Filename: ' + filename + '.csv'
print '\n'

refPt = []
cropping = False
targets = []
"""

Sample of how to define coordinates for autonomous mode:
		***Order matters, so enter your values appropriately, or figure out how it works
  Enter Upper-Left (UL) and Lower-Right (LR) Coordinates of bounding box for sample area as follows -->>
  for 0ppm...
# 0ppm
targets.append((URx,URy))
targets.append((LRx,LRy))
----or with real values----
targets.append((438,187))
targets.append((467,204))

"""
# 0ppm
targets.append((540,95))
targets.append((600,125))
# 5ppm
targets.append((540,145))
targets.append((600,175))
# 10ppm
targets.append((540,192))
targets.append((600,222))
# 20ppm
targets.append((540,240))
targets.append((600,270))
# 50ppm
targets.append((540,290))
targets.append((600,320))
# 100+ ppm
targets.append((540,340))
targets.append((600,370))
# Unknown ppm
targets.append((420,190))
targets.append((450,270))


bins = np.arange(256).reshape(256,1)

"""

	Stitched together by mike with some modifications from a few different sources and sample programs

"""
"""

print '''usage : python find_RGB.py --i <image_file>\n
Select a Region of Interest (ROI) by drawing a box around it using the mouse \n
	- start in upper left corner, click, drag, and release on lower right
	  * only one ROI can be selected at a time
	  	>(if you mess up just press 'r') \n
	- select which mode you would like from the keymap \n
	- specify the sample area name (only for average value mode)\n
	- rinse, repeat
	'''
"""

def print_keymap():

	print '''
	Keymap :\n
	    1 - Take sample for cage 1 \n
	    2 - Take sample for cage 2 \n
	    3 - Take sample for cage 3 \n
	    q - quit \n
	    '''

print_keymap()

"""

Below is a sample method for finding the BGR values autonomously. Enter in the coordinates of each sample area in the order correspo


"""

def pick_a_number():

	global selection

	while True:
		try:
			selection = int(input('Sample: '))
			break
		except:
			print("Oops! try entering an integer")
			continue
	return selection

def sample_select():
	
	print "Please specify your Sample Area"
	print '''
Sample Selections :\n
    1 - 0ppm \n
    2 - 5ppm \n
    3 - 10ppm\n
    4 - 20ppm \n
    5 - 50ppm \n
    6 - 100ppm (really 100+) \n
    7 - unknown \n
    8 - custom \n
    '''
	sample = list()
	sample = ['0','5','10','20','50','100', 'unk']

	pick_a_number()

	if selection <= 7 and selection > 0:
		print sample[int(selection)-1]
		dataPts.append(sample[int(selection)-1])
	elif selection == 8:
		custom = raw_input('Specify custom Sample Name: ')
		dataPts.append(custom)
		print dataPts
	else:
		print "You have entered an invalid input"
	return dataPts

def write_funct():
	print "Data to be sent >> "
	print dataPts
	write.writerow(dataPts)

def average_funct(im):
	h = np.zeros((300,256,3))
	if len(im.shape) == 2:
		color = [(255,255,255)]
	elif im.shape[2] == 3:
		color = [ (255,0,0),(0,255,0),(0,0,255) ]
	print "Average BGR Values:"
	for ch, col in enumerate(color):
		hist_item = cv2.calcHist([im],[ch],None,[256],[0,256])
		cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
		hist=np.int32(np.around(hist_item))
		pts = np.int32(np.column_stack((bins,hist)))
		cv2.polylines(h,[pts],False,col)
		name = ['B','G','R']
		val = 0
		total = 0
		i = 0
		for x in hist:
			x = int(x)
			val += x * i
			#print val
			#print "i: " + str(i) + " x: " + str(x)
			total += x
			i += 1
		avg = val/total
		
		"""
		# This was the previous meat and potatoes of the average_funct, but I realized it doesn't work
		val=list()
		data=list()
		for x in hist:
			val.append(x)
			for x in val:
				# changing the value of var allows for thresholding
				# it can be set from <0-255>
				# 255 means consider only the most frequently occuring value for the RGB channels
				#	----> which is essentieally what the peak value function does
				# while 0 means consider any non-zero RGB values in the calculation of the average
				# ...do with this information what you will 
				var = 0
				if x > var:
					data.append(val.index(x))
		avg = sum(data)/len(data)
		"""
		print name[ch]
		print avg
		if ch < 2:
			print "--->"
		dataPts.append(avg)
	y=np.flipud(h)
	return y

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		# Uncomment the following to prient (x,y) coordinates of target area
		#print "Upper Left: " + str(refPt[0])
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		# Uncomment the following to print (x,y) coordinates of target area
		#print "Lower Right: " + str(refPt[1])
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
if not args["image"]:
	image1 = cv2.imread(file)
	image = cv2.resize(image1, (666, 487))  
	clone = image.copy()
	cv2.namedWindow("image")
	#cv2.setMouseCallback("image", click_and_crop)
else:
	image1 = cv2.imread(args["image"])
	image = cv2.resize(image1, (666, 487))  
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)

while True:

	dataPts = list()

	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# Added by mike to test the autonomous ROI selector
	elif key == ord("1"):

		sample = ['0','5','10','20','50','100', 'unk']			
		for i in range((len(targets)/2)):
			dataPts = list()
			del refPt[:]
			ul = (i * 2)
			lr = (ul + 1)
			refPt.append(targets[ul])
			print targets[ul]
			refPt.append(targets[lr])
			cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.imshow("image", image)
			if len(refPt) == 2:
				roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
				#sample_select()
				curve = average_funct(roi)
				cv2.imshow('histogram', curve)
				cv2.imshow("ROI", roi)
				dataPts.insert(0, '1')
				dataPts.insert(1, sample[i])
				write_funct()
				"""
				del dataPts[:]
				curve = peak_funct(roi)
				cv2.imshow('histogram', curve)
				cv2.imshow("ROI", roi)
				dataPts.insert(0, 'Peak Value')
				dataPts.insert(1, sample[i])
				write_funct()
				#print dataPts
				"""
		print_keymap()
		cv2.waitKey(0)
	
	elif key == ord("2"):

		sample = ['0','5','10','20','50','100', 'unk']			
		for i in range((len(targets)/2)):
			dataPts = list()
			del refPt[:]
			ul = (i * 2)
			lr = (ul + 1)
			refPt.append(targets[ul])
			print targets[ul]
			refPt.append(targets[lr])
			cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.imshow("image", image)
			if len(refPt) == 2:
				roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
				#sample_select()
				curve = average_funct(roi)
				cv2.imshow('histogram', curve)
				cv2.imshow("ROI", roi)
				dataPts.insert(0, '2')
				dataPts.insert(1, sample[i])
				write_funct()

		print_keymap()
		cv2.waitKey(0)

	elif key == ord("3"):

		sample = ['0','5','10','20','50','100', 'unk']			
		for i in range((len(targets)/2)):
			dataPts = list()
			del refPt[:]
			ul = (i * 2)
			lr = (ul + 1)
			refPt.append(targets[ul])
			print targets[ul]
			refPt.append(targets[lr])
			cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
			cv2.imshow("image", image)
			if len(refPt) == 2:
				roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
				#sample_select()
				curve = average_funct(roi)
				cv2.imshow('histogram', curve)
				cv2.imshow("ROI", roi)
				dataPts.insert(0, '3')
				dataPts.insert(1, sample[i])
				write_funct()

		print_keymap()
		cv2.waitKey(0)


	elif key == ord("q"):
		break

# if there are two reference points, then crop the region of interest
# from the image and display it
"""
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	curve = hist_curve(roi)
	cv2.imshow('histogram', curve)
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)e
"""
 
# close all open windows
cv2.destroyAllWindows()
results.close()

#_______________________________________________
#	
#			PPM CONVERSION PART BELOW
#_______________________________________________


known_ppms = [0,5,10,20,50,100]
hue_values = [None]*6
unknown_hue = 0.0

with open("/home/" + str(user_name)+ "/ammonia_sensor/csv_files/" + filename + ".csv") as f:

	lis= f.read().splitlines()

	xy = [None]*len(lis)
	for i in lis:
		xy[lis.index(i)] = i.split(',')

	data = xy[2:]

	#print data
	cage_number = xy[2][0]
	#print cage_number
			#rate.sleep()

	i = 0

	for data_set in data:
		data_set = data_set[2:]
		values = list(map(float,data_set))
		#print values

		b_prime = values[0]/255
		g_prime = values[1]/255
		r_prime = values[2]/255

		Cmax = max(b_prime,g_prime,r_prime)
		Cmin = min(b_prime,g_prime,r_prime)
		delta = Cmax - Cmin

		if (delta == 0):
			hue = 0
		elif (Cmax == r_prime):
			hue = 60 * (((g_prime-b_prime)/delta)%6)
		elif (Cmax == g_prime):
			hue = 60 * (((b_prime-r_prime)/delta) + 2)
		else:
			hue = 60 * (((r_prime-g_prime)/delta) + 4)
		
		if (i < 6):
			hue_values[i] = hue
		elif (i == 6):
			unknown_hue = hue
		i += 1

	#print hue_values
	#print unknown_hue

	ppm_eq = interpolate.interp1d(hue_values,known_ppms,kind='cubic')

	unknown_ppm = ppm_eq(unknown_hue)

	examples = np.arange(min(hue_values),max(hue_values),1)
	example_points = ppm_eq(examples)

	#print unknown_ppm

	plt.plot(known_ppms,hue_values,'o',unknown_ppm,unknown_hue,'r^',example_points,examples,'g-')

	plt.show()



	unknown_ppm2 = np.interp(unknown_hue,hue_values,known_ppms)
	print "The concentration of ammonia in cage number " + str(cage_number) + " is: " + str(unknown_ppm) + " ppm"
	f.close()
	#print "The linear fit value is " + str(unknown_ppm2) + " ppm"
log = open("/home/" +str(user_name)+"/ammonia_sensor/concentration_log.csv", 'a')
write_new = csv.writer(log, dialect='excel')
write_new.writerow([filename,cage_number,unknown_ppm])

#command = 'python bgr_to_ppm.py ' + filename + '.csv'
#print command
#os.system(command)
