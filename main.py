import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image, show_image
import os
from PIL import Image
from mss import mss
import pyscreenshot as ImageGrab

FLAGS = []

class person_detected:
	def __init__(self):
		self.count=12
		self.actual_position=None
		pass
	def change_BBox(self,BBox):
		if self.count>0:
			self.BBox=BBox
			#print("changed to")
			#print(self.BBox)

	def check(self,width):
		#print("count= "+str(self.count))
		x=self.BBox[0]
		w=self.BBox[2]
		#print("-----------")
		#print(x)
		#print(w)
		#print(width/2)
		print("-----------")
		if(x+w/2>(width/2)):
			self.actual_position="RIGHT"
			print("right")
		else:
			self.actual_position="LEFT"
			print("left")
	def pass_one_frame(self):
		self.count -= 1
		if self.count<0:
			print("THE PERSON LEAVE THE VIEW, HE WENT TO "+self.actual_position+" SIDE")
			self.count=0

		#print("-1")
	def restart_frame(self):
		self.count=6

if __name__ == '__main__':
	parser = argparse.ArgumentParser()


	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed

	cwd = os.getcwd()
	# Get the labels
	labels = open(cwd+"\\coco-labels").read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model

	net = cv.dnn.readNetFromDarknet(cwd+"\\tiny.cfg", cwd+"\\yolov3-tiny.weights")

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	# If both image and video files are given then raise error
	if FLAGS.image_path is None and FLAGS.video_path is None:
		print ('Neither path to an image or path to video provided')
		print ('Starting The Desktop scan')

	# Do inference with given image
	if FLAGS.image_path:
		# Read the image
		img = cv.imread(cwd+"\\"+FLAGS.image_path)
		height, width = img.shape[:2]
		img, a,v, c, d = infer_image(net, layer_names, height, width, img, colors, labels, FLAGS)
		show_image(img)

	elif FLAGS.video_path:
		# Read the video

		try:
			person = None
			vid = cv.VideoCapture(cwd+"\\"+FLAGS.image_path)
			height, width = None, None
			writer = None
		except:
			raise ('Video cannot be loaded!\n\Please check the path provided!')
		finally:
			while True:
				grabbed, frame = vid.read()
				# Checking if the complete video is read
				if not grabbed:
					break

				if width is None or height is None:
					height, width = frame.shape[:2]

				frame, a, v, c, d = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)
				cv.imshow("frame", frame)

				key = cv.waitKey(1) & 0xFF
				vec = [a, v, c, d]
				count=0
				okPerson=0
				for pers in c:
					if pers==0:
						okPerson=1
						if person==None:
							person=person_detected()
						person.change_BBox(a[count])
						#person.check(width)
					count += 1
				if person!=None:
					person.check(width)
					if okPerson==0:
						person.pass_one_frame()
					else:
						person.restart_frame()
				#for vecc in vec:
				#	if len(vecc) > 0:
				#		if vecc[2][0] == 0:
				#			# its a person
				#			person.change_BBox(vecc[0])
				#			print(vecc[0])

				#if writer is None:
				#	# Initialize the video writer
				#	fourcc = cv.VideoWriter_fourcc(*"MJPG")
				#	writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
				#		            (frame.shape[1], frame.shape[0]), True)
#
#
				#writer.write(frame)

			print ("[INFO] Cleaning up...")
			#writer.release()
			vid.release()

	else:
		#mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
		person = None

		height, width = None, None
		#writer = None
		sct = mss()
		while (True):
			img = ImageGrab.grab(bbox=(100, 100, 500, 700))  # x, y, w, h #TODO
			img_np = np.array(img)

			RGB_img = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
			if width is None or height is None:
				height, width = RGB_img.shape[:2]

			frame, a, v, c, d = infer_image(net, layer_names, height, width, RGB_img, colors, labels, FLAGS)
			cv.imshow("frame", RGB_img)

			key = cv.waitKey(1) & 0xFF
			vec = [a, v, c, d]
			count = 0
			okPerson = 0
			for pers in c:
				if pers == 0:
					okPerson = 1
					if person == None:
						person = person_detected()
					person.change_BBox(a[count])
				# person.check(width)
				count += 1
			if person != None:
				person.check(width)
				if okPerson == 0:
					person.pass_one_frame()
				else:
					person.restart_frame()
			# for vecc in vec:
			#	if len(vecc) > 0:
			#		if vecc[2][0] == 0:
			#			# its a person
			#			person.change_BBox(vecc[0])
			#			print(vecc[0])

			#if writer is None:
			#	# Initialize the video writer
			#	fourcc = cv.VideoWriter_fourcc(*"MJPG")
			#	writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,
			#							(frame.shape[1], frame.shape[0]), True)
#
			#writer.write(frame)
			cv.imshow("frame", RGB_img)
			key = cv.waitKey(1)
			#if key == 27:
			#	break
	#else:
	#	# Infer real-time on webcam
	#	count = 0
#
	#	vid = cv.VideoCapture(0)
	#	while True:
	#		_, frame = vid.read()
	#		height, width = frame.shape[:2]
#
	#		if count == 0:
	#			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
	#	    						height, width, frame, colors, labels, FLAGS)
	#			count += 1
	#		else:
	#			frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
	#	    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
	#			count = (count + 1) % 6
#
	#		cv.imshow('webcam', frame)
#
	#		if cv.waitKey(1) & 0xFF == ord('q'):
	#			break
	#	vid.release()
	#	cv.destroyAllWindows()
#