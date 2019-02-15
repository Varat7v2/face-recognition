#capturing images for training for recognizers
#stores them in data folder

import cv2, sys, os
import numpy as np


#all the images will be in datasets folder
datasets = 'images'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
cascade_dir = os.path.join(BASE_DIR, "data")
face_cascade = cv2.CascadeClassifier(cascade_dir + "/"+"haarcascade_frontalface_alt2.xml")

webcam = cv2.VideoCapture(0)

### FUNCTION: FOR FETCHING THE IMAGES AND TRAINING THE MODEL
def capture_train():
	#sub-folder for storing specific images
	sub_folder=raw_input("Enter your name: ")
	path = os.path.join(datasets, sub_folder)
	if not os.path.isdir(path):
		os.mkdir(path)

	#defining sizes of images
	(width, height) = (130, 100)


	#the program loops unitil it has caputred 30 images
	count = 1
	print('Taking images for training...')
	while(count<100):
		(_, img) = webcam.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
		
		for(x,y,w,h) in faces:
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
			cv2.putText(img, 'Capturing Photo: Please rotate your face slightly', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
			face = gray[y:y+h, x:x+w]
			face_resize = cv2.resize(face, (width, height))
			cv2.imwrite('%s/% s.png' %(path, count), face_resize)
		count += 1

		cv2.imshow('OpenCV', img)
		key = cv2.waitKey(10)
		if key == 27:
			break

### FUNCTION: FOR RECOGNIGING THE FACE
def face_recognizer():
	# Part 1: Create fisherRecognizer 
	print('Recognizing Face Please Be in sufficient Lights...')
	# Create a list of images and a list of corresponding names 
	(images, lables, names, id_) = ([], [], {}, 0) 
	for (subdirs, dirs, files) in os.walk(datasets): 
	    for subdir in dirs: 
	        names[id_] = subdir 
	        subjectpath = os.path.join(datasets, subdir) 
	        for filename in os.listdir(subjectpath): 
	            path = subjectpath + '/' + filename 
	            # print(path)
	            lable = id_
	            images.append(cv2.imread(path, 0)) 
	            lables.append(int(lable)) 
	        id_ += 1
	(width, height) = (130, 100) 
	  
	# Create a Numpy array from the two lists above 
	(images, lables) = [np.array(lis) for lis in [images, lables]] 
	  
	# OpenCV trains a model from the images 
	# NOTE FOR OpenCV2: remove '.face' 
	model = cv2.face.LBPHFaceRecognizer_create() 
	model.train(images, lables) 
	  
	while True: 
	    (_, img) = webcam.read() 
	    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) 
	    for (x, y, w, h) in faces: 
	        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 
	        face = gray[y:y + h, x:x + w] 
	        face_resize = cv2.resize(face, (width, height)) 
	        # Try to recognize the face 
	        prediction = model.predict(face_resize) 
	        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3) 
	  
	        if prediction[1]<500: 
	        	cv2.putText(img, '% s - %.0f' % (names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
	        else:
	        	cv2.putText(img, 'not recognized', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
	  
	    cv2.imshow('OpenCV', img)
	    if cv2.waitKey(20) & 0xFF == ord('q'):
	    	break

	# When everything done, release the capture
	webcam.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	case = raw_input("Do you want to capture your photo? y or n --> ")
	print('Please press"q" to close the webcam')
	if case == 'y':
		capture_train()
	face_recognizer()