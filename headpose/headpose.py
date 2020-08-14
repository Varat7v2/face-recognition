## HEADPOSE DETECTION FROM QHAN
import cv2
import dlib
import numpy as np

import argparse
from headpose.headpose_utils import Annotator

class HeadPoseEstimation():
	#3D facial model coordinates
	landmarks_3d_list = [
		np.array([
			[ 0.000,  0.000,   0.000],    # Nose tip
            [ 0.000, -8.250,  -1.625],    # Chin
            [-5.625,  4.250,  -3.375],    # Left eye left corner
            [ 5.625,  4.250,  -3.375],    # Right eye right corner
            [-3.750, -3.750,  -3.125],    # Left Mouth corner
            [ 3.750, -3.750,  -3.125]     # Right mouth corner 
		], dtype=np.double),
		np.array([
			[ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
            [ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
            [-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
            [-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
            [ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
            [-2.005628,  1.409845,  6.165652],   # 49 nose right corner
            [ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
            [-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
            [ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
            [ 0.000000, -7.415691,  4.070434]    # 6 chin corner
		], dtype=np.double),
		np.array([
			[ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
            [ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
            [ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
            [-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
            [-5.311432,  5.485328,  3.987654]    # 21 right eye right corner
		], dtype=np.double)
	]

	#2D facial landmark list
	landmarks_2d_list = [
		[30, 8, 36, 45, 48, 54], # 6 points
        [33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8], # 14 points
        [33, 36, 39, 42, 45] # 5 points
	]

	def __init__(self, lm_type=1):
		self.landmark_2D_index = self.landmarks_2d_list[lm_type]
		self.landmarks_3D = self.landmarks_3d_list[lm_type]
		self.detector = dlib.get_frontal_face_detector()
		self.predictor="models/shape_predictor_68_face_landmarks.dat"
		self.landmark_predictor = dlib.shape_predictor(self.predictor)
	
	def to_numpy(self, landmarks):
		coordinates = []
		for i in self.landmark_2D_index:
			coordinates += [[landmarks.part(i).x, landmarks.part(i).y]]
		#print(np.array(coordinates).astype(np.int))
		return np.array(coordinates).astype(np.int)

	def get_headPose(self, image, landmarks_2D):
		height, width, channel = image.shape
		f = width #focal length
		cx, cy = width/2, height/2 #center of image plane
		camera_matrix = np.array(
			[[f, 0, cx],
			 [0, f, cy],
			 [0, 0, 1]], dtype=np.double
		)

		#assuming no lens distortion
		dist_coeffs = np.zeros((4,1))

		#Find rotation and translation
		(success, rotation_vector, translation_vector) = cv2.solvePnP(objectPoints = self.landmarks_3D, 
			imagePoints = landmarks_2D, cameraMatrix = camera_matrix, distCoeffs = dist_coeffs)

		return rotation_vector, translation_vector, camera_matrix, dist_coeffs

	# rotation vector to euler angles
	def get_angles(self, rotation_vec, translation_vec):
		rotation_mat = cv2.Rodrigues(rotation_vec)[0]
		projection_mat = np.hstack((rotation_mat, translation_vec)) #projection matrix [R | T]	
		degrees = cv2.decomposeProjectionMatrix(projection_mat)[6]
		rx, ry, rz = degrees[:,0]
		return [rx, ry, rz]

	def get_landmarks(self, image):
		rects = self.detector(image, 0) if image is not None else []

		if len(rects) > 0:
			#detect landmarks of first face
			landmarks_2D = self.landmark_predictor(image, rects[0])

			#choose specific landmarks corresponding to 3D facial model
			landmarks_2D = self.to_numpy(landmarks_2D)

			rect = [rects[0].left(), rects[0].top(), rects[0].right(), rects[0].bottom()]

			return landmarks_2D.astype(np.double), rect
		else:
			return None, None

	# return image and angles
	def findHeadPose(self, image, landmarks_2d, bbox, draw=True):
		#landmark detection
		# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# landmarks_2d, bbox = self.get_landmarks(image_gray) 
		# if no face detected, return original image
		# if landmarks_2d is None:
		# 	return image, None

		#choose specific landmarks corresponding to 3D facial model
		landmarks_2D = self.to_numpy(landmarks_2d).astype(np.double)

		# Headpose Detection
		rvec, tvec, cm, dc = self.get_headPose(image, landmarks_2D)

		angles = self.get_angles(rvec, tvec)

		if draw:
			annotator = Annotator(image, angles, bbox, landmarks_2D, rvec, tvec, cm, dc, b=10.0)
			image = annotator.draw_all()

		return image, angles