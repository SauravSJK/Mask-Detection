//
// Created by Saurav Jayakumar on 5/22/23.
//

#ifndef MAIN_MASKDETECTION_H
#define MAIN_MASKDETECTION_H

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "headers/helper.h"
#include "headers/preprocessing.h"
#include "headers/facedetection.h"
#include "headers/postprocessing.h"

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// Runs the pre-processing, face detection, and post-processing steps on an image to determine whether a face in it is wearing a mask
// Parameters:
//          FILEPATH:          Path to the image
//          FACE_HAAR_CASCADE: Haar Cascade classifier object for face detection
//          FACE_LBP_CASCADE:  LBP Cascade classifier object for face detection
//          LEFT_EYE_CASCADE:  Haar Cascade classifier object for left eye detection
//          RIGHT_EYE_CASCADE: Haar Cascade classifier object for right eye detection
//          EYE_GLASS_CASCADE: Haar Cascade classifier object for eyes (with or without glasses) detection
//          DEBUG_MODE:        To control the image display outputs
// Pre-condition:  The program expects the arguments to be valid and image to the available at the specified path
// Post-condition: The counts of faces detected, masks detected, etc., are returned
vector<int> maskDetection(const string& FILEPATH, const int faces, const CascadeClassifier& FACE_HAAR_CASCADE, const CascadeClassifier& FACE_LBP_CASCADE, const CascadeClassifier& LEFT_EYE_CASCADE, const CascadeClassifier& RIGHT_EYE_CASCADE, const CascadeClassifier& EYE_GLASS_CASCADE, const bool DEBUG_MODE) {

	int images_skipped = 0;
	vector<int> results = {0, 0, 0};
	// Reading an image which might have faces from disk and displaying it
	print("Reading image from disk", DEBUG_MODE);
	const Mat IMAGE = readDisplay(FILEPATH, "Image", DEBUG_MODE);
	print(FILEPATH, true);

	// Passing the image for pre-processing and receiving all modified images in the map object
	print("Pre-processing", DEBUG_MODE);
	const Mat PRE_PROCESSED_IMAGE = preProcessing(IMAGE, DEBUG_MODE);

	// Passing the images for face detection and receiving the set of faces from the image
	print("Face detection", DEBUG_MODE);
	vector<Mat> cropped_frontal_faces = faceDetection(IMAGE, PRE_PROCESSED_IMAGE,FACE_HAAR_CASCADE, DEBUG_MODE);

	// Trying LBP cascade classifier if no faces were detected by the haar cascade classifier
	print("Trying LBP cascade classifier if no faces were detected by the haar cascade classifier", DEBUG_MODE);
	if (cropped_frontal_faces.empty()) {
		cropped_frontal_faces = faceDetection(IMAGE, PRE_PROCESSED_IMAGE,FACE_LBP_CASCADE,DEBUG_MODE);
		// Exiting if no faces were found by the LBP cascade classifier too
		if (cropped_frontal_faces.empty()) {
			print("Didn't detect any faces in the image", DEBUG_MODE);

		}
	}
	images_skipped = faces - int(cropped_frontal_faces.size());
	if (!cropped_frontal_faces.empty()) {
		// Passing the cropped face images for skin color segmentation and receiving Otsu thresholded Cr components of them
		print("Skin color segmentation", DEBUG_MODE);
		const vector<Mat> OTSU_CR_FACES = skinColorSegmentation(cropped_frontal_faces, DEBUG_MODE);

		// Passing the cropped images for eye detection and receiving the bounding boxes for the eyes
		print("Eye detection", DEBUG_MODE);
		const vector<vector<int>> EYE_NOSE_MOUTH_BOXES = eyeNoseMouthDetection(cropped_frontal_faces, LEFT_EYE_CASCADE, RIGHT_EYE_CASCADE, EYE_GLASS_CASCADE, DEBUG_MODE);

		// Passing the Otsu thresholded Cr components and the eye bounding boxes for mask detection
		print("Mask detection", DEBUG_MODE);
		results = oronasalEyeRegionComparison(OTSU_CR_FACES, EYE_NOSE_MOUTH_BOXES, DEBUG_MODE);
	}
	return {results.at(0), results.at(1), results.at(2), images_skipped};
}

#endif //MAIN_MASKDETECTION_H
