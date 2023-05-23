//
// Created by Saurav Jayakumar on 5/21/23.
//

#ifndef MAIN_POSTPROCESSING_H
#define MAIN_POSTPROCESSING_H

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "headers/helper.h"

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// The skin color segmentation takes in a set of cropped faces, converts them to YCrCb color space, and uses the Cr component for Otsu thresholding
// Parameters:
//          CROPPED_FACES: A vector of matrices with cropped face images
//          DEBUG_MODE:    To control the image display outputs
// Pre-condition: The vector contains valid matrices with cropped face images
// Post-condition: Images are displayed at various stages of the segmentation if running in debug mode and then the final output is returned to the caller
vector<Mat> skinColorSegmentation (const vector<Mat>& CROPPED_FACES, const bool DEBUG_MODE) {
	// Converting cropped faces to YCrCb color space
	print("Converting cropped faces to YCrCb color space", DEBUG_MODE);
	vector<Mat> cropped_faces_ycrcb;
	for (auto &face: CROPPED_FACES) {
		Mat face_ycrcb;
		cvtColor(face, face_ycrcb, COLOR_BGR2YCrCb);
		cropped_faces_ycrcb.push_back(face_ycrcb);
		display("YCrCb Faces", face_ycrcb, DEBUG_MODE);
	}

	// Extracting Cr component of the image
	print("Extracting Cr component of the image", DEBUG_MODE);
	vector<Mat> cr_faces;
	for (auto &face: cropped_faces_ycrcb) {
		Mat channels[3];
		split(face, channels);
		cr_faces.push_back(channels[1]);
		display("Cr component of the face image", channels[1], DEBUG_MODE);
	}

	// Applying Otsu thresholding for skin color segmentation
	print("Applying Otsu thresholding for skin color segmentation", DEBUG_MODE);
	vector<Mat> otsu_cr_faces;
	for (auto &face: cr_faces) {
		Mat otsu;
		threshold(face, otsu, 0, 255, THRESH_OTSU);
		otsu_cr_faces.push_back(otsu);
		display("Otsu Thresholding", otsu, DEBUG_MODE);
	}

	return otsu_cr_faces;
}

// The detection function loads 3 eye haar cascade file and uses it to detect eyes from a face image
// This is then used to determine the bounding boxes for the eye region and oronasal region which is returned to the caller
// Parameters:
//          cropped_faces:     A vector of matrices with the cropped face images and location of the haar cascade xml file
//          left_eye_cascade:  Haar Cascade classifier object for left eye detection
//          right_eye_cascade: Haar Cascade classifier object for right eye detection
//          eye_glass_cascade: Haar Cascade classifier object for eyes (with or without glasses) detection
//          DEBUG_MODE:        To control the image display outputs
// Pre-condition: The vector contains valid matrices with cropped face images and the cascade objects should be valid
// Post-condition: The eye and oronsasal regions are first displayed if running in debug mode and then the coordinates of the bounding boxes are returned
vector<vector<int>> eyeNoseMouthDetection (vector<Mat> cropped_faces, CascadeClassifier left_eye_cascade, CascadeClassifier right_eye_cascade, CascadeClassifier eye_glass_cascade, const bool DEBUG_MODE) {

	const Scalar EYE_COLOR = Scalar(255, 0, 255);
	const Scalar NOSE_MOUTH_COLOR = Scalar(0, 0, 0);
	const int THICKNESS = 1;

	vector<vector<int>> eye_nose_mouth_boxes;
	for (auto &face: cropped_faces) {
		// Detecting eyes in the image
		print("Detecting eyes in the image", DEBUG_MODE);
		vector<Rect> eyes;
		int top_left_x = 999, top_left_y = 999, bottom_right_x = 0, bottom_right_y = 0;
		left_eye_cascade.detectMultiScale(face, eyes);
		for (auto & eye : eyes) {
			top_left_x = min(top_left_x, eye.x);
			top_left_y = min(top_left_y, eye.y);
			bottom_right_x = max(bottom_right_x, eye.x + eye.width);
			bottom_right_y = max(bottom_right_y, eye.y + eye.height);
		}
		right_eye_cascade.detectMultiScale(face, eyes);
		for (auto & eye : eyes) {
			top_left_x = min(top_left_x, eye.x);
			top_left_y = min(top_left_y, eye.y);
			bottom_right_x = max(bottom_right_x, eye.x + eye.width);
			bottom_right_y = max(bottom_right_y, eye.y + eye.height);
		}
		eye_glass_cascade.detectMultiScale(face, eyes);
		for (auto & eye : eyes) {
			top_left_x = min(top_left_x, eye.x);
			top_left_y = min(top_left_y, eye.y);
			bottom_right_x = max(bottom_right_x, eye.x + eye.width);
			bottom_right_y = max(bottom_right_y, eye.y + eye.height);
		}

		int nose_mouth_bottom_y = min(top_left_y + 3 * (bottom_right_y - top_left_y), face.rows);
		eye_nose_mouth_boxes.push_back({top_left_x, top_left_y, bottom_right_x, bottom_right_y, nose_mouth_bottom_y});

		if (DEBUG_MODE) {
			// Eyes not detected for this face, so skipping to the next face
			if (top_left_x == 999 && top_left_y == 999 && bottom_right_x == 0 && bottom_right_y == 0) {
				print("Eyes not detected for this face, so skipping to the next face", DEBUG_MODE);
				continue;
			}
			Point pt1(top_left_x, top_left_y);
			Point pt2(bottom_right_x, bottom_right_y);
			rectangle(face, pt1, pt2, EYE_COLOR, THICKNESS);

			Point pt3(top_left_x, bottom_right_y);
			Point pt4(bottom_right_x, nose_mouth_bottom_y);
			rectangle(face, pt3, pt4, NOSE_MOUTH_COLOR, THICKNESS);

			display("Eyes, Nose, and Mouth areas detected", face, DEBUG_MODE);
		}
	}

	return eye_nose_mouth_boxes;
}

// The mask detection function accepts the Otsu thresholded Cr components and eye bounding boxes for mask detection
// by comparing skin areas between eye region and oronasal region
// Parameters:
//          otsu_cr_faces:        A vector of matrices with the Otsu thresholded Cr components and vector of eye bounding boxes
//          eye_nose_mouth_boxes: Coordinates of the eye and oronasal regions
//          DEBUG_MODE:           To control the image display outputs
// Pre-condition: The vectors contains valid data and correspond to the same face in the same order
// Post-condition: The function returns the number of faces wearing a mask
vector<int>  oronasalEyeRegionComparison(vector<Mat> otsu_cr_faces, vector<vector<int>> eye_nose_mouth_boxes, const bool DEBUG_MODE) {
	// Variables to track the number of faces and masks detected
	int masks_detected = 0;
	int masks_not_detected = 0;
	int faces_skipped = 0;

	for (int i = 0; i < otsu_cr_faces.size(); i++) {
		int left_x = eye_nose_mouth_boxes.at(i).at(0);
		int right_x = eye_nose_mouth_boxes.at(i).at(2);
		int eye_top_y = eye_nose_mouth_boxes.at(i).at(1);
		int eye_bottom_nose_mouth_top_y = eye_nose_mouth_boxes.at(i).at(3);
		int nose_mouth_bottom_y = eye_nose_mouth_boxes.at(i).at(4);

		// Eyes not detected for this face, so skipping to the next face
		if (left_x == 999 && eye_top_y == 999 && right_x == 0 && eye_bottom_nose_mouth_top_y == 0) {
			print("Eyes not detected for this face, so skipping to the next face", DEBUG_MODE);
			faces_skipped += 1;
		}

		else if (countNonZero(otsu_cr_faces.at(i)(Range(eye_top_y, eye_bottom_nose_mouth_top_y), Range(left_x, right_x))) > 1.2 * countNonZero(otsu_cr_faces.at(i)(Range(eye_bottom_nose_mouth_top_y, nose_mouth_bottom_y), Range(left_x, right_x)))) {
			print("Mask detected", DEBUG_MODE);
			masks_detected += 1;
		}

		else {
			print("Mask not detected", DEBUG_MODE);
			masks_not_detected += 1;
		}
	}
	return {faces_skipped, masks_detected, masks_not_detected};
}

#endif //MAIN_POSTPROCESSING_H
