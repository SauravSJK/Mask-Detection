//
// Created by Saurav Jayakumar on 5/21/23.
//

#ifndef MAIN_POSTPROCESSING_H
#define MAIN_POSTPROCESSING_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "helper.h"

using namespace std;
using namespace cv;

// The skin color segmentation takes in a set of cropped faces, converts them to YCrCb color space, and uses the Cr component for Otsu thresholding
// Parameters: A vector of matrices with cropped face images
// Pre-condition: The vector contains valid matrices with cropped face images
// Post-condition: Images are displayed at various stages of the segmentation and the final output is returned to the caller
vector<Mat> skinColorSegmentation (const vector<Mat>& cropped_faces, const bool DEBUG_MODE) {
	// Converting cropped faces to YCrCb color space
	cout << "Converting cropped faces to YCrCb color space" << endl;
	vector<Mat> cropped_faces_ycrcb;
	for (auto &face: cropped_faces) {
		Mat face_ycrcb;
		cvtColor(face, face_ycrcb, COLOR_BGR2YCrCb);
		cropped_faces_ycrcb.push_back(face_ycrcb);
		if (DEBUG_MODE) {
			display("YCrCb Faces", face_ycrcb);
		}
	}

	// Extracting Cr component of the image
	cout << "Extracting Cr component of the image" << endl;
	vector<Mat> cr_faces;
	for (auto &face: cropped_faces_ycrcb) {
		Mat channels[3];
		split(face, channels);
		cr_faces.push_back(channels[1]);
		if (DEBUG_MODE) {
			display("Cr component of the face image", channels[1]);
		}
	}

	// Applying Otsu thresholding for skin color segmentation
	cout << "Applying Otsu thresholding for skin color segmentation" << endl;
	vector<Mat> otsu_cr_faces;
	for (auto &face: cr_faces) {
		Mat otsu;
		threshold(face, otsu, 0, 255, THRESH_OTSU);
		otsu_cr_faces.push_back(otsu);
		if (DEBUG_MODE) {
			display("Otsu Thresholding", otsu);
		}
	}

	return otsu_cr_faces;
}

// The eye detection function loads 3 eye haar cascade file and uses it to detect eyes from a face image
// Parameters: A vector of matrices with the cropped face images and location of the haar cascade xml file
// Pre-condition: The vector contains valid matrices with cropped face images and the filename should point to the correct cascade xml file
// Post-condition: The eyes detected in each of the faces are first displayed and then the coordinates of the eye area are returned
// Future improvements: Include eyes box images in the map
vector<vector<int>> eyeNoseMouthDetection (vector<Mat> cropped_faces, const String& LEFT_CASCADE_FILENAME, const String& RIGHT_CASCADE_FILENAME, const String& GLASS_CASCADE_FILENAME, const bool DEBUG_MODE) {
	// Loading the eye cascades
	cout << "Loading the eye cascades" << endl;
	CascadeClassifier left_eye_cascade;
	if(!left_eye_cascade.load(LEFT_CASCADE_FILENAME)) {
		cout << "Error loading left eye cascade\n";
		exit(0);
	}
	CascadeClassifier right_eye_cascade;
	if(!right_eye_cascade.load(RIGHT_CASCADE_FILENAME)) {
		cout << "Error loading right eye cascade\n";
		exit(0);
	}
	CascadeClassifier eye_glass_cascade;
	if(!eye_glass_cascade.load(GLASS_CASCADE_FILENAME)) {
		cout << "Error loading eye glass cascade\n";
		exit(0);
	}

	const Scalar EYE_COLOR = Scalar(255, 0, 255);
	const Scalar NOSE_MOUTH_COLOR = Scalar(0, 0, 0);
	const int THICKNESS = 1;

	vector<vector<int>> eye_nose_mouth_boxes;
	for (auto &face: cropped_faces) {
		// Detecting eyes in the image
		cout << "Detecting eyes in the image" << endl;
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

		// Eyes not detected for this face, so skipping to the next face
		if (top_left_x == 999 and top_left_y == 999 and bottom_right_x == 0 and bottom_right_y == 0) {
			cout << "Eyes not detected for this face, so skipping to the next face";
			continue;
		}

		int nose_mouth_bottom_y = min(top_left_y + 3 * (bottom_right_y - top_left_y), face.rows);
		eye_nose_mouth_boxes.push_back({top_left_x, top_left_y, bottom_right_x, bottom_right_y, nose_mouth_bottom_y});

		if (DEBUG_MODE) {
			Point pt1(top_left_x, top_left_y);
			Point pt2(bottom_right_x, bottom_right_y);
			rectangle(face, pt1, pt2, EYE_COLOR, THICKNESS);

			Point pt3(top_left_x, bottom_right_y);
			Point pt4(bottom_right_x, nose_mouth_bottom_y);
			rectangle(face, pt3, pt4, NOSE_MOUTH_COLOR, THICKNESS);

			display("Eyes, Nose, and Mouth areas detected", face);
		}
	}

	return eye_nose_mouth_boxes;
}

// The mask detection function accepts the Otsu thresholded Cr components and eye bounding boxes for mask detection
// by comparing skin areas between eye part and mouth part
// Parameters: A vector of matrices with the Otsu thresholded Cr components and vector of eye bounding boxes
// Pre-condition: The vectors contains valid data and correspond to the same face in the same order
// Post-condition: The function outputs whether the person in the image is wearing a mask or not
// Future improvements: Split the if statement
void maskDetection(vector<Mat> otsu_cr_faces, vector<vector<int>> eye_nose_mouth_boxes) {
	for (int i = 0; i < otsu_cr_faces.size(); i++) {
		int left_x = eye_nose_mouth_boxes.at(i).at(0);
		int right_x = eye_nose_mouth_boxes.at(i).at(2);
		int eye_top_y = eye_nose_mouth_boxes.at(i).at(1);
		int eye_bottom_nose_mouth_top_y = eye_nose_mouth_boxes.at(i).at(3);
		int nose_mouth_bottom_y = eye_nose_mouth_boxes.at(i).at(4);

		if (countNonZero(otsu_cr_faces.at(i)(Range(eye_top_y, eye_bottom_nose_mouth_top_y), Range(left_x, right_x))) > 1.2 * countNonZero(otsu_cr_faces.at(i)(Range(eye_bottom_nose_mouth_top_y, nose_mouth_bottom_y), Range(left_x, right_x)))) {
			cout << "Mask detected" << endl;
		}
		else {
			cout << "Mask not detected" << endl;
		}
	}
}

#endif //MAIN_POSTPROCESSING_H
