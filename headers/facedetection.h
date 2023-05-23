//
// Created by Saurav Jayakumar on 5/21/23.
//

#ifndef MAIN_FACEDETECTION_H
#define MAIN_FACEDETECTION_H

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include "headers/helper.h"

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// The face detection function uses a face cascade classifier to detect faces from an image
// Parameters:
//          IMAGE:               The original image used for mask detection
//          PRE_PROCESSED_IMAGE: The pre-processed image
//          face_cascade:        Cascade classifier object for face detection
//          DEBUG_MODE:          To control the image display outputs
// Pre-condition: The images and cascade classifier objects should be valid
// Post-condition: The faces detected in the image are first displayed if running in debug mode and then returned to the caller function as a vector of matrices
vector<Mat> faceDetection (const Mat& IMAGE, const Mat& PRE_PROCESSED_IMAGE, CascadeClassifier face_cascade, const bool DEBUG_MODE) {

	// Detecting faces in the image
	print("Detecting faces in the image", DEBUG_MODE);
	vector<Rect> faces;
	vector<Mat> cropped_faces;
	const Scalar COLOR = Scalar(255, 0, 255);
	const int THICKNESS = 1;
	face_cascade.detectMultiScale(PRE_PROCESSED_IMAGE, faces);

	for (auto & i : faces) {
		Point pt1(i.x - 1, i.y - 1);
		Point pt2(i.x + i.width + 1, i.y + i.height + 1);
		rectangle(IMAGE, pt1, pt2, COLOR, THICKNESS);
		cropped_faces.push_back(IMAGE(Range(i.y, i.y + i.height), Range(i.x, i.x + i.width)));
	}
	print("Faces count: " + to_string(faces.size()), DEBUG_MODE);

	display("Faces detected", IMAGE, DEBUG_MODE);

	// Displaying cropped faces from the original image
	print("Displaying cropped faces from the original image", DEBUG_MODE);
	if (DEBUG_MODE) {
		for (auto &face: cropped_faces) {
			display("Cropped Face", face, DEBUG_MODE);
		}
	}

	return cropped_faces;
}

#endif //MAIN_FACEDETECTION_H
