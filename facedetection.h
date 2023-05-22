//
// Created by Saurav Jayakumar on 5/21/23.
//

#ifndef MAIN_FACEDETECTION_H
#define MAIN_FACEDETECTION_H

#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include "helper.h"

using namespace std;
using namespace cv;

// The face detection function loads a face haar cascade file and uses it to detect faces from an image
// Parameters: A map variable with the pre-processed images and location of the haar cascade xml file
// Pre-condition: The map variable should contain valid image matrices with the expected keys and the filename should point to the correct cascade xml file
// Post-condition: The faces detected in the image are first displayed and then returned to the main function as a vector of matrices
vector<Mat> faceDetection (const Mat& image, const Mat& pre_processed_image, const String& CASCADE_FILENAME, const bool DEBUG_MODE) {
	// Loading the face cascades
	cout << "Loading the face cascades" << endl;
	CascadeClassifier face_cascade;
	if(!face_cascade.load(CASCADE_FILENAME)) {
		cout << "Error loading face cascade\n";
		exit(0);
	}

	// Detecting faces in the image
	cout << "Detecting faces in the image" << endl;
	vector<Rect> faces;
	vector<Mat> cropped_faces;
	const Scalar COLOR = Scalar(255, 0, 255);
	const int THICKNESS = 1;
	face_cascade.detectMultiScale(pre_processed_image, faces);

	for (auto & i : faces) {
		Point pt1(i.x - 1, i.y - 1);
		Point pt2(i.x + i.width + 1, i.y + i.height + 1);
		rectangle(image, pt1, pt2, COLOR, THICKNESS);
		cropped_faces.push_back(image(Range(i.y, i.y + i.height), Range(i.x, i.x + i.width)));
	}

	if (DEBUG_MODE) {
		display("Faces detected", image);

		// Displaying cropped faces from the original image
		cout << "Displaying cropped faces from the original image" << endl;
		for (auto &face: cropped_faces) {
			display("Cropped Face", face);
		}
	}

	return cropped_faces;
}

#endif //MAIN_FACEDETECTION_H
