// Main.cpp
// Description:
// Assumptions:
// Authors: Harpreet Kour, Saurav Jayakumar, Utkarsh Darbari
// Future improvements:
//      Try other pre-processing steps to improve detection - Harpreet
//      Update program to read multiple random images or Implement a GUI to select images or use video inputs - Utkarsh

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include "helper.h"
#include "preprocessing.h"
#include "facedetection.h"
#include "postprocessing.h"

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// Controls the display function calls to reduce the number of images displayed
const bool DEBUG_MODE = false;

// The main function loads an image, pre-processes it, detects face, models the skin color, detects the oronasal region, and detects mask
// Parameters: N/A
// Pre-condition: Expects a valid jpg image and a valid haar cascade face xml file at the specified locations
// Post-condition: Returns whether the faces in the image, if any, wore a mask or not
// Future implementation: Add try catch blocks to avoid exceptions
int main()
{
	// Reading an image which might have faces from disk and displaying it
	cout << "Reading image from disk" << endl;
	Mat image = readDisplay("Images/with_mask/with_mask_1564.jpg", "Image");

	// Example of not detecting a face
	// images.insert(make_pair("Image", readDisplay("Images/without_mask/without_mask_3732.jpg", "Image")));

	// Example of not detecting an eye
	// images.insert(make_pair("Image", readDisplay("Images/without_mask/without_mask_3731.jpg", "Image")));

	// Example of multiple unaligned faces in an image
	// images.insert(make_pair("Image", readDisplay("Images/without_mask/without_mask_3502.jpg", "Image")));

	// Example of a profile face image
	// images.insert(make_pair("Image", readDisplay("Images/without_mask/without_mask_2885.jpg", "Image")));

	// Passing the image for pre-processing and receiving all modified images in the map object
	cout << "Pre-processing" << endl;
	Mat pre_processed_image = preProcessing(image, DEBUG_MODE);

	// Passing the images for face detection and receiving the set of faces from the image
	cout << "Face detection" << endl;
	vector<Mat> cropped_frontal_faces = faceDetection(image, pre_processed_image, "Haarcascades/haarcascade_frontalface_default.xml", DEBUG_MODE);

	// Trying LBP cascade classifier if no faces were detected by the haar cascade classifier
	cout << "Trying LBP cascade classifier if no faces were detected by the haar cascade classifier" << endl;
	if (cropped_frontal_faces.empty()) {
		cropped_frontal_faces = faceDetection(image, pre_processed_image, "LBPcascades/lbpcascade_frontalface_improved.xml", DEBUG_MODE);
		// Exiting if no faces were found by the LBP cascade classifier too
		if (cropped_frontal_faces.empty()) {
			cout << "Didn't detect any faces in the image" << endl;
			return (0);
		}
	}

	// Passing the cropped face images for skin color segmentation and receiving Otsu thresholded Cr components of them
	cout << "Skin color segmentation" << endl;
	vector<Mat> otsu_cr_faces = skinColorSegmentation(cropped_frontal_faces, DEBUG_MODE);

	// Passing the cropped images for eye detection and receiving the bounding boxes for the eyes
	cout << "Eye detection" << endl;
	vector<vector<int>> eye_nose_mouth_boxes = eyeNoseMouthDetection(cropped_frontal_faces, "Haarcascades/haarcascade_lefteye_2splits.xml", "Haarcascades/haarcascade_righteye_2splits.xml", "Haarcascades/haarcascade_eye_tree_eyeglasses.xml", DEBUG_MODE);

	// Passing the Otsu thresholded Cr components and the eye bounding boxes for mask detection
	cout << "Mask detection" << endl;
	maskDetection(otsu_cr_faces, eye_nose_mouth_boxes);

	return 0;
}