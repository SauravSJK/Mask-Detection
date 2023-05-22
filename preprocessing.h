//
// Created by Saurav Jayakumar on 5/21/23.
//

#ifndef MAIN_PREPROCESSING_H
#define MAIN_PREPROCESSING_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

// The pre-processing function accepts an image, converts it to grayscale, equalizes the histogram, and smoothens it
// Parameters: A map variable to hold the images from various pre-processing stages
// Pre-condition: A valid map variable is passed to the argument with the original image at key = "Image"
// Post-condition: The map variable will be updated with the various images
// Future improvements:
//      Experiment with the blurring parameters
Mat preProcessing (Mat image, const bool DEBUG_MODE) {
	// Converting the image to grayscale
	cout << "Converting the image to grayscale" << endl;
	cvtColor(image, image, COLOR_BGR2GRAY);
	if (DEBUG_MODE) {
		display("Grayscale", image);
	}

	// Equalizing the histogram of the grayscale image to normalize brightness and increase contrast
	cout << "Equalizing the histogram of the grayscale image" << endl;
	equalizeHist(image, image);
	if (DEBUG_MODE) {
		display("Equalized Histogram", image);
	}

	// Blurring the image using a Gaussian Kernel to smoothen the image
	cout << "Blurring the image" << endl;
	int k_width = 5, k_height = 5, k_sigma_X = 0, k_sigma_Y = 0;
	GaussianBlur(image, image, Size(k_width,k_height), k_sigma_X, k_sigma_Y);
	if (DEBUG_MODE) {
		display("Smoothened Image", image);
	}

	return image;
}

#endif //MAIN_PREPROCESSING_H
