//
// Created by Saurav Jayakumar on 5/21/23.
//

#ifndef MAIN_PREPROCESSING_H
#define MAIN_PREPROCESSING_H

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "headers/helper.h"

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// The pre-processing function accepts an image, converts it to grayscale, equalizes the histogram, and smoothens it
// Parameters:
//          image:      A map variable to hold the images from various pre-processing stages
//          DEBUG_MODE: To control the image display outputs
// Pre-condition: A valid image is passed to the function
// Post-condition: The pre-processed image will be returned
// Future improvements: Experiment with the blurring parameters
Mat preProcessing (Mat image, const bool DEBUG_MODE) {
	// Converting the image to grayscale
	print("Converting the image to grayscale", DEBUG_MODE);
	cvtColor(image, image, COLOR_BGR2GRAY);
	display("Grayscale", image, DEBUG_MODE);

	// Equalizing the histogram of the grayscale image to normalize brightness and increase contrast
	print("Equalizing the histogram of the grayscale image", DEBUG_MODE);
	equalizeHist(image, image);
	display("Equalized Histogram", image, DEBUG_MODE);

	// Blurring the image using a Gaussian Kernel to smoothen the image
	print("Blurring the image", DEBUG_MODE);
	int k_width = 5, k_height = 5, k_sigma_X = 0, k_sigma_Y = 0;
	GaussianBlur(image, image, Size(k_width,k_height), k_sigma_X, k_sigma_Y);
	display("Smoothened Image", image, DEBUG_MODE);

	return image;
}

#endif //MAIN_PREPROCESSING_H
