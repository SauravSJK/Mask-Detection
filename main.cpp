// Main.cpp
// Description: A program to run the mask detection algorithm on a set of images and predict the number of faces in the image wearing a mask
// Assumptions: The cascade and image files are present in the expected locations
// Authors: Saurav Jayakumar, Utkarsh Darbari
// Future improvements:
//      Try other pre-processing steps to improve detection - Harpreet

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <vector>
#include <string>
#include "headers/helper.h"
#include "headers/maskdetection.h"

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// Controls the display function calls to reduce the number of images displayed
const bool DEBUG_MODE = false;

// The main function runs the mask detection function on a set of images
// Parameters: N/A
// Pre-condition: Expects valid jpg images and cascade files in the specified locations
// Post-condition: Prints the count of faces with masks, without masks, faces not detected, and eyes not detected
int main()
{
	vector<int> counts = {0, 0, 0, 0};
	const string DIRECTORY_PATH = "Images/without_mask/";
	const string FACE_HAAR_CASCADE_FILENAME = "Haarcascades/haarcascade_frontalface_default.xml";
	const string FACE_LBP_CASCADE_FILENAME = "LBPcascades/lbpcascade_frontalface_improved.xml";
	const string LEFT_CASCADE_FILENAME = "Haarcascades/haarcascade_lefteye_2splits.xml";
	const string RIGHT_CASCADE_FILENAME = "Haarcascades/haarcascade_righteye_2splits.xml";
	const string GLASS_CASCADE_FILENAME = "Haarcascades/haarcascade_eye_tree_eyeglasses.xml";

	// Loading the file names
	print("Loading the file names", DEBUG_MODE);
	const vector<string> FILE_PATHS = getFileNames(DIRECTORY_PATH, DEBUG_MODE);

	// Loading the cascade files
	print("Loading the cascade files", DEBUG_MODE);
	const CascadeClassifier FACE_HAAR_CASCADE = loadCascade(FACE_HAAR_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier FACE_LBP_CASCADE = loadCascade(FACE_LBP_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier LEFT_EYE_CASCADE = loadCascade(LEFT_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier RIGHT_EYE_CASCADE = loadCascade(RIGHT_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier EYE_GLASS_CASCADE = loadCascade(GLASS_CASCADE_FILENAME, DEBUG_MODE);

	// Running the mask detection algorithm through each of the image file
	for (const auto & FILE_PATH : FILE_PATHS){
		const vector<int> TEMP_COUNTS = maskDetection(FILE_PATH, FACE_HAAR_CASCADE, FACE_LBP_CASCADE, LEFT_EYE_CASCADE, RIGHT_EYE_CASCADE, EYE_GLASS_CASCADE, DEBUG_MODE);
		counts.at(0) += TEMP_COUNTS.at(0);
		counts.at(1) += TEMP_COUNTS.at(1);
		counts.at(2) += TEMP_COUNTS.at(2);
		counts.at(3) += TEMP_COUNTS.at(3);
	}

	// Printing the final counts
	cout << "Total Count of Masked faces: " << counts.at(1) << endl;
	cout << "Total Count of Non_Masked faces: " << counts.at(2) << endl;
	cout << "Total Count of skipped Faces: " << counts.at(0) << endl;
	cout << "Total Count of skipped Images: " << counts.at(3) << endl;
	cout << "Count of all images: " << FILE_PATHS.size() << endl;

	return 0;
}

/*
 *  With_Mask
 *
 *  Total Count of Masked faces: 1008
 *  Total Count of Non_Masked faces: 72
 *  Total Count of skipped Faces: 925
 *  Actually considered number of images: 1416
 *  Total Count of skipped Images: 2309
 *  Count of all images: 3725
*/

/*
 *  Without_Mask
 *
 *  Total Count of Masked faces: 473
 *  Total Count of Non_Masked faces: 2774
 *  Total Count of skipped Faces: 739
 *  Actually considered number of images: 3213
 *  Total Count of skipped Images: 597
 *  Count of all images: 3828

 */