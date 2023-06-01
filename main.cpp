// Main.cpp
// Description: A program to run the mask detection algorithm on a set of images and predict the number of faces in the image wearing a mask
// Assumptions: The cascade and image files are present in the expected locations
// Authors: Saurav Jayakumar, Utkarsh Darbari

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <fstream>
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
// Post-condition:
//              Prints the count of faces with masks, without masks, faces not detected, and eyes not detected for the set of masked and non-masked images
//              Outputs the results per image to a csv file
int main()
{
	// Initial variables for the mask detection testing program
	vector<int> masked_counts = {0, 0, 0, 0}, not_masked_counts = {0, 0, 0, 0};
	vector<int> TEMP_COUNTS;
	int ground_truth_masks = 0, ground_truth_no_masks = 0;
	const string DIRECTORY_PATH = "Dataset";
	const string FACE_HAAR_CASCADE_FILENAME = "Haarcascades/haarcascade_frontalface_default.xml";
	const string FACE_LBP_CASCADE_FILENAME = "LBPcascades/lbpcascade_frontalface_improved.xml";
	const string LEFT_CASCADE_FILENAME = "Haarcascades/haarcascade_lefteye_2splits.xml";
	const string RIGHT_CASCADE_FILENAME = "Haarcascades/haarcascade_righteye_2splits.xml";
	const string GLASS_CASCADE_FILENAME = "Haarcascades/haarcascade_eye_tree_eyeglasses.xml";

	// Loading the file names
	print("Loading the file names", DEBUG_MODE);
	const vector<vector<string>> FILES = getFileNames(DIRECTORY_PATH, DEBUG_MODE);

	// Loading the cascade files
	print("Loading the cascade files", DEBUG_MODE);
	const CascadeClassifier FACE_HAAR_CASCADE = loadCascade(FACE_HAAR_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier FACE_LBP_CASCADE = loadCascade(FACE_LBP_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier LEFT_EYE_CASCADE = loadCascade(LEFT_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier RIGHT_EYE_CASCADE = loadCascade(RIGHT_CASCADE_FILENAME, DEBUG_MODE);
	const CascadeClassifier EYE_GLASS_CASCADE = loadCascade(GLASS_CASCADE_FILENAME, DEBUG_MODE);

	// Loading the file to store the detection results for all images
	ofstream output;
	output.open("output.csv", ofstream::trunc);
	output << "File Type,Image ID,Ground Truth,Skipped Faces (Face issue),Skipped Faces (Eye issue),Masked Faces,Non-masked Faces\n";

	// Running the mask detection algorithm through each of the image file
	for (const auto & FILE : FILES){
		const string& FILE_PATH = FILE.at(0);
		const string& FILE_TYPE = FILE.at(1);
		int image_id = stoi(FILE.at(2));
		int faces = stoi(FILE.at(3));

		if (FILE_TYPE == "With Mask") {
			ground_truth_masks += faces;
			TEMP_COUNTS = maskDetection(FILE_PATH, faces, FACE_HAAR_CASCADE, FACE_LBP_CASCADE, LEFT_EYE_CASCADE, RIGHT_EYE_CASCADE, EYE_GLASS_CASCADE, DEBUG_MODE);
			masked_counts.at(0) += TEMP_COUNTS.at(0);
			masked_counts.at(1) += TEMP_COUNTS.at(1);
			masked_counts.at(2) += TEMP_COUNTS.at(2);
			masked_counts.at(3) += TEMP_COUNTS.at(3);
			output << FILE_TYPE << "," << image_id << "," << faces << "," << TEMP_COUNTS.at(3) * faces << "," << TEMP_COUNTS.at(0) << "," << TEMP_COUNTS.at(1) << "," << TEMP_COUNTS.at(2) << "\n";
		}
		else {
			ground_truth_no_masks += faces;
			TEMP_COUNTS = maskDetection(FILE_PATH, faces, FACE_HAAR_CASCADE, FACE_LBP_CASCADE, LEFT_EYE_CASCADE, RIGHT_EYE_CASCADE, EYE_GLASS_CASCADE, DEBUG_MODE);
			not_masked_counts.at(0) += TEMP_COUNTS.at(0);
			not_masked_counts.at(1) += TEMP_COUNTS.at(1);
			not_masked_counts.at(2) += TEMP_COUNTS.at(2);
			not_masked_counts.at(3) += TEMP_COUNTS.at(3);
			output << FILE_TYPE << "," << image_id << "," << faces << "," << TEMP_COUNTS.at(3) << "," << TEMP_COUNTS.at(0) << "," << TEMP_COUNTS.at(1) << "," << TEMP_COUNTS.at(2) << "\n";
		}
	}

	// Printing the final counts
	cout << endl;
	cout << "Number of Masked faces: " << ground_truth_masks << endl;
	cout << "Detected Masked faces: " << masked_counts.at(1) << endl;
	cout << "Detected Non-masked faces: " << masked_counts.at(2) << endl;
	cout << "Skipped Faces due to eye detection issue: " << masked_counts.at(0) << endl;
	cout << "Skipped Faces due to face detection issue: " << masked_counts.at(3) << endl;

	cout << endl;
	cout << "Number of Non-masked faces: " << ground_truth_no_masks << endl;
	cout << "Detected Masked faces: " << not_masked_counts.at(1) << endl;
	cout << "Detected Non-masked faces: " << not_masked_counts.at(2) << endl;
	cout << "Skipped Faces due to eye detection issue: " << not_masked_counts.at(0) << endl;
	cout << "Skipped Faces due to face detection issue: " << not_masked_counts.at(3) << endl;

	return 0;
}

/*
 *  Number of Masked Images: 3725
 *  Total Count of Masked faces: 1008
 *  Total Count of Non_Masked faces: 72
 *  Total Count of skipped Faces: 925
 *  Total Count of skipped Images: 2309
 *
 *  Number of Non-Masked Images: 3828
 *  Total Count of Masked faces: 473
 *  Total Count of Non_Masked faces: 2774
 *  Total Count of skipped Faces: 739
 *  Total Count of skipped Images: 597
 */