//
// Created by Saurav Jayakumar on 5/21/23.
//

#ifndef MAIN_HELPER_H
#define MAIN_HELPER_H

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// If running in debug mode, displays image in the specified window, waits for the user's input to change, and then destroys the created window if its not closed
// Parameters:
//          WINNAME:    String used to represent the window name which will be used to display the image
//          IMG:        Mat used to hold the image that needs to be displayed
//          DEBUG_MODE: To control the image display outputs
//          SCALE:      Value used to scale the image down before displaying it
// Pre-condition:  The program expects the arguments to be a string, an image, a boolean, and an integer
// Post-condition: The image is displayed in a window with the window name passed to the function and then the window will be destroyed
void display(const string& WINNAME, const Mat& IMG, const bool DEBUG_MODE, const int SCALE = 1) {
	if (DEBUG_MODE) {
		namedWindow(WINNAME, WINDOW_AUTOSIZE);
		resizeWindow(WINNAME, IMG.cols / SCALE, IMG.rows / SCALE);

		imshow(WINNAME, IMG);

		waitKey(0);
		if (getWindowProperty(WINNAME, WND_PROP_VISIBLE ) != 0)
			destroyWindow(WINNAME);
	}
}

// Reads an image with the specified name from the current directory and displays (if running in debug mode) and returns it
// Parameters:
//          PATH:       Location containing the image
//          WINNAME:    A window name for displaying the image
//          DEBUG_MODE: To control the image display outputs
// Pre-condition:   The program expects the path to point to a valid jpg image, the winname to be a string, and the debug_mode to be a boolean
// Post-condition:  The image is displayed in a window with the window name same as the filename if in debug mode and then the image is returned
Mat readDisplay(const string &PATH, const string& WINNAME, const bool DEBUG_MODE) {
	Mat img = imread(PATH);
	// If the image is empty, exit the program
	if (img.empty()) {
		cout << "Invalid path: " << PATH << endl;
		exit(0);
	}
	display(WINNAME, img, DEBUG_MODE);
	return img;
}

// If running in debug mode, outputs a text to the console
// Parameters:
//          TEXT:       Text to be displayed
//          DEBUG_MODE: To control the image display outputs
// Pre-condition:   The program expects a string as the text and a boolean for debug mode
// Post-condition:  Displays the text in console if running in debug mode
void print(const string &TEXT, const bool DEBUG_MODE) {
	if (DEBUG_MODE) {
		cout << TEXT << endl;
	}
}

// Fetches the list of jpg filenames recursively from the directory and extracts the relevant information from the file name
// Parameters:
//          DIRECTORY_PATH: Path to the directory containing the files
//          DEBUG_MODE:     To control the image display outputs
// Pre-condition:   The program expects a valid path for the directory and a boolean for debug mode
// Post-condition:  Returns the list of jpg file_paths from the directory and data about the file type, image id, and number of faces
vector<vector<string>> getFileNames(const string& DIRECTORY_PATH, const bool DEBUG_MODE) {
	vector<vector<string>> files;
	// Getting the filenames from the directory
	print("Getting the filenames from the directory", DEBUG_MODE);
	for (const auto& entry : __fs::filesystem::recursive_directory_iterator(DIRECTORY_PATH)) {
		vector<string> file_names;
		if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
			string file_name = entry.path().string();
			file_names.push_back(file_name);
			if (file_name.substr(file_name.find('/') + 1, file_name.rfind('/') - file_name.find('/') - 1) == "withmask") {
				file_names.emplace_back("With Mask");
			}
			else {
				file_names.emplace_back("Without Mask");
			}
			auto first_underscore = file_name.find('_');
			auto second_underscore = file_name.find('_', first_underscore + 1);
			auto third_underscore = file_name.find('_', second_underscore + 1);
			auto fourth_underscore = file_name.rfind('_');
			auto dot = file_name.rfind('.');
			file_names.push_back(file_name.substr(second_underscore + 1, third_underscore - second_underscore - 1));
			file_names.push_back(file_name.substr(fourth_underscore + 1, dot - fourth_underscore - 1));
			files.push_back(file_names);
		}
	}
	return files;
}

// Loads the specified cascade file and returns it
// Parameters:
//          FILENAME:   Path to the cascade file
//          DEBUG_MODE: To control the image display outputs
// Pre-condition:   The program expects a valid path for the cascade file and a boolean for debug mode
// Post-condition:  Returns the loaded cascade classifier object
CascadeClassifier loadCascade(const string& FILENAME, const bool DEBUG_MODE) {
	// Loading the cascade xml file
	print("Loading the cascade xml file", DEBUG_MODE);
	CascadeClassifier cascade;
	if(!cascade.load(FILENAME)) {
		cout << "Error loading the cascade file: " << FILENAME << endl;
		exit(0);
	}
	return cascade;
}

#endif //MAIN_HELPER_H
