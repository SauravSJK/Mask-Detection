// Program3.cpp
// Description:
// Assumptions:
// Authors: Harpreet Kour, Saurav Jayakumar, Utkarsh Darbari
// Future improvements:
//      Separate functions to different files
//		Implement the rest of the mask detection methods
//      Try other pre-processing steps to improve detection
//      Update program to read multiple random images or Implement a GUI to select images or use video inputs
//      Suppress display function calls or display images in parallel
//

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

// Declaring the namespaces that would be used throughout the program
// We can use 2 namespaces as long as there aren't any conflicts
using namespace std;
using namespace cv;

// Displays image in the specified window, waits for the user's input to change, and then destroys the created window if its not closed
// Parameters:
//        winname: String used to represent the window name which will be used to display the image
//        img:     Mat used to hold the image that needs to be displayed
//        scale:   Value used to scale the image down before displaying it
// Pre-condition:  The program expects the arguments to be a string and an image
// Post-condition: The image is displayed in a window with the window name passed to the function and then the window will be destroyed
void display(const string& WINNAME, const Mat& IMG, const int SCALE = 1) {
	namedWindow(WINNAME, WINDOW_AUTOSIZE);
	resizeWindow(WINNAME, IMG.cols / SCALE, IMG.rows / SCALE);

	imshow(WINNAME, IMG);

	waitKey(0);
	if (getWindowProperty(WINNAME, WND_PROP_VISIBLE ) != 0)
		destroyWindow(WINNAME);
}

// Reads an image with the specified name from the current directory and displays and returns it
// Parameters:
//         path:    Location containing the image
//         winname: A window name for displaying the image
// Pre-condition:   The program expects the path to point to a valid jpg image and the winname to be a string
// Post-condition:  The image is displayed in a window with the window name same as the filename and the image is returned
Mat readDisplay(const string &PATH, const string& WINNAME) {
	Mat img = imread(PATH);
	// If the image is empty, exit the program
	if (img.empty()) {
		cout << "Invalid path" << endl;
		exit(0);
	}

	display(WINNAME, img);

	return img;
}

// The pre-processing function accepts an image, converts it to grayscale, equalizes the histogram, and smoothens it
// Parameters: A map variable to hold the images from various pre-processing stages
// Pre-condition: A valid map variable is passed to the arguement with the original image at key = "Image"
// Post-condition: The map variable will be updated with the various images
// Future improvements:
//      Set up a default constructor for the map to avoid temp variables
//      Experiment with the blurring parameters
void preProcessing (map<string, Mat> &images) {
	// Converting the image to grayscale
	cout << "Converting the image to grayscale" << endl;
	Mat temp;
	cvtColor(images.at("Image"), temp, COLOR_BGR2GRAY);
	images.insert(make_pair("Grayscale", temp));
	display("Grayscale", images.at("Grayscale"));

	// Equalizing the histogram of the grayscale image to normalize brightness and increase contrast
	cout << "Equalizing the histogram of the grayscale image" << endl;
	equalizeHist(images.at("Grayscale"), temp);
	images.insert(make_pair("Grayscale_EQ", temp));
	display("Equalized Histogram", images.at("Grayscale_EQ"));

	// Blurring the image using a Gaussian Kernel to smoothen the image
	cout << "Blurring the image" << endl;
	int k_width = 5, k_height = 5, k_sigma_X = 0, k_sigma_Y = 0;
	GaussianBlur(images.at("Grayscale_EQ"), temp, Size(k_width,k_height), k_sigma_X, k_sigma_Y);
	images.insert(make_pair("Grayscale_EQ_Blur", temp));
	display("Smoothened Image", images.at("Grayscale_EQ_Blur"));
}

// The face detection function loads a face haar cascade file and uses it to detect faces from an image
// Parameters: A map variable with the pre-processes images and location of the haar cascade xml file
// Pre-condition: The map variable should contain valid jpg images with the expected keys and the filename should point to the correct cascade xml file
// Post-condition: The faces detected in the image are first displayed and then returned to the main function as a vector of matrices
// Future improvements: Include cropped images in the map
vector<Mat> faceDetection (map<string, Mat> &images, const String& CASCADE_FILENAME) {
	// Loading the face cascades
	cout << "Loading the face cascades" << endl;
	CascadeClassifier face_cascade;
	if(!face_cascade.load(CASCADE_FILENAME)) {
		cout << "Error loading face cascade\n";
		exit(0);
	};

	// Detecting faces in the image
	cout << "Detecting faces in the image" << endl;
	vector<Rect> faces;
	vector<Mat> cropped_faces;
	face_cascade.detectMultiScale(images.at("Grayscale_EQ_Blur"), faces);
	for (auto & i : faces) {
		Point pt1(i.x, i.y);
		Point pt2(i.x + i.width, i.y + i.height);
		rectangle(images.at("Image"), pt1, pt2, Scalar(255, 0, 255), 4);
		cropped_faces.push_back(images.at("Image")(Range(i.y, i.y + i.height), Range(i.x, i.x + i.width)));
	}

	display("Faces detected", images.at("Image"));

	// Displaying cropped faces from the original image
	cout << "Displaying cropped faces from the original image" << endl;
	for (auto & face: cropped_faces) {
		display("Cropped Face", face);
	}

	return cropped_faces;
}

// The main function loads an image, pre-processes it, detects face, models the skin color, detects the oronasal region, and detects mask
// Parameters: N/A
// Pre-condition: Expects a valid jpg image and a valid haar cascade face xml file at the specified locations
// Post-condition: Returns whether the faces in the image, if any, wore a mask or not
int main()
{
	map<string, Mat> images;
	// Reading an image which might have faces from disk and displaying it
	cout << "Reading image from disk" << endl;
	images.insert(make_pair("Image", readDisplay("../Images/with_mask/with_mask_1564.jpg", "Image")));

	// Sending the image for pre-processing and receiving all modified images in the map object
	cout << "Pre-processing" << endl;
	preProcessing(images);

	// Sending the images for face detection and receiving the set of faces from the image
	cout << "Face detection" << endl;
	vector<Mat> cropped_faces = faceDetection(images, "../Haarcascades/haarcascade_frontalface_alt.xml");

	// Exiting the program if no faces were detected
	cout << "Exiting if no faces were detected" << endl;
	if (cropped_faces.empty()) {
		cout << "Didn't detect any faces in the image" << endl;
		return 0;
	}

	return 0;
}