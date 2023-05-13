// Program3.cpp
// Description:
// Assumptions:
// Authors: Harpreet Kour, Saurav Jayakumar, Utkarsh Darbari

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
Mat readDisplay(const string &path, const string& winname) {
	Mat img = imread(path);
	// If the image is empty, exit the program
	if (img.empty()) {
		cout << "Invalid path" << endl;
		exit(0);
	}

	display(winname, img);

	return img;
}

// The pre-processing function
// Parameters:
// Pre-condition:
// Post-condition:
// Future improvements: setup a default constructor for the map to avoid temp variables
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

// The face detection function
// Parameters:
// Pre-condition:
// Post-condition:
// Future improvements: Include cropped images in the map
vector<Mat> faceDetection (map<string, Mat> &images, const String& cascade_filename) {
	// Loading the face cascades
	cout << "Loading the face cascades" << endl;
	CascadeClassifier face_cascade;
	if(!face_cascade.load(cascade_filename)) {
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

// The main function
// Parameters:
// Pre-condition:
// Post-condition:
int main()
{
	map<string, Mat> images;
	// Reading an image which might have faces from disk and displaying it
	cout << "Reading image from disk" << endl;
	images.insert(make_pair("Image", readDisplay("Images/with_mask_1564.jpg", "Image")));

	// Sending the image for pre-processing and receiving all modified images in the map object
	preProcessing(images);

	// Sending the images for face detection and receiving the set of faces from the image
	vector<Mat> cropped_faces = faceDetection(images, "Haarcascades/haarcascade_frontalface_alt.xml");

	return 0;
}