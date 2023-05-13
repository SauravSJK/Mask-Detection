// Program3.cpp
// Description:
// Assumptions:
// Authors: Harpreet Kour, Saurav Jayakumar, Utkarsh Darbari

// Import the necessary libraries for opencv and i/o
#include <iostream>
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
	namedWindow(WINNAME, WINDOW_NORMAL);
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


// The main function
// Parameters:
// Pre-condition:
// Post-condition:
int main()
{
	// Reading kitten images from disk and displaying it
	cout << "Reading images from disk" << endl;
	Mat face = readDisplay("Images/with_mask_1564.jpg", "face");

	// Converting the image to grayscale
	Mat face_gray;
	cvtColor(face, face_gray, COLOR_BGR2GRAY);
	display("Grayscale", face_gray);

	// Equalizing the histogram of the grayscale image to normalize brightness and increase contrast
	Mat face_gray_eq;
	equalizeHist(face_gray, face_gray_eq);
	display("Equalized Histogram", face_gray_eq);

	// Blurring the image using a Gaussian Kernel to smoothen the image
	Mat face_gray_eq_blur;
	GaussianBlur(face_gray_eq, face_gray_eq_blur, Size(5,5), 0);
	display("Smoothened Image", face_gray_eq_blur);

	// Loading the face cascades
	CascadeClassifier face_cascade;
	if( !face_cascade.load( "Haarcascades/haarcascade_frontalface_alt.xml" ) )
	{
		cout << "Error loading face cascade\n";
		return -1;
	};

	// Detecting face in the image
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(face_gray_eq_blur, faces);

	for (auto & i : faces) {
		Point center(i.x + i.width / 2, i.y + i.height / 2);
		ellipse(face, center, Size(i.width / 2, i.height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
	}

	display("Face detected", face);

	return 0;
}