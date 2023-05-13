// Program3.cpp
// Description:
// Assumptions:
// Authors: Harpreet Kour, Saurav Jayakumar, Utkarsh Darbari

// Import the necessary libraries for opencv and i/o
#include <iostream>
#include <vector>
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


// The main function
// Parameters:
// Pre-condition:
// Post-condition:
int main()
{
	// Reading an image which might have faces from disk and displaying it
	cout << "Reading image from disk" << endl;
	Mat image = readDisplay("Images/with_mask_1564.jpg", "Image");

	// Converting the image to grayscale
	Mat image_gray;
	cvtColor(image, image_gray, COLOR_BGR2GRAY);
	display("Grayscale", image_gray);

	// Equalizing the histogram of the grayscale image to normalize brightness and increase contrast
	Mat image_gray_eq;
	equalizeHist(image_gray, image_gray_eq);
	display("Equalized Histogram", image_gray_eq);

	// Blurring the image using a Gaussian Kernel to smoothen the image
	Mat image_gray_eq_blur;
	int kwidth = 5, kheight = 5;
	GaussianBlur(image_gray_eq, image_gray_eq_blur, Size(kwidth,kheight), 0);
	display("Smoothened Image", image_gray_eq_blur);

	// Loading the face cascades
	CascadeClassifier face_cascade;
	String filename = "Haarcascades/haarcascade_frontalface_alt.xml";
	if(!face_cascade.load(filename)) {
		cout << "Error loading face cascade\n";
		return -1;
	};

	// Detecting faces in the image
	std::vector<Rect> faces;
	vector<Mat> cropped_faces;
	face_cascade.detectMultiScale(image_gray_eq_blur, faces);
	for (auto & i : faces) {
		Point pt1(i.x, i.y);
		Point pt2(i.x + i.width, i.y + i.height);
		rectangle(image, pt1, pt2, Scalar(255, 0, 255), 4);
		cropped_faces.push_back(image(Range(i.y, i.y + i.height), Range(i.x, i.x + i.width)));
	}
	display("Faces detected", image);

	// Displaying cropped faces from the original image
	for (auto & i: cropped_faces) {
		display("Cropped Face", i);
	}

	return 0;
}