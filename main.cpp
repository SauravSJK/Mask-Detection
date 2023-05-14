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
//      Add all images and vectors to the map variable

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
// Pre-condition: A valid map variable is passed to the argument with the original image at key = "Image"
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
// Parameters: A map variable with the pre-processed images and location of the haar cascade xml file
// Pre-condition: The map variable should contain valid image matrices with the expected keys and the filename should point to the correct cascade xml file
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
	const Scalar COLOR = Scalar(255, 0, 255);
	const int THICKNESS = 1;
	face_cascade.detectMultiScale(images.at("Grayscale_EQ_Blur"), faces);

	for (auto & i : faces) {
		Point pt1(i.x - 1, i.y - 1);
		Point pt2(i.x + i.width + 1, i.y + i.height + 1);
		rectangle(images.at("Image"), pt1, pt2, COLOR, THICKNESS);
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

// The skin color segmentation takes in a set of cropped faces, converts them to YCrCb color space, and uses the Cr component for Otsu thresholding
// Parameters: A vector of matrices with cropped face images
// Pre-condition: The vector contains valid matrices with cropped face images
// Post-condition: Images are displayed at various stages of the segmentation and the final output is returned to the caller
// Future improvements: Include the vectors in the map
vector<Mat> skinColorSegmentation (const vector<Mat>& cropped_faces) {
	// Converting cropped faces to YCrCb color space
	cout << "Converting cropped faces to YCrCb color space" << endl;
	vector<Mat> cropped_faces_ycrcb;
	for (auto &face: cropped_faces) {
		Mat face_ycrcb;
		cvtColor(face, face_ycrcb, COLOR_BGR2YCrCb);
		cropped_faces_ycrcb.push_back(face_ycrcb);
		display("YCrCb Faces", face_ycrcb);
	}

	// Extracting Cr component of the image
	cout << "Extracting Cr component of the image" << endl;
	vector<Mat> cr_faces;
	for (auto &face: cropped_faces_ycrcb) {
		Mat channels[3];
		split(face, channels);
		cr_faces.push_back(channels[1]);
		display("Cr component of the face image", channels[1]);
	}

	// Applying Otsu thresholding for skin color segmentation
	cout << "Applying Otsu thresholding for skin color segmentation" << endl;
	vector<Mat> otsu_cr_faces;
	for (auto &face: cr_faces) {
		Mat otsu;
		threshold(cr_faces[0], otsu, 0, 255, THRESH_OTSU);
		otsu_cr_faces.push_back(otsu);
		display("Otsu Thresholding", otsu);
	}

	return otsu_cr_faces;
}

// The eye detection function loads a eye haar cascade file and uses it to detect eyes from a face image
// Parameters: A vector of matrices with the cropped face images and location of the haar cascade xml file
// Pre-condition: The vector contains valid matrices with cropped face images and the filename should point to the correct cascade xml file
// Post-condition: The eyes detected in each of the faces are first displayed and then the coordinates of the bounding boxes are returned
// Future improvements: Include eyes box images in the map
vector<vector<Rect>> eyeDetection (vector<Mat> cropped_faces, const String& CASCADE_FILENAME) {
	// Loading the face cascades
	cout << "Loading the face cascades" << endl;
	CascadeClassifier eye_cascade;
	if(!eye_cascade.load(CASCADE_FILENAME)) {
		cout << "Error loading face cascade\n";
		exit(0);
	};

	vector<vector<Rect>> eye_boxes;
	for (auto &face: cropped_faces) {
		// Detecting eyes in the image
		cout << "Detecting eyes in the image" << endl;
		vector<Rect> eyes;
		const Scalar COLOR = Scalar(255, 0, 255);
		const int THICKNESS = 1;
		eye_cascade.detectMultiScale(face, eyes);
		eye_boxes.push_back(eyes);

		for (auto & i : eyes) {
			Point pt1(i.x - 1, i.y - 1);
			Point pt2(i.x + i.width + 1, i.y + i.height + 1);
			rectangle(face, pt1, pt2, COLOR, THICKNESS);
		}

		display("Eyes detected", face);
	}

	return eye_boxes;
}

// The mask detection function accepts the Otsu thresholded Cr components and eye bounding boxes for mask detection
// by comparing skin areas between eye part and mouth part
// Parameters: A vector of matrices with the Otsu thresholded Cr components and vector of eye bounding boxes
// Pre-condition: The vectors contains valid data and correspond to the same face in the same order
// Post-condition: The function outputs whether the person in the image is wearing a mask or not
// Future improvements: Split the if statement
void maskDetection(vector<Mat> otsu_cr_faces, vector<vector<Rect>> eye_boxes) {
	for (int i = 0; i < otsu_cr_faces.size(); i++) {
		if (countNonZero(otsu_cr_faces.at(i)(Range(min(eye_boxes.at(i).at(0).y, eye_boxes.at(i).at(1).y), max(eye_boxes.at(i).at(0).y + eye_boxes.at(i).at(0).height, eye_boxes.at(i).at(1).y + eye_boxes.at(i).at(1).height)), Range(min(eye_boxes.at(i).at(0).x, eye_boxes.at(i).at(1).x), max(eye_boxes.at(i).at(0).x + eye_boxes.at(i).at(0).width, eye_boxes.at(i).at(1).x + eye_boxes.at(i).at(1).width)))) > 1.2 * countNonZero(otsu_cr_faces.at(i)(Range(min(eye_boxes.at(i).at(0).y + eye_boxes.at(i).at(0).height, eye_boxes.at(i).at(1).y + eye_boxes.at(i).at(1).height), max(min(eye_boxes.at(i).at(0).y + 3 * eye_boxes.at(i).at(0).height, otsu_cr_faces.at(i).rows), min(eye_boxes.at(i).at(1).y + 3 * eye_boxes.at(i).at(1).height, otsu_cr_faces.at(i).rows))), Range(min(eye_boxes.at(i).at(0).x, eye_boxes.at(i).at(1).x), max(eye_boxes.at(i).at(0).x + eye_boxes.at(i).at(0).width, eye_boxes.at(i).at(1).x + eye_boxes.at(i).at(1).width))))) {
			cout << "Mask detected" << endl;
		}
		else {
			cout << "Mask not detected" << endl;
		}
	}
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
	images.insert(make_pair("Image", readDisplay("Images/with_mask/with_mask_1564.jpg", "Image")));

	// Example of not detecting a face
	// images.insert(make_pair("Image", readDisplay("Images/without_mask/without_mask_3732.jpg", "Image")));

	// Example of not detecting an eye
	// images.insert(make_pair("Image", readDisplay("Images/without_mask/without_mask_3731.jpg", "Image")));

	// Example of multiple unaligned faces in an image
	// images.insert(make_pair("Image", readDisplay("Images/without_mask/without_mask_3502.jpg", "Image")));

	// Passing the image for pre-processing and receiving all modified images in the map object
	cout << "Pre-processing" << endl;
	preProcessing(images);

	// Passing the images for face detection and receiving the set of faces from the image
	cout << "Face detection" << endl;
	vector<Mat> cropped_faces = faceDetection(images, "Haarcascades/haarcascade_frontalface_alt.xml");

	// Exiting the program if no faces were detected
	cout << "Exiting if no faces were detected" << endl;
	if (cropped_faces.empty()) {
		cout << "Didn't detect any faces in the image" << endl;
		return(0);
	}

	// Passing the cropped face images for skin color segmentation and receiving Otsu thresholded Cr components of them
	cout << "Skin color segmentation" << endl;
	vector<Mat> otsu_cr_faces = skinColorSegmentation(cropped_faces);

	// Passing the cropped images for eye detection and receiving the bounding boxes for the eyes
	cout << "Eye detection" << endl;
	vector<vector<Rect>> eye_boxes = eyeDetection(cropped_faces, "Haarcascades/haarcascade_eye_tree_eyeglasses.xml");

	// Passing the Otsu thresholded Cr components and the eye bounding boxes for mask detection
	cout << "Mask detection" << endl;
	maskDetection(otsu_cr_faces, eye_boxes);

	return 0;
}