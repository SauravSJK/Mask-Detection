# Mask Detection using OpenCV
A C++ program to detect whether an image contains faces wearing a mask using OpenCV

## Introduction

## Methods

1. Image pre-processing:
    1. Grayscale color conversion
    2. Histogram equalization
    3. Gaussian filtering
2. Face detection:
    1. Haar: haarcascade_frontalface_default.xml
    2. LBP: lbpcascade_frontalface_improved.xml
3. Eye detection:
    1. haarcascade_lefteye_2splits.xml
    2. haarcascade_righteye_2splits.xml
    3. haarcascade_eye_tree_eyeglasses.xml
4. Skin area segmentation:
    1. YCrCb color conversion
    2. Otsu thresholding using Cr component
5. Oronasal region selection using eye bounding boxes
6. Comparison of skin areas

## Results

## Known Issues

1. If the face and eye detection classifiers fail, our program will not be able to check for masks
2. Results displayed and written are based on the classifier and detection variables which may not be accurate (E.g., the program might detect 2 faces as expected but the detections might not actually be faces)

## Future Improvements

1. Incorporate smile detection classifiers to distinguish masks from beards
2. Improve face and eye detection performance by testing additional pre-processing methods or different classifiers

## Testing Steps

1. Clone the repository
2. For testing with new images (".jpg"s only), update the DIRECTORY_PATH variable in the main.cpp file to point to the directory containing the test images
3. Ensure that C++ 17 is available in the system as the program utilizes the "filesystem" library which is only supported in C++ 17
4. Update the OpenCV library path under OpenCV_DIR in the CMakeLists.txt file on line 29
5. Build and run the main.cpp program to execute the mask detection algorithm
6. Summary results are displayed in the terminal and individual image results are written to a csv file 