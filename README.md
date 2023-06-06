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

We tested our program on the selected subset of the entire dataset and manually noted whether the program was able to accurately detect the correct faces and eyes before the actual mask detection algorithm. Based on the individual image results, we calculated the summary results shown in the below table:

| Images           | Face Detection | Additional Incorrect Face Detection | Eye Detection | Additional Incorrect Eye Detection | Mask Detection | Non-Mask Detection |
|------------------|----------------|-------------------------------------|---------------|------------------------------------|----------------|--------------------|
| Masked Faces     | 98.8%          | 0%                                  | 97.7%         | 2.3%                               | 96.3%          | 3.7%               |
| Non-Masked Faces | 98.9%          | 5.8%                                | 88.5%         | 6.9%                               | 11%            | 89%                |
| Combined         | 98.85%         | 2.9%                                | 90.1%         | 4.6%                               | N/A            | N/A                |

Based on the combined dataset, we also calculated the below metrics:

| Metric    | Value |
|-----------|-------|
| Accuracy  | 92.7% |
| Precision | 82.8% |
| Recall    | 96.3% |
| F1-Score  | 92.9% |


## Known Issues

1. If the face and eye detection classifiers fail, our program will not be able to check for masks
2. Results displayed and written are based on the classifier and detection variables which may not be accurate (E.g., the program might detect 2 faces as expected but the detections might not actually be faces)

## Future Improvements

1. Incorporate smile detection classifiers to distinguish masks from beards
2. Improve face and eye detection performance by testing additional pre-processing methods or different classifiers

## Testing Steps

1. Clone the repository
2. For testing with new images (".jpg"s only):
	1. update the DIRECTORY\_PATH variable in the main.cpp file to point to the directory containing the test images
	2. rename images to [with/without]\_mask\_[image id]\_count\_[number of faces].jpg to get summary results
3. Ensure that C++ 17 is available in the system as the program utilizes the "filesystem" library which is only supported in C++ 17
4. Update the OpenCV library path under OpenCV\_DIR in the CMakeLists.txt file on line 29
5. Build and run the main.cpp program to execute the mask detection algorithm
6. Summary results are displayed in the terminal and individual image results are written to a csv file 