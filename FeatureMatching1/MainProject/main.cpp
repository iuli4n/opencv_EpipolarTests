//#include "stdafx.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <fstream>


using namespace cv;
using namespace std;


void checkFileExists(String filename) {
	ifstream infile;
	infile.open(filename, ios::in);
	if (infile.fail())
	{
		cout << "ERROR. Failed to open file.\n";
		return;
	}
	else {
		cout << "File opened.\n";
		infile.close();
		return;
	}
}


int main(int argc, char** argv)
{
	String filename = "..\\IulianData\\Iulian_Headshot.jpg";
	//checkFileExists(filename);



	const cv::Mat input = cv::imread(filename, 0); //Load as grayscale

	// detect keypoints in image using the feature detector
	std::vector<cv::KeyPoint> keypoints;
	cv::OrbFeatureDetector detector;
	detector.detect(input, keypoints);

	// Add results to image and save.
	cv::Mat output;
	cv::drawKeypoints(input, keypoints, output, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);




	////////// Display the image
	Mat image = output; //imread(filename);

	if (image.empty()) // Check for failure
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //wait for any key press
		return -1;
	}

	String windowName = "My HelloWorld Window"; //Name of the window

	namedWindow(windowName); // Create a window

	imshow(windowName, image); // Show our image inside the created window.

	waitKey(0); // Wait for any keystroke in the window

	destroyWindow(windowName); //destroy the created window

	return 0;
}