//#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <fstream>


using namespace cv;
using namespace std;

#include "misc.hpp"


std::vector<DMatch> filterFeatureMatches(std::vector< std::vector<DMatch> >& knn_matches, float thresh = 0.7f) {
	// Returns only matches below the threshold

	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	return good_matches;
}



int main_imgs(int argc, char** argv)
{
	// LOAD images

	String filename1 = "..\\..\\_RawImages\\box.png";
	String filename2 = "..\\..\\_RawImages\\box_in_scene.png";

	
	const cv::Mat img1 = cv::imread(filename1, cv::IMREAD_GRAYSCALE); //Load as grayscale
	const cv::Mat img2 = cv::imread(filename2, cv::IMREAD_GRAYSCALE); //Load as grayscale

	if (img1.empty() || img2.empty()) // Check for failure
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //wait for any key press
		return -1;
	}


	// 
	// ===== DETECT FEATURES ======= using SIFT
	// 
	Ptr<SIFT> detector = SIFT::create(); 
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

	// SHOW keypoints
	if (false) {
		cv::Mat output;
		cv::drawKeypoints(img1, keypoints1, output);//, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		imshow("output", output);
	}


	//
	// ===== MATCHING THE FEATURES ======= with knn, and filtering
	//

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors2, descriptors1, knn_matches, 2);

	// Filter only good enough matches
	std::vector<DMatch> topMatches = filterFeatureMatches(knn_matches);

	// Extract the original point locations within each image (will be needed for homography)
	std::vector<Point2f> objPoints;
	std::vector<Point2f> scenePoints;
	for (size_t i = 0; i < topMatches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		objPoints.push_back(keypoints1[topMatches[i].trainIdx].pt);
		scenePoints.push_back(keypoints2[topMatches[i].queryIdx].pt);
	}


	// SHOW matches
	//if (false) 
		Mat img_matches;
		drawMatches(img2, keypoints2,
			img1, keypoints1,
			topMatches,
			img_matches, Scalar::all(-1),
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		imshow("Good Matches", img_matches);
	




	//
	// ===== POSE FINDING ======= with findHomography
	//

	// Figure out how the original points were moved to the scene
	Mat H = findHomography(objPoints, scenePoints, RANSAC);
	
	// Now draw the original rectangle into the scene
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img1.cols, 0);
	obj_corners[2] = Point2f((float)img1.cols, (float)img1.rows);
	obj_corners[3] = Point2f(0, (float)img1.rows);

	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

	// SHOW the rectangle in the scene
	line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

	imshow("Good Matches & Object detection", img_matches);

	waitKey(0); // Wait for any keystroke in the window

	return 0;
}



int main(int, char**)
{

	//// TRAINING DATA

	String filename1 = "..\\..\\_RawImages\\box.png";

	const cv::Mat img1 = cv::imread(filename1, cv::IMREAD_GRAYSCALE); //Load as grayscale
	assert(!img1.empty());

	Ptr<SIFT> siftdetector = SIFT::create();
	std::vector<cv::KeyPoint> keypoints1;
	cv::Mat descriptors1;
	siftdetector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	


	Mat frame;
	//--- INITIALIZE VIDEOCAPTURE
	VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_ANY;      // 0 = autodetect default API
	// open selected camera using selected API
	cap.open(deviceID, apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}



	std::vector<cv::KeyPoint> keypoints2;
	cv::Mat descriptors2;
	cv::Mat output;



	//--- GRAB AND WRITE LOOP
	cout << "Start grabbing" << endl
		<< "Press any key to terminate" << endl;
	for (;;)
	{
		// wait for a new frame from camera and store it into 'frame'
		cap.read(frame);
		// check if we succeeded
		if (frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		// show live and wait for a key with timeout long enough to show images
		imshow("Live", frame);
		
		// 
		// ===== DETECT FEATURES ======= using SIFT
		// 
			siftdetector->detectAndCompute(frame, noArray(), keypoints2, descriptors2);

			// SHOW keypoints
			if (true) {
				cv::drawKeypoints(frame, keypoints2, output);//, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				imshow("output", output);
			}

			if (waitKey(1) >= 0)
				return 0;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}