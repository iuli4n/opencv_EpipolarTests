/****
The way it works :

-does augmented reality by tracking a training image and overlaying a 3D cube on it

- the way it does AR is like this :
	--(one time) find the features from the training image
	-- FEATURE DETECT: find features from the live camera image
	-- MATCH: match between the features between the camera and the training
	-- POSE: calculate the pose transform between the camera space and the training object space
	-- DRAW: draws a 3d cube by projecting from object space to camera space

- if you want a new tracking image, you can train it from what the camera sees: 
	Push 1 to see the area of the camera that will be scanned, and then 
	Push SPACE to use that area as the tracking image

- debugging:
	Push 2 to see all the features detected in the camera image
	Push 3 to see what features are matched between the camera and the training image

Author: Iulian Radu 2024

***/


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
#include "main.h"


// draw the 3d cube in camera space
void drawCube(vector<Point2f> projectedPoints, Mat img) {
	// base
	for (int i = 0; i < 3; i++) {
		cv::line(img, projectedPoints[i], projectedPoints[i + 1], Scalar(0, 0, 255), 3);
	}
	cv::line(img, projectedPoints[3], projectedPoints[0], Scalar(0, 0, 255), 3);
	
	// top
	for (int i = 0; i < 3; i++) {
		cv::line(img, projectedPoints[4 + i], projectedPoints[4 + i + 1], Scalar(250, 0, 255), 3);
	}
	cv::line(img, projectedPoints[4+3], projectedPoints[4 + 0 ], Scalar(250, 0, 255), 3);

	// columns
	for (int i = 0; i < 4; i++) {
		cv::line(img, projectedPoints[i], projectedPoints[4 + i + 0], Scalar(0, 250, 255), 3);
	}

}

// filter the feature matches by choosing only the best ones based on good distance < threshold
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


// draw the training area rectangle
void train_drawArea(Mat frame, Rect trainingArea, Scalar color) {
	
	cv::rectangle(frame, trainingArea, color);
}

// Given a new image frame, detects SIFT features on the image
void processNewTrainingImage(Mat img1, Ptr<SIFT>& siftdetector, std::vector<cv::KeyPoint>& keypoints1, Mat& descriptors1) {
	keypoints1.clear();
	siftdetector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	
}

int main(int, char**)
{

	// LOAD CAMERA CONFIGURATION FROM FILE

	FileStorage fs("..\\..\\_RawImages\\camera.yml", FileStorage::READ);
	cv::Mat cam_cameramatrix; 
	fs["camera_matrix"] >> cam_cameramatrix;
	cv::Mat cam_distortioncoefficients;
	fs["distortion_coefficients"] >> cam_distortioncoefficients;

	//cout << cam_cameramatrix;
	fs.release();

	// OPEN LIVE CAMERA 

	VideoCapture cap;
	// open the default camera using default API
	// cap.open(0);
	// OR advance usage: select any API backend
	int deviceID = 0;             // 0 = open default camera
	int apiID = cv::CAP_DSHOW;      
	// open selected camera using selected API
	cap.open(deviceID, apiID);
	// check if we succeeded
	if (!cap.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		return -1;
	}



	// SECTION HOLDING ALL THE VARIABLES

	// GUI flags
	bool gui_showTrainingArea = true; // show training area (enable with 1) 
	bool gui_showAllCamFeatures = false; // show all features tracked on the camera (enable with 2)
	bool gui_showMatchedFeatures = false; // show all the matched features (enable with 3)

	// Image of the live camera frame
	Mat img_liveframe;
	// Image that will show the matches between the camera and the training imag
	Mat img_final;

	// Features will be detected using this
	Ptr<SIFT> siftdetector = SIFT::create();

	// these will hold the TRAINING IMAGE features keypoints and their descriptors
	std::vector<cv::KeyPoint> fkeypoints_train;
	cv::Mat fdescriptors_train;

	// these will hold the LIVE CAMERA IMAGE features keypoints and their descriptors
	std::vector<cv::KeyPoint> fkeypoints_cam;
	cv::Mat fdescriptors_cam;


	// Matching between the two image sets of features will be done using this
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	std::vector<DMatch> topMatches;


	
	
	// this is the AR cube that will show on the tracked image
	std::vector<cv::Point3f> cube3dpoints(8);
	cube3dpoints[0] = (Point3f(0, 0, 0));
	cube3dpoints[1] = (Point3f(-0.1, 0, 0));
	cube3dpoints[2] = (Point3f(-0.1, 0.1, 0));
	cube3dpoints[3] = (Point3f(0, 0.1, 0));

	cube3dpoints[4] = (Point3f(0, 0, 0.2));
	cube3dpoints[5] = (Point3f(-0.1, 0, 0.2));
	cube3dpoints[6] = (Point3f(-0.1, 0.1, 0.2));
	cube3dpoints[7] = (Point3f(0, 0.1, 0.2));

	// scale all the cube points
	for (int i = 0; i < cube3dpoints.size(); i++) {
		cube3dpoints[i] = Point3f(cube3dpoints[i].x *-1, cube3dpoints[i].y * 1, cube3dpoints[i].z * -1  + 1);
	}



	// area of the camera image used for training the new dataset
	Rect trainingArea;
	{
		int pad = 50, width = 300, height = 300;
		int x1 = pad, x2 = pad + width;
		int y1 = pad, y2 = pad + height;
		trainingArea = Rect(x1, y1, x2, y2);
	}


	// whether we have any training data
	bool haveTrained = false;
	cv::Mat img_training;


	/***
	// LOAD THE DEFAULT TRAINING IMAGE FROM FILE
	// this can be changed at runtime from the live camera

	String filename1 = "..\\..\\_RawImages\\box.png";

	cv::Mat img_training = cv::imread(filename1, cv::IMREAD_GRAYSCALE); //Load as grayscale
	assert(!img_training.empty());

	processNewTrainingImage(img_training, siftdetector, fkeypoints_train, fdescriptors_train);
	****/
	



	// MAIN LOOP

	cout << "Starting grabbing frames" << endl
		<< "Press any key to terminate" << endl;
	for (;;)
	{
		
		// wait for a new frame from camera and store it into 'frame'
		cap.read(img_liveframe);
		// check if we succeeded
		if (img_liveframe.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}
		flip(img_liveframe, img_liveframe, 1);


		// KEYBOARD CONTROLS

		char c = (char)waitKey(1);

		// '1' - show the area where we will be training from
		if (c == '1') {
			gui_showTrainingArea = !gui_showTrainingArea;
		}
		if (gui_showTrainingArea) train_drawArea(img_liveframe, trainingArea, Scalar(0, 255, 0));;


		// '2' - show all the features in the camera image
		if (c == '2') {
			gui_showAllCamFeatures = !gui_showAllCamFeatures;
		}

		// '3' - show all the features in the camera image
		if (c == '3') {
			gui_showMatchedFeatures = !gui_showMatchedFeatures;
		}

		// ' ' - trigger training of new dataset from that area
		if (c == ' ') {
			train_drawArea(img_liveframe, trainingArea, Scalar(255, 0, 0));

			Mat imgCroppedCamera = img_liveframe(trainingArea);
			cvtColor(imgCroppedCamera, img_training, COLOR_RGB2GRAY);
			
			processNewTrainingImage(img_training, siftdetector, fkeypoints_train, fdescriptors_train);

			haveTrained = true;
			// turn off training area
			gui_showTrainingArea = false;
		}

		// show live , will be a bit later
		//imshow("Live", img_liveframe);





		// NOW DO AR BASED ON THE TRAINING DATA VS. THE CAMERA IMAGE

		

		// 
		// ===== DETECT FEATURES IN THE CAMERA FRAME ======= using SIFT
		// 
		siftdetector->detectAndCompute(img_liveframe, noArray(), fkeypoints_cam, fdescriptors_cam);

		// show all the features found in the camera frame, regardless of whether they're in the training or not
		if (gui_showAllCamFeatures) {
			cv::Mat output;
			cv::drawKeypoints(img_liveframe, fkeypoints_cam, img_liveframe);//, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			//imshow("All features in camera", output);
		}

		// show live 
		if (!haveTrained) {
			putText(img_liveframe, "Place an image in the training area and push SPACE to train", 
				Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.4, 0);
		}
		else {
			putText(img_liveframe, "KEYS: 1=Train Area; SPACE=Train;   2=Features; 3=Matches",
				Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.4, 0);
		}
		imshow("Live", img_liveframe);



		//
		// ===== MATCHING THE FEATURES ======= with knn, and filtering
		//

		// if we didn't train any tracking image, then can't do anything now. 
		if (!haveTrained) continue;


		knn_matches.clear();

			// match the featuers from the camera and the training 
			matcher->knnMatch(fdescriptors_cam, fdescriptors_train, knn_matches, 2);

			// Filter only good enough matches
			topMatches = filterFeatureMatches(knn_matches);

			// Extract the original eature point locations within each image (will be needed for homography)
			std::vector<Point2f> matchedpoints_train;
			std::vector<Point2f> matchedpoints_cam;
			matchedpoints_train.clear();
			matchedpoints_cam.clear();
			for (size_t i = 0; i < topMatches.size(); i++)
			{
				//-- Get the keypoints from the good matches and locate them on the two images
				matchedpoints_train.push_back(fkeypoints_train[topMatches[i].trainIdx].pt);
				matchedpoints_cam.push_back(fkeypoints_cam[topMatches[i].queryIdx].pt);
			}


			// SHOW the matches in the two images
			if (gui_showMatchedFeatures) {

				// draw matches on the liveframe and output to img_final
				drawMatches(img_liveframe, fkeypoints_cam,
					img_training, fkeypoints_train,
					topMatches,
					img_final, 
					Scalar::all(255),
					Scalar::all(255), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				putText(img_final, "TRAINING IMAGE",
					Point(img_liveframe.cols + 10, 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255));

				//imshow("Features Matched between Live and Training image", img_matches);


				// do some extra stats on the matching
				int num_good_matches = topMatches.size();
				float avg_matchDist = 0;
				for (size_t i = 0; i < num_good_matches; i++)
				{
					avg_matchDist += topMatches[i].distance;
				}
				avg_matchDist /= num_good_matches;

				
				cout << String("Matching stats. Num good = ") << num_good_matches << "\t  Avg = " << avg_matchDist
					<< endl;

				
			}
			else {
				// we'll just use the live image to build the img_final
				img_final = img_liveframe;
			}

			// checkpoint to make sure we have enough match between the two images
			const int MINPOINTS = 20;
			if (matchedpoints_cam.size() < MINPOINTS) continue;

			
			//
			// ===== POSE FINDING ======= with solvePnP
			//

			// need to scale the points a bit
			vector<Point3f> matchedpoints_train_new;
			for (Point2f p : matchedpoints_train) {
				const float SCALE = 1000;
				matchedpoints_train_new.push_back(Point3f(p.x / SCALE, p.y / SCALE, 1));
			}

			Mat rvec, tvec;
			solvePnPRansac(matchedpoints_train_new, matchedpoints_cam, cam_cameramatrix, cam_distortioncoefficients, rvec, tvec);


			


			// NOW PROJECT THE AR CUBE FROM OBJECT SPACE ONTO THE TRACKED IMAGE SPACE

			std::vector<cv::Point2f> projectedPoints(2);

			// calculate where the 3d cube vertices will be in the camera image space
			cv::projectPoints(cube3dpoints, rvec, tvec, cam_cameramatrix, cam_distortioncoefficients, projectedPoints);
			// draw circles where the cube vertices are in the camera space
			cv::circle(img_final, projectedPoints[0], 5, (255, 0, 0), 3);
			// and draw the cube edges
			drawCube(projectedPoints, img_final);
			

			imshow("Final", img_final);
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
