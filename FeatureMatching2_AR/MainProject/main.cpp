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

/***

int main__images(int argc, char** argv)
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

*****/


void train_drawArea(Mat frame, Rect trainingArea, Scalar color) {
	
	cv::rectangle(frame, trainingArea, color);
}

void processNewTrainingImage(Mat img1, Ptr<SIFT>& siftdetector, std::vector<cv::KeyPoint>& keypoints1, Mat& descriptors1, std::vector<Point2f>& obj_corners) {
	keypoints1.clear();
	siftdetector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	obj_corners.clear();
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img1.cols, 0);
	obj_corners[2] = Point2f((float)img1.cols, (float)img1.rows);
	obj_corners[3] = Point2f(0, (float)img1.rows);
}

int main(int, char**)
{

	// CAMERA MATRIX FROM FILE

	FileStorage fs("..\\..\\_RawImages\\camera.yml", FileStorage::READ);
	cv::Mat cam_cameramatrix; 
	fs["camera_matrix"] >> cam_cameramatrix;
	cv::Mat cam_distortioncoefficients;
	fs["distortion_coefficients"] >> cam_distortioncoefficients;

	cout << cam_cameramatrix;
	fs.release();

	//// TRAINING DATA

	String filename1 = "..\\..\\_RawImages\\box.png";

	cv::Mat img1 = cv::imread(filename1, cv::IMREAD_GRAYSCALE); //Load as grayscale
	assert(!img1.empty());

	Ptr<SIFT> siftdetector = SIFT::create();
	std::vector<cv::KeyPoint> keypoints1;
	cv::Mat descriptors1;
	



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

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;
	std::vector<DMatch> topMatches;

	std::vector<Point2f> objPoints;
	std::vector<Point2f> scenePoints;
	Mat img_matches;

	std::vector<Point2f> obj_corners(4);

	Mat H;
	std::vector<Point2f> scene_corners(4);
	
	std::vector<cv::Point3f> sps(8);
	sps[0] = (Point3f(0, 0, 0));
	sps[1] = (Point3f(-0.1, 0, 0));
	sps[2] = (Point3f(-0.1, 0.1, 0));
	sps[3] = (Point3f(0, 0.1, 0));

	sps[4] = (Point3f(0, 0, 0.2));
	sps[5] = (Point3f(-0.1, 0, 0.2));
	sps[6] = (Point3f(-0.1, 0.1, 0.2));
	sps[7] = (Point3f(0, 0.1, 0.2));

	// scale all 
	for (int i = 0; i < sps.size(); i++) {
		sps[i] = Point3f(sps[i].x *-1, sps[i].y * 1, sps[i].z * -1  + 1);
	}

	Rect trainingArea;
	{
		int pad = 50, width = 300, height = 300;
		int x1 = pad, x2 = pad + width;
		int y1 = pad, y2 = pad + height;
		trainingArea = Rect(x1, y1, x2, y2);
	}
	processNewTrainingImage(img1, siftdetector, keypoints1, descriptors1, obj_corners);

	bool gui_showArea = false;
	bool gui_showFeatures = false;

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



		char c = (char)waitKey(1);

		if (c == '1') {
			gui_showArea = !gui_showArea;
		}
		if (gui_showArea) train_drawArea(frame, trainingArea, Scalar(0, 255, 0));;

		if (c == ' ') {
			// train new dataset
			train_drawArea(frame, trainingArea, Scalar(255, 0, 0));

			Mat ni = frame(trainingArea);
			cvtColor(ni, img1, COLOR_RGB2GRAY);
			
			processNewTrainingImage(img1, siftdetector, keypoints1, descriptors1, obj_corners);

		}




		// show live and wait for a key with timeout long enough to show images
		imshow("Live", frame);

		// 
		// ===== DETECT FEATURES ======= using SIFT
		// 
		siftdetector->detectAndCompute(frame, noArray(), keypoints2, descriptors2);

		// SHOW keypoints
		if (false) {
			cv::drawKeypoints(frame, keypoints2, output);//, cv::Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			imshow("output", output);
		}



		//
		// ===== MATCHING THE FEATURES ======= with knn, and filtering
		//


		knn_matches.clear();

			matcher->knnMatch(descriptors2, descriptors1, knn_matches, 2);

			// Filter only good enough matches
			topMatches = filterFeatureMatches(knn_matches);

			// Extract the original point locations within each image (will be needed for homography)
			objPoints.clear();
			scenePoints.clear();
			for (size_t i = 0; i < topMatches.size(); i++)
			{
				//-- Get the keypoints from the good matches
				objPoints.push_back(keypoints1[topMatches[i].trainIdx].pt);
				scenePoints.push_back(keypoints2[topMatches[i].queryIdx].pt);
			}


			// SHOW matches
			if (gui_showFeatures) {
				drawMatches(frame, keypoints2,
					img1, keypoints1,
					topMatches,
					img_matches, 
					Scalar::all(255),
					Scalar::all(255), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

				if (false) imshow("Good Matches", img_matches);
			}
			else {
				img_matches = frame;
			}


			const int MINPOINTS = 20;
			if (scenePoints.size() < MINPOINTS) continue;

			/***
			
			//
			// ===== POSE FINDING ======= with findHomography
			//

			// Figure out how the original points were moved to the scene


			

					H = findHomography(objPoints, scenePoints, RANSAC);
					if (H.dims == 0 || H.cols == 0) continue;


					// Now draw the original rectangle into the scene
					scene_corners.clear();
					perspectiveTransform(obj_corners, scene_corners, H);

					// SHOW the rectangle in the scene
					line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
					line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

					//imshow("Good Matches & Object detection", img_matches);



					// GO TO 3D

					vector<Mat> Rs_decomp, ts_decomp, normals_decomp;
					int solutions = decomposeHomographyMat(H, cam_cameramatrix, Rs_decomp, ts_decomp, normals_decomp);
					cout << "Decompose homography matrix computed from the camera displacement:" << endl << endl;
					for (int i = 0; i < solutions; i++)
					{
						double factor_d1 = 1.0; //    / d_inv1;
						Mat rvec_decomp;
						Rodrigues(Rs_decomp[i], rvec_decomp);

						cout << "Solution " << i << ":" << endl;

						if ( countNonZero(rvec_decomp == 0)) { cout << "BAD" << endl; continue; }

						cout << "rvec from homography decomposition: " << rvec_decomp.t() << endl;
						//cout << "rvec from camera displacement: " << rvec_1to2.t() << endl;
						cout << "tvec from homography decomposition: " << ts_decomp[i].t() << " and scaled by d: " << factor_d1 * ts_decomp[i].t() << endl;
						//cout << "tvec from camera displacement: " << t_1to2.t() << endl;
						cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << endl;
						//cout << "plane normal at camera 1 pose: " << normal1.t() << endl << endl;

						std::vector<cv::Point3d> sps(8);
						sps[0] = (Point3d(0, 0,1));
						sps[1] = (Point3d(-0.1, 0,1));
						sps[2] = (Point3d(0, 0.1, 1));
						sps[3] = (Point3d(-0.1, 0.1, 1));
						sps[4] = (Point3d(0, 0, 1.1));
						sps[5] = (Point3d(-0.1, 0, 1.1));
						sps[6] = (Point3d(0, 0.1, 1.1));
						sps[7] = (Point3d(-0.1, 0.1, 1.1));

						std::vector<cv::Point2d> projectedPoints(2);

						cout << "PROJECTING " << endl;
						cv::projectPoints(sps,rvec_decomp.t(), ts_decomp[i].t(), cam_cameramatrix, cam_distortioncoefficients, projectedPoints);
						
						cout << "DRAWING" << endl;

						cv::line(img_matches, projectedPoints[0], projectedPoints[1], (0, 0, 255), 3);
						for (int i = 0; i < sps.size(); i++) {
							cv::circle(img_matches, projectedPoints[i], 5, (255, 0, 0), 3);
						}
						cout << "DONE LINES" << endl;
					}


					imshow("Good Matches & Object detection", img_matches);

			*****/


			//
			// ===== POSE FINDING ======= with solvePnP
			//

			vector<Point3f> objPoints3;
			for (Point2f p : objPoints) {
				const float SCALE = 1000;
				objPoints3.push_back(Point3f(p.x / SCALE, p.y / SCALE, 1));
			}

			Mat rvec, tvec;
			solvePnPRansac(objPoints3, scenePoints, cam_cameramatrix, cam_distortioncoefficients, rvec, tvec);


			

			std::vector<cv::Point2f> projectedPoints(2);

			cv::projectPoints(sps, rvec, tvec, cam_cameramatrix, cam_distortioncoefficients, projectedPoints);

			cv::circle(img_matches, projectedPoints[0], 5, (255, 0, 0), 3);
			
			drawCube(projectedPoints, img_matches);
			

			imshow("Good Matches & Object detection", img_matches);
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
