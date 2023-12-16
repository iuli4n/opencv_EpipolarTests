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
		cout << "ERROR. Failed to open image file.\n";
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
	String filename = "..\\..\\_RawImages\\box.png";
	checkFileExists(filename);

	// Read the image file
	Mat image = imread(filename);

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