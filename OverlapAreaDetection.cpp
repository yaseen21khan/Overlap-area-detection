#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/calib3d.hpp"


//#include <opencv2/xfeatures2d.hpp>

// stitching_project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

using namespace std;
using namespace cv;


Mat drawMatches2Imgaes(Mat mat1, Mat mat2);

int main()
{
   
	Mat image, img_, img, img1, img2;
	Mat result_Mat;
	vector<Mat> images;
	String folder = "E:/Android_Data/images/Real_time_pano/*.jpg";
	vector<String> filenames;
	glob(folder, filenames);
	cout << filenames.size() << endl; //to display no of files

	for (size_t i = 0; i < filenames.size(); ++i)
	{
		cout << filenames[i] << endl;
		image = imread(filenames[i]);
		images.push_back(image);
	}

	//img_ = images[0];
	//img = images[1];

	//cvtColor(img_, img1, COLOR_BGR2GRAY);
	//cvtColor(img, img2, COLOR_BGR2GRAY);
	//
	////int w = img_.cols * 0.1;
	////int h = img_.rows * 0.1;

	////int w1 = img.cols * 0.1;
	////int h1 = img.rows * 0.1;

	////resize(img_, img_, Size(w, h), INTER_AREA);
	////resize(img, img, Size(w1, h1), INTER_AREA);

	Mat img_matches = drawMatches2Imgaes(images[0], images[1]);
	namedWindow("Good Matches & Object detection", WINDOW_GUI_NORMAL);
	imshow("Good Matches & Object detection", img_matches);
	
	waitKey(0);
	return 0;
}

Mat drawMatches2Imgaes(Mat mat1, Mat mat2) {

	Mat img_ = mat1;
	Mat img = mat2;

	Ptr<Feature2D> sift_finder = AKAZE::create();
	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	sift_finder->detectAndCompute(img_, noArray(), keypoints1, descriptors1);
	sift_finder->detectAndCompute(img, noArray(), keypoints2, descriptors2);

	BFMatcher matcher(NORM_HAMMING);
	vector< vector<DMatch> > matches;
	vector<cv::DMatch> good_matches;
	matcher.knnMatch(descriptors1, descriptors2, matches, 2);

	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.8; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
	}

	Mat img_matches;
	drawMatches(img_, keypoints1, img, keypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, RANSAC);
	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point2f(0, 0);
	obj_corners[1] = Point2f((float)img_.cols, 0);
	obj_corners[2] = Point2f((float)img_.cols, (float)img_.rows);
	obj_corners[3] = Point2f(0, (float)img_.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f((float)img_.cols, 0), scene_corners[1] + Point2f((float)img_.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f((float)img_.cols, 0), scene_corners[2] + Point2f((float)img_.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f((float)img_.cols, 0), scene_corners[3] + Point2f((float)img_.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f((float)img_.cols, 0), scene_corners[0] + Point2f((float)img_.cols, 0), Scalar(0, 255, 0), 4);
	//-- Show detected matches
	cout << H;
	return img_matches;

}
