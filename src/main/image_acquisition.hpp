
#ifndef IMAGE_ACQUISITION_HPP_
#define IMAGE_ACQUISITION_HPP_

#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>
#include<unistd.h>
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<sys/types.h>
#include<dirent.h>
#include<map>

class ImageAcquisition
	{
	public:
		ImageAcquisition();
		virtual ~ImageAcquisition();
		void Init();


		cv::Mat Get_first_RGB();
		cv::Mat Get_Next_RGB();

		cv::Mat Get_Depth_Image_same_time_to_RGB();

        cv::Rect Get_Init_Rect(void);

        cv::Rect Get_Current_GroundTruth_Rect(void);

		bool Get_Depth_Image(int k,cv::Mat& image);
		bool Get_RGB_Image(int k, cv::Mat& image);

		void Get_Time_And_K(std::string str, int & t, int & k);
		cv::Mat Shift_Bit_Depth_Image(cv::Mat& image);

        std::string _name;
       // cv::Rect ImageAcquisition::Get_Groundtruth(int k);
	private:
        const std::string _path;
		std::map<int, std::string> _FrameID_path, _FrameID_path_depth;
		std::map<int, cv::Rect> _FrameID_rect;
		std::map<int, int> _FrameID_t, _FrameID_t_depth;
		std::map<int, int> _RGB_DEPTH_ID;
		int _size, _rgb_FrameID, _depth_FrameID;

	};

#endif
