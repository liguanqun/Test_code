/*
 * test.cpp
 *
 *  Created on: Jun 17, 2018
 *      Author: orbbec
 */
#include "image_acquisition.hpp"
#include <iostream>
#include <ctype.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <fstream>
#include <stdio.h>

#include "DepthSegmenter.hpp"
#include "math_helper.hpp"
#include "json/json.h"
using namespace std;

/***************测试分割效果********************/

/*int main()
	{
		cv::Mat RGB, depth, result;
		int _frameIdx = 1;
		cv::namedWindow("depth", 0);
		cv::namedWindow("RGB", 0);

		ImageAcquisition ic;
		ic.Init();
		ic.Get_RGB_Image(0, RGB);
		ic.Get_Depth_Image(0, depth);
		cv::Rect rect = ic.Get_Current_GroundTruth_Rect();

		DepthSegmenter Ds;
		Ds.init(depth, rect);
		result = Ds._result;

		stringstream stream;
		stream << "seg_reslt_" << _frameIdx << ".jpg";

		cv::imwrite(stream.str(), result);

		cv::rectangle(RGB, rect, cv::Scalar(0, 0, 255), 2);
		cv::imshow("RGB", RGB);
		cv::imshow("depth", depth);
		for (;;)
			{
				_frameIdx++;
				ic.Get_RGB_Image(1, RGB);
				ic.Get_Depth_Image(1, depth);
				rect = ic.Get_Current_GroundTruth_Rect();

				Ds.update(depth, rect);
				result = Ds._result;
				stringstream stream;
				stream << "seg_reslt_" << _frameIdx << ".jpg";
				cv::imwrite(stream.str(), result);

				cv::rectangle(RGB, rect, cv::Scalar(0, 0, 255), 2);
				cv::imshow("RGB", RGB);
				cv::imshow("depth", depth);

				cv::waitKey(33);
			}

		return 0;
	}*/
/**************测试分割效果********************/

/*void readFileJson(); //从文件中读取JSON，一个存储了JSON格式字符串的文件

 int main(int argc, char *argv[])
 {
 readFileJson();

 return 0;
 }

 //从文件中读取JSON
 void readFileJson()
 {
 Json::Reader reader;
 Json::Value root;

 //从文件中读取，保证当前文件有test.json文件
 ifstream in("frames.json", ios::binary);

 if (!in.is_open())
 {
 cout << "Error opening file\n";
 return;
 }

 if (reader.parse(in, root))
 {
 //读取根节点信息
 string path = root["path"].asString();
 cout << "My path is " << path << endl;

 cout << "Here's my achievements:" << endl;
 for (unsigned int i = 0; i < root["depthFrameID"].size(); i++)
 {
 int ach = root["depthFrameID"][i].asInt();
 cout << ach << "  ";
 }
 cout << endl;
 }
 else
 {
 cout << "parse error\n" << endl;
 }

 in.close();
 }*/
int main()
	{
		ImageAcquisition ic;
		ic.Init();
		return 0;
	}
