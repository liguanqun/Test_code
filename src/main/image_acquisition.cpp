#include "image_acquisition.hpp"
#include <iostream>
#include <ctype.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <fstream>
#include <stdio.h>
#include "json/json.h"
ImageAcquisition::ImageAcquisition()
		: _path("/home/orbbec/Downloads/data/ValidationSet/bear_front")
	{
//遮挡物特别大，遮挡之后的恢复问题
//cc_occ1  express1_occ express2_occ  new_ex_occ1 new_ex_occ3  rose1.2"
		_rgb_FrameID = 1;
		_depth_FrameID = 1;
		//  /home/orbbec/Downloads/data/ValidationSet/child_no1/
		//zcup_move_1  bear_front child_no1 new_ex_occ4 face_occ5

		//  /media/orbbec/7024AED824AEA118/EvaluationSet/bag1/
		// bag1 basketball1 basketball2(效果不好) basketball2.2 basketballnew bdog_occ2
		//bear_back bear_change bird1.1_no  bird3.1_no book_move1 book_turn book_turn2
		//box_no_occ  br_occ1 br_occ_0 br_occ_turn0 cafe_occ1 cc_occ1 cf_difficult cf_no_occ
		//cf_occ2 cf_occ3 computerbar1 computerBar2 cup_book dog_no_1 dog_occ_2 dog_occ_3
		//express1_occ express2_occ express3_static_occ face_move1 face_occ2  face_occ3
		//face_turn2 flower_red_occ gre_book  hand_no_occ hand_occ  library2.1_occ library2.2_occ
		//mouse_no1 new_ex_no_occ new_ex_occ1 new_ex_occ2 new_ex_occ3 new_ex_occ5_long new_ex_occ6 new_ex_occ7.1
		//new_student_center1 new_student_center2 new_student_center3 new_student_center4  new_student_center_no_occ
		//new_ye_no_occ new_ye_occ one_book_move rose1.2 static_sign1 studentcenter2.1 studentcenter3.1 studentcenter3.2
		//three_people toy_car_no toy_car_occ toy_green_occ toy_mo_occ toy_no   toy_no_occ toy_wg_no_occ toy_wg_occ toy_wg_occ1
		//toy_yellow_no tracking4 tracking7.1 two_book two_dog_occ1 two_people_1.1 two_people_1.2 two_people_1.3
		//walking_no_occ walking_occ1 walking_occ_long  wdog_no1 wdog_occ3 wr_no  wr_no1    wr_occ2 wuguiTwo_no zball_no1 zball_no2 zball_no2
		//zballpat_no1

		//_RGB_path = "/home/orbbec/Downloads/data/ValidationSet/child_no1/rgb/";
		//_DEPTH_path = "/home/orbbec/Downloads/data/ValidationSet/child_no1/depth/";
	}
void ImageAcquisition::Init()
	{
		/*******************读RGB图*************************/
		int t, k;
		DIR *dp;
		struct dirent *dirp;
		std::string rgb_path= _path+"/rgb/";
		if ((dp = opendir(rgb_path.c_str())) == NULL)
			{
				perror("opendir error");
				exit(1);
			}

		while ((dirp = readdir(dp)) != NULL)
			{

				if ((strcmp(dirp->d_name, ".") == 0) || (strcmp(dirp->d_name, "..") == 0))
					continue;

				char dirname[100];
				std::string tmp = _path + "/rgb/";
				tmp.copy(dirname, tmp.size(), 0);
				dirname[tmp.size()] = '\0';

				Get_Time_And_K(dirp->d_name, t, k);
				_FrameID_path[k] = strcat(dirname, dirp->d_name);

				_FrameID_t[k] = t;
				//_FrameID_path.insert(std::make_pair<int,std::string>(k, strcat(dirname, dirp->d_name)));
				//_FrameID_t.insert(std::make_pair<int,int>(k,t));

			}

		/**************************读深度图*******************************/

		DIR *dp_d;
		struct dirent *dirp_d;
		std::string depth_path= _path+"/depth/";
		if ((dp_d = opendir(depth_path.c_str())) == NULL)
			{
				perror("opendir error");
				exit(1);
			}
		while ((dirp_d = readdir(dp_d)) != NULL)
			{

				if ((strcmp(dirp_d->d_name, ".") == 0) || (strcmp(dirp_d->d_name, "..") == 0))
					continue;

				char dirname[100];
				std::string tmp = _path + "/depth/";
				tmp.copy(dirname, tmp.size(), 0);
				dirname[tmp.size()] = '\0';

				Get_Time_And_K(dirp_d->d_name, t, k);

				_FrameID_path_depth[k] = strcat(dirname, dirp_d->d_name);
				_FrameID_t_depth[k] = t;
				//_FrameID_path_depth.insert(std::make_pair<int,std::string>(k, strcat(dirname, dirp_d->d_name)));
				//_FrameID_t_depth.insert(std::make_pair<int,int>(k,t));

			}

		_size = std::min(_FrameID_path_depth.size(), _FrameID_path.size());
		std::cout << "total image is " << _size << std::endl;

		/******************读groundtruth***************************/
		using namespace std;

		std::string name, path;
		name = _path.substr(_path.find_last_of('/') + 1, _path.size());
		std::cout << "current image set is  " << name << std::endl;
		_name = name;
		path = _path + "/" + name + ".txt";

		ifstream myfile(path.c_str());
		if (!myfile.is_open())
			{
				cout << "can not open the init file" << endl;
			}

		string temp;

		while (getline(myfile, temp))
			{
				cv::Rect r;
				r.x = atoi(temp.substr(0, temp.find_first_of(',')).c_str());
				temp.erase(0, temp.find_first_of(',') + 1);

				r.y = atoi(temp.substr(0, temp.find_first_of(',')).c_str());
				temp.erase(0, temp.find_first_of(',') + 1);

				r.width = atoi(temp.substr(0, temp.find_first_of(',')).c_str());
				temp.erase(0, temp.find_first_of(',') + 1);

				r.height = atoi(temp.substr(0, temp.find_first_of(',')).c_str());
				temp.erase(0, temp.find_first_of(',') + 1);

				int k = atoi(temp.c_str());
				_FrameID_rect[k] = r;
				//	std::cout<<"rect .x y width height  k  == "<<r.x<<"  "<<r.y<<"  "<<r.width<<"  "<<r.height<<"  "<<k<<std::endl;

			}
		myfile.close();   //关闭文件
		/**********读frames.json******************/
		std::string path_JSON = _path + "/frames.json";

		Json::Reader reader;
		Json::Value root;

		//从文件中读取，保证当前文件有*.json文件
		ifstream in(path_JSON, ios::binary);

		if (!in.is_open())
			{
				cout << "Error opening JSON file\n";
				return;
			}

		if (reader.parse(in, root))
			{

				for (unsigned int i = 0; i < root["depthFrameID"].size(); i++)
					{
						int ach = root["depthFrameID"][i].asInt();
						_RGB_DEPTH_ID[i + 1] = ach;
					}
			}
		else
			{
				cout << "JSON file parse error\n" << endl;
			}

	}
cv::Mat ImageAcquisition::Get_first_RGB()
	{
		cv::Mat image = cv::imread(_FrameID_path[_rgb_FrameID]);
		_rgb_FrameID++;
		return image;

	}
cv::Mat ImageAcquisition::Get_Next_RGB()
	{
		cv::Mat image;
		if (_rgb_FrameID < _size)
			{
				image = cv::imread(_FrameID_path[_rgb_FrameID]);
				_rgb_FrameID++;

			}
		else
			{
				std::cout << "there is no RGB image" << std::endl;
			}
		return image;

	}
bool ImageAcquisition::Get_RGB_Image(int k, cv::Mat& image)
	{

		if (k == 0)
			{
				image = cv::imread(_FrameID_path[1]);
				double time_diff = ((double) (_FrameID_t_depth[1] - _FrameID_t[1])) / 33333;
				std::cout << "the " << _rgb_FrameID << " frame RGB and depth time diff is " << time_diff << std::endl;
				//	char k; std::cin>>k;
				//std::cout << "*******************************************************"<< std::endl;

				return true;
			}

		else if (k == 1)
			{
				_rgb_FrameID += 1;
				if (_rgb_FrameID < _size)
					{
						image = cv::imread(_FrameID_path[_rgb_FrameID]);

						double time_diff = ((double) (_FrameID_t_depth[_rgb_FrameID] - _FrameID_t[_rgb_FrameID])) / 33333;
						std::cout << "the " << _rgb_FrameID << " frame RGB and depth time diff is " << time_diff << std::endl;
						//		char k; std::cin>>k;
						//std::cout << "*******************************************************"<< std::endl;
						return true;
					}
				else
					return false;
			}
		else
			return false;
	}
cv::Rect ImageAcquisition::Get_Current_GroundTruth_Rect(void)
	{
		cv::Rect r = _FrameID_rect[_rgb_FrameID];
		return r;
	}
bool ImageAcquisition::Get_Depth_Image(int k, cv::Mat& image)
	{

		if (k == 0)
			{
				image = cv::imread(_FrameID_path_depth[1], CV_LOAD_IMAGE_ANYDEPTH);
				image = Shift_Bit_Depth_Image(image);

				return true;
			}
		else if (k == 1)
			{
				_depth_FrameID += 1;
				if (_depth_FrameID < _size)
					{

						image = cv::imread(_FrameID_path_depth[_depth_FrameID], CV_LOAD_IMAGE_ANYDEPTH);
						image = Shift_Bit_Depth_Image(image);
						return true;
					}
				else
					{
						std::cout << "there is no depth image " << std::endl;
						return false;
					}
			}
		else
			return false;
	}
cv::Mat ImageAcquisition::Get_Depth_Image_same_time_to_RGB()
	{
		int ask = _RGB_DEPTH_ID[_rgb_FrameID];
		cv::Mat depth;
		if (ask < _size)
			{

				depth = cv::imread(_FrameID_path_depth[ask], CV_LOAD_IMAGE_ANYDEPTH);
				depth = Shift_Bit_Depth_Image(depth);

			}
		else
			{
				ask = 0;
			}
		return depth;

	}
void ImageAcquisition::Get_Time_And_K(std::string str, int & t, int & k)
	{
		std::string abc = str;

		abc.erase(0, 2);

		int pose = abc.find("-");
		t = atoi(abc.substr(0, pose).c_str());

		abc.erase(0, pose + 1);

		pose = abc.find(".");
		k = atoi(abc.substr(0, pose).c_str());

	}
cv::Mat ImageAcquisition::Shift_Bit_Depth_Image(cv::Mat& image)
	{

		/*		cv::Mat image_show;
		 image.copyTo(image_show);
		 //cv::rectangle(image_show, Get_Init_Rect, cv::Scalar(255), 2);
		 //cv::rectangle(image_show, Get_Init_Rect,cv::Scalar(255,0,0));
		 cv::Rect rect = Get_Init_Rect();
		 cv::rectangle(image_show,rect.tl(),rect.br(),cv::Scalar(255));
		 cv::imshow("depth_rectangle",image_show);*/
		cv::Mat image_2(image.rows, image.cols, image.type(), cv::Scalar(0));

//image.copyTo(image_2);
		image_2 = image.clone();
		/*		image.copyTo(image_2);
		 int nl = image.rows;
		 int nc = image.cols;

		 for (int j = 0; j < nl; j++)
		 {
		 //unsigned short *pt = image_2.ptr<unsigned short>(j);

		 unsigned short *pt = image.ptr<unsigned short>(j);
		 unsigned short *ptt = image_2.ptr<unsigned short>(j);
		 //unsigned short temp;
		 for (int i = 0; i < nc; i++)
		 {
		 //对像素进行某些操作
		 ptt[i] = (unsigned short) (pt[i] / 8);
		 }
		 }*/

		for (int i = 0; i < image.rows; i++)
			{
				for (int j = 0; j < image.cols; j++)
					{

						unsigned short temp = image.at<unsigned short>(i, j);
						//std::cout<<temp<<" ";
						temp = temp / 8 + (0xffff & (temp << 13));
						//temp = temp /8 ;
						image_2.at<unsigned short>(i, j) = temp;
					}
			}
		std::cout << std::endl;

		return image_2;
	}
cv::Rect ImageAcquisition::Get_Init_Rect(void)
	{
		using namespace std;
		cv::Rect r;
		std::string path;
		path = _path + "/init.txt";

//	std::cout << "path == " << path << std::endl;
		ifstream myfile(path.c_str());
		if (!myfile.is_open())
			{
				cout << "can not open the init file" << endl;
			}
//std::cout << "read finish" << std::endl;

		string temp;
		getline(myfile, temp);
		myfile.close();   //关闭文件

//std::cout << temp << std::endl;

//std::cout<<temp.substr(0,temp.find_first_of(',')).c_str()<<std::endl;

		r.x = atoi(temp.substr(0, temp.find_first_of(',')).c_str());
		temp.erase(0, temp.find_first_of(',') + 1);

		r.y = atoi(temp.substr(0, temp.find_first_of(',')).c_str());
		temp.erase(0, temp.find_first_of(',') + 1);

		r.width = atoi(temp.substr(0, temp.find_first_of(',')).c_str());
		temp.erase(0, temp.find_first_of(',') + 1);

		r.height = atoi(temp.c_str());

//std::cout << "r.x y width height " << r.x << "*" << r.y << "*" << r.width << "*" << r.height << std::endl;

		return r;
	}

/*cv::Rect ImageAcquisition::Get_Groundtruth(int k)
 {

 cv::Rect r;
 return r;
 }*/
ImageAcquisition::~ImageAcquisition()
	{

	}
