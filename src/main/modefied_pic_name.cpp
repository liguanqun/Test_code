/*
 * modefied_pic_name.cpp
 *
 *  Created on: Jun 11, 2018
 *      Author: orbbec
 */

#include "opencv2/highgui/highgui.hpp"
#include "image_acquisition.hpp"
#include <string>
#include <sstream>
int main()
	{
		ImageAcquisition ic;
		ic.Init();
		cv::Mat src;
		ic.Get_RGB_Image(0, src);
		cv::imwrite("0001.jpg", src);
		int k = 1;

		while (ic.Get_RGB_Image(1, src))
			{
				k++;

				if (k < 10)
					{
						std::stringstream ss;
						std::string s;
						ss << "000" << k << ".jpg";
						ss >> s;
						cv::imwrite(s, src);
					}
				else if (k < 100)
					{
						std::stringstream ss;
						std::string s;
						ss << "00" << k << ".jpg";
						ss >> s;
						cv::imwrite(s, src);

					}
				else if (k < 1000)
					{
						std::stringstream ss;
						std::string s;
						ss << "0" << k << ".jpg";
						ss >> s;
						cv::imwrite(s, src);

					}
				else
					{
						std::cout << "error is occ" << std::endl;
					}
			}

		return 0;

	}
