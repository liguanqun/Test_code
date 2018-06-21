#include <iostream>

#include "DepthSegmenter.hpp"
#include "math_helper.hpp"
#include "tbb/tick_count.h"
#include "kde.hpp"
typedef cv::Size_<double> Size;
typedef cv::Rect_<double> Rect;
typedef cv::Point_<double> Point;

DepthSegmenter::DepthSegmenter()
	{
		this->m_targetDepth = 0.0;
		this->m_targetSTD = 0.0;
		this->minSTD = 20;
		//KDE* kde = new KDE();
	}

cv::Mat1i DepthSegmenter::init(const cv::Mat & image, const Rect & boundingBox)
	{
		//Extract the target region of interest from the depth image
		Size windowSize = boundingBox.size();
		Point windowPosition = centerPoint(boundingBox);

		cv::Mat1w front_depth;
		if (getSubWindow(image, front_depth, windowSize, windowPosition))
			{
				/*				              cv::namedWindow("front_depth",0);
				 cv::imshow("front_depth",front_depth);*/
				cv::Scalar mean, stddev;

				/*在这里需要先计算深度图的均值和方差，根据均值和方差，去掉离群值，之后再对深度图进行目标分割*/

				/*				cv::Scalar pre_mean;
				 cv::Scalar pre_stddev;
				 cv::meanStdDev(front_depth, pre_mean, pre_stddev);
				 double pre_mean_pxl = pre_mean.val[0];
				 double pre_stddev_pxl = pre_stddev.val[0];
				 std::cout << "pre mean == " << pre_mean_pxl << "   pre stddev  == " << pre_stddev_pxl<<"  max == "<< pre_mean_pxl + 1.9 * pre_stddev_pxl<< std::endl;

				 cv::Mat1b mask = createMask_Remove_Noise<uint16_t>(front_depth, pre_mean_pxl + 1.9 * pre_stddev_pxl);*/
				//std::cout<<"front_depth == "<<std::endl<<front_depth<<std::endl;
				//std::cout<<"mask == "<<std::endl<<mask<<std::endl;
				//Find and store the empty depth values to be excluded from the histogram
				cv::Mat1b mask = createMaskTenmm<uchar>(front_depth);

				//front_depth = Mat_mul(front_depth, mask);
				//std::cout<<"front_depth == "<<std::endl<<front_depth<<std::endl;
				/*
				 cv::Mat1b mask_show;
				 //mask_show = mask.clone();
				 mask.copyTo(mask_show);
				 mask_show = mask_show.mul(255);
				 cv::namedWindow("mask_show", 0);
				 cv::imshow("mask_show", mask_show);*/
				//    cv::waitKey(0);
				//Create the histogram of depths in the region excluding the masked
				this->m_histogram = DepthHistogram::createHistogram(50, front_depth, mask);
				/*******************************************************/
				/*KDE* kde = new KDE();
				for (uchar i = 0; i < this->m_histogram.size(); i++)
					{
						std::vector<double> tmp;
						tmp.push_back(double(this->m_histogram[i]));
						kde->add_data(tmp);
					}
				std::vector<double> result_for_kde;
				if (kde->get_vars_count() == 1)
					{

						int length =800;
						int max=3000;
						double min_x = kde->get_min(0);
						double max_x = kde->get_max(0);
						double x_increment = (max_x - min_x) / length;
						cout << "# bandwidth var 1: " << kde->get_bandwidth(0) << endl;

						for (double x = min_x; x <= max_x; x += x_increment)
							{
								result_for_kde.push_back(kde->pdf(x));
					//		printf("%2.6F,%2.6F\n", x, kde->pdf(x));

							}
						cv::Mat picture(max, length, CV_8UC3, cv::Scalar(255, 255, 255));
						cv::namedWindow("kde", 0);
						for (double x = 1; x <length; x++)
							{

								int h1 = (int) (result_for_kde[x - 1] * 1000000)/4;
								int h2 = (int) (result_for_kde[x] * 1000000)/4;
								if(h1>max)h1=max;
								if(h2>max)h2=max;
					//			std::cout << h2 << "  ";
								cv::Point p1 = cv::Point(x - 1, max - h1);
								cv::Point p2 = cv::Point(x, max - h2);
								cv::line(picture, p1, p2, cv::Scalar(255, 0, 0),3);
							}
					//	std::cout << std::endl;
						auto position_max_value = std::max_element(result_for_kde.begin(),result_for_kde.end());
						std::cout << "max value is at "<<position_max_value -result_for_kde.begin() <<std::endl;
						cv::imshow("kde", picture);
					}
*/
				/**********************************************************/
				this->m_histogram.visualise("histogram visua");

				//Find the peaks in the histogram
				//在直方图中找出peaks，其中峰值不小于最大峰值的0.02，峰之间的距离不小于5
				std::vector<int> peaks = this->m_histogram.getPeaks(5, 0.02);
				//std::vector<int> peaks = this->m_histogram.get_fix_Peaks(pre_mean_pxl, pre_stddev_pxl);

			/*	std::cout << " get the peak value  ";
				for (auto i = peaks.begin(); i != peaks.end(); i++)
					{
						std::cout << *i << "  ";
					}
				std::cout << std::endl;*/
				//Group the points and label them
				this->labelsResults = this->m_histogram.getLabels(peaks);

				/*std::cout << " get the new centers vector  labelsResults.centers ";
				for (auto i = this->labelsResults.centers.begin(); i != this->labelsResults.centers.end(); i++)
					{
						std::cout << *i << "  ";
					}
				std::cout << std::endl;
				std::cout << " get the labelsResults.labels vector   ";
				for (auto i = this->labelsResults.labels.begin(); i != this->labelsResults.labels.end(); i++)
					{
						std::cout << *i << "  ";
					}
				std::cout << std::endl;

				std::cout << " get the labelsResults.labelsC vector   ";
				for (auto i = this->labelsResults.labelsC.begin(); i != this->labelsResults.labelsC.end(); i++)
					{
						std::cout << *i << "  ";
					}
				std::cout << std::endl;*/
				//矩阵L 表示每个像素属于分割出的第几块区域
				cv::Mat1i L = this->createLabelImageCC(front_depth, mask, this->labelsResults.centers, this->labelsResults.labels, this->labelsResults.labelsC);

				/*				std::cout << "after createLabelImageCC " << std::endl << std::endl;
				 std::cout << " get the new centers vector  labelsResults.centers ";
				 for (auto i = this->labelsResults.centers.begin(); i != this->labelsResults.centers.end(); i++)
				 {
				 std::cout << *i << "  ";
				 }
				 std::cout << std::endl;
				 std::cout << " get the labelsResults.labels vector   ";
				 for (auto i = this->labelsResults.labels.begin(); i != this->labelsResults.labels.end(); i++)
				 {
				 std::cout << *i << "  ";
				 }
				 std::cout << std::endl;

				 std::cout << " get the labelsResults.labelsC vector   ";
				 for (auto i = this->labelsResults.labelsC.begin(); i != this->labelsResults.labelsC.end(); i++)
				 {
				 std::cout << *i << "  ";
				 }
				 std::cout << std::endl;*/

				//Find the nearest object and calculate its mean depth and standard deviation
				int indexCenter = selectClosestObject(this->labelsResults.centers);
				//labelsResults.labelsC[indexCenter] 中存的是当前区域的区域号，挑选出属于当前区域的像素
				//std::cout << "select " << indexCenter << "th peak as target" << std::endl;
				//std::cout << "labelsResults.labelsC[indexCenter]  == " << labelsResults.labelsC[indexCenter] << std::endl;
				cv::Mat1b objectMask = createMask<uchar>(L, this->labelsResults.labelsC[indexCenter], false);
				objectMask.mul(mask);

				this->_ObjectMask =objectMask;

				//std::cout << "objectMask == " << std::endl  << objectMask<< std::endl;


				cv::Mat aaa;
				aaa = objectMask.clone();
				aaa =  aaa*255;
				//aaa.mul(255);
				// std::cout<<"aaa == "<<aaa<<std::endl;
				//aaa = aaa * 8;
			    cv::Rect ttt = this->rectRegions[indexCenter];
				//std::cout << "ttt x y width height == " << ttt.x << "  " << ttt.y << "  " << ttt.width << " " << ttt.height << std::endl;
			   cv::rectangle(aaa, ttt, cv::Scalar(255), 1);
				cv::namedWindow("aaa", 0);
				cv::imshow("aaa", aaa);
				this->_result =aaa;
				//	cv::waitKey(0);
				//std::cout<<"objectMask == "<<objectMask<<std::endl;
				cv::meanStdDev(front_depth, mean, stddev, objectMask);

				this->m_targetDepth = mean.val[0];
				if (stddev.val[0] < this->minSTD)
					stddev.val[0] = this->minSTD;

				this->m_targetSTD = stddev.val[0];
				cvCeil(modelNoise(this->m_targetDepth, this->m_targetSTD));

				std::cout << "this->m_targetDepth == " << this->m_targetDepth << "this->m_targetSTD == " << this->m_targetSTD << std::endl;

				return L;
			}

		return cv::Mat1i();
	}

static int frame = 0;

int DepthSegmenter::update(const cv::Mat & image, const Rect & boundingBox)
	{
		frame++;
		//Extract the target region of interest from the depth image
		Rect boundingBoxNEW = resizeBoundingBox(boundingBox, boundingBox.size() * 1.05); //Rect boundingBoxNEW=boundingBox;
		Size windowSize = boundingBoxNEW.size();
		Point windowPosition = centerPoint(boundingBoxNEW);

		cv::Mat1w front_depth;
		if (getSubWindow(image, front_depth, windowSize, windowPosition))
			{
				double minDepth, maxDepth;
				cv::Scalar mean, stddev;


				/*				cv::Scalar pre_mean;
				 cv::Scalar pre_stddev;
				 cv::meanStdDev(front_depth, pre_mean, pre_stddev);
				 double pre_mean_pxl = pre_mean.val[0];
				 double pre_stddev_pxl = pre_stddev.val[0];
				 std::cout << "pre mean == " << pre_mean_pxl << "   pre stddev  == " << pre_stddev_pxl<<"  max == "<< pre_mean_pxl + 1.9 * pre_stddev_pxl<< std::endl;


				 cv::Mat1b mask = createMask_Remove_Noise<uint16_t>(front_depth, pre_mean_pxl + 1.9 * pre_stddev_pxl);*/
				//std::cout<<"mask == "<<std::endl<<mask<<std::endl;
				//Find and store the empty depth values to be excluded from the histogram
				cv::Mat1b mask = createMaskTenmm<uchar>(front_depth);

				//	front_depth = Mat_mul(front_depth, mask);

				/*				cv::Mat1b mask_show;
				 mask.copyTo(mask_show);
				 mask_show = mask_show.mul(255);
				 cv::namedWindow("mask_show", 0);
				 cv::imshow("mask_show", mask_show);*/
				//    cv::waitKey(0);
				//Create the histogram of depths in the region excluding the mask
				this->m_histogram = DepthHistogram::createHistogram(cvFloor(modelNoise(this->m_targetDepth, this->m_targetSTD)), front_depth, mask);
				//this->m_histogram = DepthHistogram::createHistogram(50, front_depth, mask);
			/*********************************************************/
				/*KDE* kde = new KDE();
				for (uchar i = 0; i < this->m_histogram.size(); i++)
					{
						std::vector<double> tmp;
						tmp.push_back(double(this->m_histogram[i]));
						kde->add_data(tmp);
					}
				std::vector<double> result_for_kde;
				if (kde->get_vars_count() == 1)
					{

						int length =800;
						int max =3000;
						double min_x = kde->get_min(0);
						double max_x = kde->get_max(0);
						double x_increment = (max_x - min_x) / length;
						cout << "# bandwidth var 1: " << kde->get_bandwidth(0) << endl;

						for (double x = min_x; x <= max_x; x += x_increment)
							{
								result_for_kde.push_back(kde->pdf(x));
						//		printf("%2.6F,%2.6F\n", x, kde->pdf(x));

							}
						cv::Mat picture(max, length, CV_8UC3, cv::Scalar(255, 255, 255));
						cv::namedWindow("kde", 0);
						for (double x = 1; x < length; x++)
							{

								int h1 = (int) (result_for_kde[x - 1] * 1000000)/4;
								int h2 = (int) (result_for_kde[x] * 1000000)/4;
								//if(h1>max)h1=max;
								//if(h2>max)h2=max;
							//	std::cout << h2 << "  ";
								cv::Point p1 = cv::Point(x - 1, max - h1);
								cv::Point p2 = cv::Point(x, max - h2);
								cv::line(picture, p1, p2, cv::Scalar(255, 0, 0),3);
							}
						//std::cout << std::endl;
						auto position_max_value = std::max_element(result_for_kde.begin(),result_for_kde.end());
						std::cout << "max value is at "<<position_max_value -result_for_kde.begin() <<std::endl;
						cv::imshow("kde", picture);
					}*/

				/**********************************************************/
				this->m_histogram.visualise("histogram visua");

				cv::minMaxLoc(front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask);
				std::cout << "after mask  minDepth == " << minDepth << "     maxDepth == " << maxDepth << std::endl;
				//Find the peaks in the histogram
				std::vector<int> peaks = this->m_histogram.getPeaks();
				//std::vector<int> peaks = this->m_histogram.get_fix_Peaks(pre_mean_pxl, pre_stddev_pxl);

				/*std::cout << " get the peak value  ";
				for (auto i = peaks.begin(); i != peaks.end(); i++)
					{
						std::cout << *i << "  ";
					}
				std::cout << std::endl;*/

				bool emptyDepth = minDepth == 0 && maxDepth == 0;

				if (peaks.size() > 0 && (emptyDepth == false))
					{
						//Group the points and label them 其实是调用的 kmean
						this->labelsResults = this->m_histogram.getLabels(peaks);

					/*	std::cout << " get the new centers vector  labelsResults.centers ";
						for (auto i = this->labelsResults.centers.begin(); i != this->labelsResults.centers.end(); i++)
							{
								std::cout << *i << "  ";
							}
						std::cout << std::endl;
						std::cout << " get the labelsResults.labels vector   ";
						for (auto i = this->labelsResults.labels.begin(); i != this->labelsResults.labels.end(); i++)
							{
								std::cout << *i << "  ";
							}
						std::cout << std::endl;

						std::cout << " get the labelsResults.labelsC vector   ";
						for (auto i = this->labelsResults.labelsC.begin(); i != this->labelsResults.labelsC.end(); i++)
							{
								std::cout << *i << "  ";
							}
						std::cout << std::endl;*/
						this->m_labeledImage = this->createLabelImageCC(front_depth, mask, this->labelsResults.centers, this->labelsResults.labels, this->labelsResults.labelsC);
						if (maxDepth == minDepth)
							{
								maxDepth += 1;
							}

						/*						std::cout << " get the new centers vector  labelsResults.centers ";
						 for (auto i = this->labelsResults.centers.begin(); i != this->labelsResults.centers.end(); i++)
						 {
						 std::cout << *i << "  ";
						 }
						 std::cout << std::endl;
						 std::cout << " get the labelsResults.labels vector   ";
						 for (auto i = this->labelsResults.labels.begin(); i != this->labelsResults.labels.end(); i++)
						 {
						 std::cout << *i << "  ";
						 }
						 std::cout << std::endl;

						 std::cout << " get the labelsResults.labelsC vector   ";
						 for (auto i = this->labelsResults.labelsC.begin(); i != this->labelsResults.labelsC.end(); i++)
						 {
						 std::cout << *i << "  ";
						 }
						 std::cout << std::endl;*/

						int indexCloseCenter = selectClosestObject(this->labelsResults.centers);
						//Find the nearest object and calculate its mean depth and standard deviation
						/*

						 std::cout << "select " << indexCloseCenter << "th area as target" << std::endl;
						 std::cout << "labelsResults.labelsC[indexCenter]  == " << labelsResults.labelsC[indexCloseCenter] << std::endl;
						 */

						cv::Mat1b objectMask = createMask<uchar>(this->m_labeledImage, this->labelsResults.labelsC[indexCloseCenter], false);
						objectMask.mul(mask);
						this->_ObjectMask =objectMask;

						//std::cout << "objectMask == " << std::endl  << objectMask<< std::endl;

						cv::Mat aaa;
						aaa = objectMask.clone();
						aaa =  aaa*255;
						//aaa.mul(255);
						// std::cout<<"aaa == "<<aaa<<std::endl;
						//aaa = aaa * 8;
					    cv::Rect ttt = this->rectRegions[indexCloseCenter];
						//std::cout << "ttt x y width height == " << ttt.x << "  " << ttt.y << "  " << ttt.width << " " << ttt.height << std::endl;
					   cv::rectangle(aaa, ttt, cv::Scalar(255), 1);
						cv::namedWindow("aaa", 0);
						cv::imshow("aaa", aaa);

						this->_result =aaa;

						cv::meanStdDev(front_depth, mean, stddev, objectMask);
						//std::cout << "mean.val[0]  " << mean.val[0] << "  stddev.val[0] == " << stddev.val[0] << std::endl;
						int indexCenter = this->handleOcclusion(front_depth, this->labelsResults.centers, this->labelsResults.labelsC, this->m_targetDepth, this->m_targetSTD, mean.val[0],
								stddev.val[0]);

						//	std::cout<<"indexCenter == "<<indexCenter<<std::endl;

						std::cout << "this->m_targetDepth == " << mean.val[0] << "this->m_targetSTD == " << stddev.val[0] << std::endl;
						//float centerDepth=(indexCenter>-1) ? this->labelsResults.centers[indexCenter] : this->labelsResults.centers.size()-1;
						float centerDepth = (indexCenter > -1) ? this->labelsResults.centers[indexCenter] : this->labelsResults.labels.size() - 1;
						//float centerDepthBUG = (indexCenter>-1) ? this->labelsResults.centers[indexCenter] : this->labelsResults.centers.size() - 1;
						//	std::cout<<"centerDepth == "<<centerDepth<<std::endl;
						int binCenter = cvRound(centerDepth); //the center are already in bin coordinates
						//int binCenterBUG = cvRound(centerDepthBUG);//the center are already in bin coordinates
						//if (binCenterBUG >= this->labelsResults.labels.size())
						//printf("binCenter: %d labelsSize: %d\n", binCenter, this->labelsResults.labels.size());

						int bin = std::find(this->labelsResults.labels.begin(), this->labelsResults.labels.end(), this->labelsResults.labels[binCenter]) - this->labelsResults.labels.begin();
						//int bin = std::find(this->labelsResults.labels.begin(), this->labelsResults.labels.end(), this->labelsResults.labelsC[binCenter]) - this->labelsResults.labels.begin();
						int tmpBin = (this->m_histogram.depthToBin(this->getTargetDepth() - 1.5 * this->getTargetSTD()));
						//		std::cout << "bin  == " << bin << "   tmpBin == " << tmpBin << std::endl;
						bin = std::min<int>(bin, tmpBin);

						return bin;
					}
			}

		return 0;
	}

double DepthSegmenter::getTargetDepth() const
	{
		return this->m_targetDepth;
	}

double DepthSegmenter::getTargetSTD() const
	{
		return this->m_targetSTD;
	}

const DepthHistogram & DepthSegmenter::getHistogram() const
	{
		return this->m_histogram;
	}

const cv::Mat1b DepthSegmenter::createLabelImage(const cv::Mat1w & region, const cv::Mat1b mask, const std::vector<float> & C, const std::vector<int> & labels) const
	{
		//给图片每个像素的深度值分配一个 峰的序号，，深度为0 的分配峰的最大序号加1
		double min, max;
		cv::Mat1b L = cv::Mat1b::zeros(region.rows, region.cols);

		cv::minMaxLoc(region, &min, &max, nullptr, nullptr, mask);

		for (int x = 0; x < region.cols; x++)
			{
				for (int y = 0; y < region.rows; y++)
					{
						double depth = static_cast<double>(region(y, x));

						if (depth != 0.0)
							{

								int index = this->m_histogram.depthToBin(depth);
								L(y, x) = static_cast<uchar>(labels[index]);						//第几个峰
							}
						else
							{
								L(y, x) = static_cast<uchar>(C.size());						//峰个数
							}
					}
			}

		return L;
	}

const cv::Mat1b DepthSegmenter::createLabelImage(const cv::Mat1w & region, const cv::Mat1b mask, const std::vector<float> & C, const std::vector<int> & labels, const DepthHistogram &histogram) const
	{
		double min, max;
		cv::Mat1b L = cv::Mat1b::zeros(region.rows, region.cols);

		cv::minMaxLoc(region, &min, &max, nullptr, nullptr, mask);

		for (int x = 0; x < region.cols; x++)
			{
				for (int y = 0; y < region.rows; y++)
					{
						double depth = static_cast<double>(region(y, x));

						if (depth != 0.0)
							{

								int index = histogram.depthToBin(depth);
								L(y, x) = static_cast<uchar>(labels[index]);
							}
						else
							{
								L(y, x) = static_cast<uchar>(C.size());
							}
					}
			}

		return L;
	}

const cv::Mat1i DepthSegmenter::createLabelImageCC(const cv::Mat1w & region, const cv::Mat1b mask, std::vector<float> & C, const std::vector<int> & labels, std::vector<int> & labelsC,
		float smallAreaFraction)
	{
		int minimumArea = cvRound(smallAreaFraction * region.rows * region.cols);
		//给图片每个像素的深度值分配一个 峰的序号，，深度为0 的分配峰的最大序号加1
		cv::Mat1b LnoCC = this->createLabelImage(region, mask, C, labels);
		cv::Mat L = cv::Mat::zeros(region.rows, region.cols, CV_32SC1);
		cv::Mat1b tmpResults = cv::Mat1b::zeros(region.rows, region.cols);
		cv::Mat statsCC, centroidsCC, Ltemp;

		//std::vector< cv::Rect_< int > > boxesTmp,boxesL;
		labelsC.clear();			//std::vector<int> labelsCnew;
		this->areaRegions.clear();
		this->rectRegions.clear();
		std::vector<float> centerNew;

		//set the iterator to the first element
		std::vector<float>::iterator itC = C.begin();
		int numElements = (int) C.size();
		int offset = 0;

		//std::cout<<"the number of peak is "<< numElements<<std::endl;

		//iterate for every label and eventually split it
		//遍历每一个峰
		for (int i = 0; i < numElements; i++)
			{

				float center = C[i];
				//比较 LabelImage 跟当前峰值的序号是否相等，如果相等，对于位置为255 ，否则 为0
				cv::compare(LnoCC, i, tmpResults, CV_CMP_EQ);		//

				/*				cv::namedWindow("tmpResults", 0);
				 cv::imshow("tmpResults", tmpResults);*/
				//cv::waitKey(0);
				//check if necessary to split the matrix....
				//联通域函数
				//statsCC的每一行的五个元素分别为 每一块区域的x,y,width,height area,行数为分割的区域数+1（背景）
				//centroidsCC对于中心点
				//Ltemp 则表示当前像素是第几个轮廓 CV_32S
				cv::connectedComponentsWithStats(tmpResults, Ltemp, statsCC, centroidsCC);

				//std::cout<<"Ltemp == "<<std::endl<<Ltemp<<std::endl;
				//std::cout<<"elementsToAdd == "<<centroidsCC.rows - 1<<std::endl;
				//cv::waitKey(0);

				int elementsToAdd = centroidsCC.rows - 1; //not consider the black pixels...不考虑背景，所以 -1
				//cv::add(Ltemp,100,Ltemp);
				cv::add(Ltemp, offset, Ltemp, tmpResults);
				//	std::cout<<"Ltemp == "<<std::endl<<Ltemp<<std::endl;
				//	cv::waitKey(0);
				//cv::imwrite("C:/myDocs/variousSequences/dummyFolder/face_occ5/LtempNEW.png",Ltemp);

				//按位或，相当于相加，L表示当前元素在所有峰的下所在第几个轮廓,因为前面compare函数，所以每个像素值肯定只存在于一个轮廓，所有相或等于相加
				cv::bitwise_or(L, Ltemp, L);

				//	std::cout<<"L == "<<std::endl<<L<<std::endl;
				//	cv::waitKey(0);

				//if necessary add the new labels and indexes
				for (int j = 1; j < elementsToAdd + 1; j++)
					{
						//把分割出来的区域的面积存起来
						int tmpArea = statsCC.at<int>(j, cv::CC_STAT_AREA);
						this->areaRegions.push_back(tmpArea);

						cv::Rect rect(statsCC.at<int>(j, 0), statsCC.at<int>(j, 1), statsCC.at<int>(j, 2), statsCC.at<int>(j, 3));
						this->rectRegions.push_back(rect);
						//eventually insert a dummy number for the small regions
						//
						(tmpArea > minimumArea ? centerNew.push_back(center) : centerNew.push_back(1000000));

						labelsC.push_back(offset + j);
					}
				offset = (int) labelsC.size();
				//std::cout<<"offset == "<<offset<<std::endl;
			}
		/*std::cout<<"labelsC  == ";
		 for(auto i = labelsC.begin();i!=labelsC.end();i++)
		 {
		 std::cout<<" "<< *i<<" ";
		 }
		 std::cout<<std::endl;*/

		C = centerNew;

		Ltemp.release();
		tmpResults.release();
		LnoCC.release();

		return L;
	}

const std::vector<cv::Point_<double> > DepthSegmenter::createLabelImageCCOccluder(const cv::Mat1w & region, const cv::Mat1b mask, std::vector<float> & C, const std::vector<int> & labels,
		std::vector<int> & labelsC, const DepthHistogram &histogram, int minimumArea, cv::Mat1b &objectMask) const

	{

		std::vector<Point> result;
		std::vector<Point> tmpVector;
		std::vector<int> areaVector;

		cv::Mat1b LnoCC = this->createLabelImage(region, mask, C, labels, histogram);
		cv::Mat L = cv::Mat::zeros(region.rows, region.cols, CV_32SC1);
		cv::Mat1b tmpResults = cv::Mat1b::zeros(region.rows, region.cols);
		cv::Mat statsCC, centroidsCC, Ltemp;

		labelsC.clear();
		std::vector<float> centerNew;
		//set the iterator to the first element
		std::vector<float>::iterator itC = C.begin();
		int numElements = (int) C.size();
		int offset = 0;

		//iterate for every label and eventually split it
		for (int i = 0; i < numElements; i++)
			{

				float center = C[i];
				cv::compare(LnoCC, i, tmpResults, CV_CMP_EQ);

				//check if necessary to split the matrix....

				cv::connectedComponentsWithStats(tmpResults, Ltemp, statsCC, centroidsCC);

				int elementsToAdd = centroidsCC.rows - 1; //not consider the black pixels...

				cv::add(Ltemp, offset, Ltemp, tmpResults);
				cv::bitwise_or(L, Ltemp, L);

				//if necessary add the new labels and indexes
				for (int j = 1; j < elementsToAdd + 1; j++)
					{
						int tmpArea = statsCC.at<int>(j, cv::CC_STAT_AREA);
						areaVector.push_back(tmpArea);
						//eventually insert a dummy number for the small regions
						if (tmpArea > minimumArea)
							{
								tmpVector.push_back(cv::Point(centroidsCC.at<double>(j, 0), centroidsCC.at<double>(j, 1)));

								centerNew.push_back(center);
							}
						else
							{
								centerNew.push_back(1000000);
								tmpVector.push_back(cv::Point(-1, -1));
							}

						labelsC.push_back(offset + j);
					}

				offset = (int) labelsC.size();

			}

		//now exclude from the list also the closest object....as it is the occluder
		//假设最近的目标是遮挡，去掉遮挡
		int indexCenter = this->selectClosestObject(centerNew, areaVector);
		tmpVector[indexCenter].x = -1;
		tmpVector[indexCenter].y = -1;

		int indexLabel = labelsC[indexCenter];
		cv::Mat tmpMask = createMask<uchar>(L, indexLabel, false);
		tmpMask.copyTo(objectMask);
		tmpMask.release();

		for (int j = 0; j < tmpVector.size(); j++)
			{
				if (tmpVector[j].x != -1)
					{
						result.push_back(tmpVector[j]);
					}
			}

		C = centerNew;

		Ltemp.release();
		tmpResults.release();
		LnoCC.release();
		L.release();
		return result;
	}

const std::vector<cv::Point_<double> > DepthSegmenter::createLabelImageCCOccluder(const cv::Mat1w & region, const cv::Mat1b mask, std::vector<float> & C, const std::vector<int> & labels,
		std::vector<int> & labelsC, const DepthHistogram &histogram, int minimumArea, cv::Mat1b &objectMask, std::vector<float> & centroidsCandidates, cv::Rect_<double> &occluderRect) const

	{

		std::vector<Point> result;
		std::vector<Point> tmpVector;
		std::vector<int> areaVector;
		std::vector<cv::Rect_<double> > rectVector;

		cv::Mat1b LnoCC = this->createLabelImage(region, mask, C, labels, histogram);
		cv::Mat L = cv::Mat::zeros(region.rows, region.cols, CV_32SC1);
		cv::Mat1b tmpResults = cv::Mat1b::zeros(region.rows, region.cols);
		cv::Mat statsCC, centroidsCC, Ltemp;

		labelsC.clear(); //std::vector<int> labelsCnew;
		std::vector<float> centerNew;
		//set the iterator to the first element
		std::vector<float>::iterator itC = C.begin();
		int numElements = (int) C.size();
		int offset = 0;

		//iterate for every label and eventually split it
		for (int i = 0; i < numElements; i++)
			{

				float center = C[i];
				cv::compare(LnoCC, i, tmpResults, CV_CMP_EQ);

				//check if necessary to split the matrix....

				cv::connectedComponentsWithStats(tmpResults, Ltemp, statsCC, centroidsCC);

				int elementsToAdd = centroidsCC.rows - 1; //not consider the black pixels...

				cv::add(Ltemp, offset, Ltemp, tmpResults);
				cv::bitwise_or(L, Ltemp, L);

				//if necessary add the new labels and indexes
				for (int j = 1; j < elementsToAdd + 1; j++)
					{
						int tmpArea = statsCC.at<int>(j, cv::CC_STAT_AREA);
						areaVector.push_back(tmpArea);
						rectVector.push_back(
								Rect(statsCC.at<int>(j, cv::CC_STAT_LEFT), statsCC.at<int>(j, cv::CC_STAT_TOP), statsCC.at<int>(j, cv::CC_STAT_WIDTH), statsCC.at<int>(j, cv::CC_STAT_HEIGHT)));
						//eventually insert a dummy number for the small regions
						if (tmpArea > minimumArea)
							{
								tmpVector.push_back(cv::Point(centroidsCC.at<double>(j, 0), centroidsCC.at<double>(j, 1)));

								centerNew.push_back(center);
								centroidsCandidates.push_back(histogram.binToDepth(center));
							}
						else
							{
								centerNew.push_back(1000000);
								tmpVector.push_back(cv::Point(-1, -1));
							}

						labelsC.push_back(offset + j);
					}

				offset = (int) labelsC.size();

			}

		//now exclude from the list also the closest object....as it is the occluder
		int indexCenter = this->selectClosestObject(centerNew, areaVector);
		tmpVector[indexCenter].x = -1;
		tmpVector[indexCenter].y = -1;
		occluderRect = rectVector[indexCenter];

		int indexLabel = labelsC[indexCenter];
		cv::Mat tmpMask = createMask<uchar>(L, indexLabel, false);
		tmpMask.copyTo(objectMask);
		tmpMask.release();

		for (int j = 0; j < tmpVector.size(); j++)
			{
				if (tmpVector[j].x != -1)
					{
						result.push_back(tmpVector[j]);
					}
			}

		C = centerNew;

		Ltemp.release();
		tmpResults.release();
		LnoCC.release();
		L.release();
		return result;
	}

const int DepthSegmenter::selectClosestObject(std::vector<float> & centroids)
	{

		std::vector<int>::iterator result;

		int numElements = (int) centroids.size();
		int returnedIndex = numElements - 1;

		for (int i = 0; i < numElements - 1; i++)
			{

				//前面判断若分割出的区域小于ROI的0.09，则压入1000000，表示无效区域
				//if it is not a smaller region, this is the closer object
				if (centroids[i] < 1000000)
					{
						//select the interval
						//挑出下一个面积大于总面积0.09的块，
						int tmpIndex = i;
						for (int j = i + 1; j < numElements; j++)
							{
								if (centroids[j] > centroids[i] && centroids[j] < 1000000)
									{
										tmpIndex = j;
										std::cout << "i == " << i << "   j == " << j << std::endl;
										break;
									}

								else
									{
										tmpIndex++;
									}

							}

						result = std::max_element(this->areaRegions.begin() + i, this->areaRegions.begin() + tmpIndex);
						returnedIndex = std::distance(this->areaRegions.begin(), result);

						break;
					}
			}

		return returnedIndex;
	}

const int DepthSegmenter::selectClosestObject(std::vector<float> & centroids, std::vector<int> & areaVector) const
	{

		std::vector<int>::const_iterator result;

		int numElements = (int) centroids.size();
		int returnedIndex = numElements - 1;

		for (int i = 0; i < numElements - 1; i++)
			{

				//if it is not a smaller region, this is the closer object
				if (centroids[i] < 1000000)
					{
						//select the interval
						int tmpIndex = i;
						for (int j = i + 1; j < numElements; j++)
							{
								if (centroids[j] > centroids[i] && centroids[j] < 1000000)
									{
										tmpIndex = j;
										break;
									}

								else
									{
										tmpIndex++;
									}

							}

						result = std::max_element(areaVector.begin() + i, areaVector.begin() + tmpIndex);
						returnedIndex = std::distance<std::vector<int>::const_iterator>(areaVector.begin(), result);

						break;
					}
			}

		return returnedIndex;
	}

const cv::Mat1i & DepthSegmenter::getLabeledImage() const
	{
		return this->m_labeledImage;
	}

const std::vector<int> & DepthSegmenter::getAreaRegions() const
	{
		return this->areaRegions;
	}

const DepthHistogram::Labels &DepthSegmenter::getLabelsResults() const
	{
		return this->labelsResults;
	}

const int DepthSegmenter::handleOcclusion(const std::vector<float> & centroids, const double previousDepth, const double previousSTD, const double targetDepth, const double targetSTD)
	{
		int minIndex = 0;
		double minDistance;
		std::vector<float> peakDistances = centroids;

		double max = this->m_histogram.size();
		//把vector 中 peakDistances 转换想对于上一次目标深度的距离
		for (auto itr = peakDistances.begin(); itr != peakDistances.end(); itr++)
			{
				*itr = std::abs(this->m_histogram.binToDepth(*itr) - previousDepth);
			}

		minDistance = peakDistances[0];
		//找出距离最小值
		for (size_t i = 1; i < peakDistances.size(); i++)
			{
				if (peakDistances[i] < peakDistances[minIndex])
					{
						minIndex = static_cast<int>(i);
						minDistance = peakDistances[i];
					}
			}

		//%register the plane index when you filtered out some small
		if ((minIndex == 0) && (minDistance < 3.0 * previousSTD))
			{
				this->m_occluded = false;
				//everything seems ok....no occluding object, just a movement
				//of the object....update the depth!!!
				this->m_targetDepth = targetDepth;
				this->m_targetSTD = targetSTD;
			}
		else
			{
				//// THERE IS AN OCCLUSION......WHAT TO DO?
				//find the new corresponding region (if exist) and calculate
				if (minDistance < 2.5 * previousSTD)
					{
						this->m_occluded = true;
						this->m_targetDepth = this->m_histogram.binToDepth(centroids[minIndex]);
						this->m_targetSTD = targetSTD;

						if (this->m_targetSTD < this->minSTD)
							{
								this->m_targetSTD = previousSTD;
							}
					}
				else
					{
						this->m_occluded = false;
						this->m_targetDepth = targetDepth;
						this->m_targetSTD = targetSTD;
					}
			}

		return minIndex;
	}

const int DepthSegmenter::handleOcclusion(const cv::Mat& front_depth, const std::vector<float> & centroids, const std::vector<int> & labelsC, const double previousDepth, const double previousSTD,
		const double targetDepth, const double targetSTD)
	{
		int minIndex = 0;
		double minDistance;
		std::vector<float> peakDistances = centroids;

		int minNonZero = 0;
		bool foundMin = false;

		double max = this->m_histogram.size();

		for (auto itr = peakDistances.begin(); itr != peakDistances.end(); itr++)
			{
				if (*itr >= (1000000 - 1) && foundMin == false)
					minNonZero++;
				else
					foundMin = true;

				*itr = std::abs(this->m_histogram.binToDepth(*itr) - previousDepth);
			}

		minDistance = peakDistances[0];

		for (size_t i = 1; i < peakDistances.size(); i++)
			{
				if (peakDistances[i] < peakDistances[minIndex])
					{
						minIndex = static_cast<int>(i);
						minDistance = peakDistances[i];
					}
			}
		std::cout << "target remove minDistance == " << minDistance << std::endl;

		//	std::cout << "the closest thing  is  the target ==  " << (minIndex == minNonZero) << std::endl;
		//%register the plane index when you filtered out some small
		//%regions....
		if ((minIndex == minNonZero) && (minDistance < 3.0 * previousSTD))
			{
				this->m_occluded = false;
				//everything seems ok....no occluding object, just a movement
				//of the object....update the depth!!!
				this->m_targetDepth = targetDepth;
				this->m_targetSTD = targetSTD;
			}
		else
			{
				//// THERE IS AN OCCLUSION......WHAT TO DO?
				//find the new corresponding region (if exist) and calculate
				if (minDistance < 2.5 * previousSTD)
					{
						this->m_occluded = true;
						//this->m_targetDepth = ( ( ( centroids[ minIndex ] / max ) * ( this->m_histogram.maximum() - this->m_histogram.minimum() ) ) + this->m_histogram.minimum() );
						this->m_targetDepth = this->m_histogram.binToDepth(centroids[minIndex]);
						//WRONG!!! NEED TO BE RECALCULATED....this->m_targetSTD = targetSTD;
						int indexLabel = labelsC[minIndex];
						cv::Mat1b objectMask = createMask<uchar>(this->m_labeledImage, indexLabel, false);
						cv::Scalar mean, stddev;
						cv::meanStdDev(front_depth, mean, stddev, objectMask);
						std::cout << "*********segmenter funtion think into occlusion****************" << std::endl;
						objectMask.release();
						this->m_targetSTD = stddev.val[0];
						if (this->m_targetSTD < this->minSTD)
							{
								this->m_targetSTD = previousSTD;
							}
					}
				else
					{
						this->m_occluded = false;
						this->m_targetDepth = previousDepth;
						this->m_targetSTD = previousSTD;
						minIndex = -1;
					}
			}

		return minIndex;
	}

const bool DepthSegmenter::isOccluded() const
	{
		return this->m_occluded;
	}

//debug function to save histogram
void DepthSegmenter::debugSaveHistogram(std::string filename)
	{
		FILE *pfile = fopen(filename.c_str(), "w");

		for (int i = 0; i < this->m_histogram.size(); i++)
			fprintf(pfile, "%f %f\n", this->m_histogram.binToDepth(i), this->m_histogram[i]);

		fclose(pfile);
	}

const cv::Mat1b DepthSegmenter::segment(const cv::Mat1w & frame, const cv::Rect_<double> & boundingBox) const
	{
		//Extract the target region of interest from the depth image
		Size windowSize = boundingBox.size();
		Point windowPosition = centerPoint(boundingBox);

		cv::Mat1w front_depth;
		if (getSubWindow(frame, front_depth, windowSize, windowPosition))
			{
				double minDepth, maxDepth;

				//Find and store the empty depth values to be excluded from the histogram
				cv::Mat1b mask = createMask(front_depth);

				//Create the histogram of depths in the region excluding the mask
				DepthHistogram histogram = DepthHistogram::createHistogram(cvCeil(modelNoise(this->m_targetDepth, this->m_targetSTD)), front_depth, mask);

				cv::minMaxLoc(front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask);

				//Find the peaks in the histogram
				int minimumPeakDistance = (histogram.size() < 50) ? 1 : 3;
				std::vector<int> peaks = histogram.getPeaks(minimumPeakDistance, 0.005);

				if (peaks.size() > 0)
					{
						//Group the points and label them
						DepthHistogram::Labels labels = histogram.getLabels(peaks);
						return this->createLabelImage(front_depth, mask, labels.centers, labels.labels, histogram);
					}
			}

		return cv::Mat1b();
	}

const std::vector<cv::Point_<double> > DepthSegmenter::segmentOccluder(const cv::Mat1w & frame, const cv::Rect_<double> & boundingBox, const int minimumArea, cv::Mat1b &objectMask) const
	{
		std::vector<Point> result;
		//Extract the target region of interest from the depth image
		Size windowSize = boundingBox.size();
		Point windowPosition = centerPoint(boundingBox);

		cv::Mat1w front_depth;
		if (getSubWindow(frame, front_depth, windowSize, windowPosition))
			{
				double minDepth, maxDepth;

				//Find and store the empty depth values to be excluded from the histogram
				cv::Mat1b mask = createMask(front_depth);

				//Create the histogram of depths in the region excluding the mask
				DepthHistogram histogram = DepthHistogram::createHistogram(cvCeil(modelNoise(this->m_targetDepth, this->m_targetSTD)), front_depth, mask);

				cv::minMaxLoc(front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask);

				//Find the peaks in the histogram
				int minimumPeakDistance = (histogram.size() < 50) ? 1 : 3;
				std::vector<int> peaks = histogram.getPeaks(minimumPeakDistance, 0.005);

				if (peaks.size() > 0)
					{
						//Group the points and label them
						DepthHistogram::Labels labels = histogram.getLabels(peaks);
						result = this->createLabelImageCCOccluder(front_depth, mask, labels.centers, labels.labels, labels.labelsC, histogram, minimumArea, objectMask);

						return result;
					}
			}

		return result;
	}

const std::vector<cv::Point_<double> > DepthSegmenter::segmentOccluder(const cv::Mat1w & frame, const cv::Rect_<double> & boundingBox, const int minimumArea, cv::Mat1b &objectMask,
		std::vector<float> &centersCandidate, cv::Rect_<double> &occluderRect) const
	{
		std::vector<Point> result;
		//Extract the target region of interest from the depth image
		Size windowSize = boundingBox.size();
		Point windowPosition = centerPoint(boundingBox);

		cv::Mat1w front_depth;
		if (getSubWindow(frame, front_depth, windowSize, windowPosition))
			{
				double minDepth, maxDepth;

				//Find and store the empty depth values to be excluded from the histogram
				cv::Mat1b mask = createMask(front_depth);

				//Create the histogram of depths in the region excluding the mask
				DepthHistogram histogram = DepthHistogram::createHistogram(cvCeil(modelNoise(this->m_targetDepth, this->m_targetSTD)), front_depth, mask);

				cv::minMaxLoc(front_depth, &minDepth, &maxDepth, nullptr, nullptr, mask);

				//Find the peaks in the histogram
				int minimumPeakDistance = (histogram.size() < 50) ? 1 : 3;
				std::vector<int> peaks = histogram.getPeaks(minimumPeakDistance, 0.005);
				bool emptyHIST = (minDepth == 0 && maxDepth == 0);
				if (peaks.size() > 0 && emptyHIST == false)
					{
						//Group the points and label them
						DepthHistogram::Labels labels = histogram.getLabels(peaks);
						result = this->createLabelImageCCOccluder(front_depth, mask, labels.centers, labels.labels, labels.labelsC, histogram, minimumArea, objectMask, centersCandidate, occluderRect);

						return result;
					}
			}

		return result;
	}
