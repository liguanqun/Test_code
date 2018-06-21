#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <tbb/parallel_for.h>
#include "DepthHistogram.h"
#include "math_helper.hpp"

DepthHistogram::DepthHistogram()
	{
		this->m_minimum = 0;
		this->m_maximum = 0;
	}

const std::vector<int> DepthHistogram::getPeaks() const
	{
		int start = 0;
		int end = this->size() - 1;
		double max = 0.0;

		if (this->size() > 50)
			{
				cv::minMaxLoc(this->m_bins, nullptr, &max);

				for (int i = 0; i < this->size(); i++)
					{
						if (this->m_bins(i) > max * 0.05) //0.005
							{
								/*	原代码出错
								 start = std::min(start, i);
								 end = std::max(end, i);*/
								start = std::max(start, i);
								end = std::min(end, i);
							}
					}
			}
		std::cout << "end - start == " << abs(end - start) << std::endl;
		return this->getPeaks((abs(end - start) > 50 ? 3 : 1));
	}
const std::vector<int> DepthHistogram::get_fix_Peaks(const double pre_mean, const double pre_stddev) const
	{
		//首先判断是否有遮挡，有遮挡的话选出三个峰，没有遮挡的话选出两个峰，作为一下聚类的种子点，（三个种子代表：遮挡物，目标，背景）
		double histDepth = this->depthToBin(pre_mean+pre_stddev);
		std::vector<float> pre_depth;
		for (int i = 0; i <= histDepth; i++)
			{
				pre_depth.push_back(this->m_bins(i));
			}
		//从最近的距离到中值之间连续0的长度，判断是否有遮挡物与目标之间的断层
		std::vector<int> max_tmp;
		bool flag = false;
		int tmp = 0;
		for (auto it = pre_depth.begin(); it != pre_depth.end(); it++)
			{
				//以下判断太繁琐，但是容易理解，先不优化，以后再优化
				if (*it == 0 && !flag)
					{
						flag = true;
						tmp++;
					}
				else if (*it == 0 && flag)
					{
						tmp++;
					}
				else if (*it != 0 && flag)
					{
						flag = false;
						max_tmp.push_back(tmp++);
						tmp = 0;
					}
				else if (*it != 0 && !flag)
					{
						flag = false;
						tmp = 0;
					}
				else
					{
						flag = false;
						tmp = 0;
					}

			}

		std::vector<int> candidates;
		if (max_tmp.empty())
			{
				candidates.push_back(0);
				candidates.push_back(this->m_bins.rows);
			}
		else if (max_tmp[std::max_element(max_tmp.begin(), max_tmp.end()) - max_tmp.begin()] > 3)
			{
				std::cout<<"**********************maybe occ is being"<<std::endl;
				candidates.push_back(0);
				candidates.push_back(histDepth);
				candidates.push_back(this->m_bins.rows);
			}
		else
			{
				candidates.push_back(0);
				candidates.push_back(this->m_bins.rows);
			}

		return candidates;
	}
const std::vector<int> DepthHistogram::getPeaks(const int minimumPeakdistance, const double minimumPeakHeight) const
	{
		if (!this->m_bins.empty())
			{
				double max = 0.0;
				double current_max = 0.0;
				cv::Point current_max_location;
				cv::Mat1f local_histogram;
				this->m_bins.copyTo(local_histogram);
				std::vector<int> candidates;

				cv::minMaxLoc(this->m_bins, nullptr, &max);
				cv::minMaxLoc(local_histogram, nullptr, &current_max, nullptr, &current_max_location);

				//std::cout << "current_max = " << current_max << std::endl;

				//Check if the first bin is a peak (special case, no left neighbour)
				//	std::cout<<"this->m_bins(0) == "<<this->m_bins(0)<<"  this->m_bins(1)=="<<this->m_bins(1)<<std::endl;
				//	std::cout<<"minimumPeakHeight * max == "<<minimumPeakHeight * max<<std::endl;
				if ((this->m_bins(0) > this->m_bins(1)) && (this->m_bins(0) > minimumPeakHeight * max))
					{
						candidates.push_back(0);
						//	std::cout<< "candidates.push_back  0"<<std::endl;

					}

				//Check if the last bin is a peak (special case, no right neighbour)
				if ((this->m_bins(this->m_bins.rows - 1) > this->m_bins(this->m_bins.rows - 2)) && (this->m_bins(this->m_bins.rows - 1) > minimumPeakHeight * max))
					{
						candidates.push_back(this->m_bins.rows - 1);
						//	std::cout<< "candidates.push_back  "<<this->m_bins.rows - 1<<std::endl;
					}

				//Check the rest of the bins to see if they are a peak
				for (int i = 1; i < this->m_bins.rows - 1; i++)
					{
						float left = this->m_bins(i - 1);
						float right = this->m_bins(i + 1);
						float value = this->m_bins(i);

						if ((value > left) && (value > right) && (value > minimumPeakHeight * max))
							{
								candidates.push_back(i);
								//	std::cout<< "candidates.push_back  "<< i<<std::endl;
							}
					}

				//Filter out the neighbouring peaks that are closer than the minimum peak distance
				//把距离 小于设置最小的峰值之间距离 过滤掉
				//std::cout<<"candidates.size == "<<candidates.size()<<std::endl;
				if (minimumPeakdistance > 1)
					{
						while (current_max > minimumPeakHeight * max)
							{
								auto itr = std::find(candidates.begin(), candidates.end(), current_max_location.y);

								if (itr != candidates.end())
									{
										int index = *itr;
										//      std::cout<<"index == "<<index<<std::endl;
										for (auto itr2 = candidates.begin(); itr2 != candidates.end();)
											{
												//std::cout<<"erase index "<< (*itr2)<<std::endl;

												if ((*itr2 >= index - minimumPeakdistance) && (*itr2 <= index + minimumPeakdistance) && (*itr2 != index))
													{
														//		std::cout<<"erase the peak "<< (*itr2)<<std::endl;
														itr2 = candidates.erase(itr2);

													}
												else
													{
														itr2++;
														//std::cout<<"erase index "<< *itr2<<std::endl;
													}
											}
									}

								local_histogram(current_max_location.y) = 0;

								cv::minMaxLoc(local_histogram, nullptr, &current_max, nullptr, &current_max_location);
							}
					}

				//Sort the candidates so that the nearest is at candiates[ 0 ]
				std::sort(candidates.begin(), candidates.end(), std::less<int>());

				return candidates;
			}

		return std::vector<int>();
	}

const DepthHistogram::Labels DepthHistogram::getLabels(const std::vector<int> & peaks) const
	{
		std::vector<float> centroids(peaks.size());
		for (uint i = 0; i < peaks.size(); i++)
			{
				centroids[i] = static_cast<float>(peaks[i]);
				//std::cout << "centroids[i] == " << centroids[i] << std::endl;
			}

		return this->kmeans(centroids);
	}

const DepthHistogram::Labels DepthHistogram::kmeans(const std::vector<float> & centroids) const
	{
		float dC = 1000.0f;
		DepthHistogram::Labels result;
		result.centers = centroids;
		result.labelsC.assign(centroids.size(), 0); //先赋初始值为0
		result.labels.resize(this->m_bins.rows); //vector里存的是与当前位置的bin距离最近的是第几个峰

		while (dC > 1.0f)
			{
				std::vector<float> oldCentroids = result.centers;

				//Assign each label to the nearest centroid
				//TODO: Investigate possible parallelisation
				//for( int i = 0; i < this->m_bins.rows; i++ )
				//为 每一个m_bins的值设置距离最近的峰
				//result.labels[ i ] 里存的是 当前的bin 最近的是第 j 个峰
				tbb::parallel_for<uint>(0, this->m_bins.rows, 1, [&result]( const uint i )
					{
						for( uint j = 0; j < result.centers.size(); j++ )
							{
								if( std::abs( result.centers[ j ] - i ) < std::abs( result.centers[ result.labels[ i ] ] - i ) )
									{
										result.labels[ i ] = static_cast< int >( j );
									}
							}
					});

				/*				for (int i = 0; i < this->m_bins.rows; i++)
				 {
				 for (uint j = 0; j < result.centers.size(); j++)
				 {
				 if (std::abs(result.centers[j] - i) < std::abs(result.centers[result.labels[i]] - i))
				 {
				 result.labels[i] = static_cast<int>(j);
				 }
				 }
				 }*/

				//TODO: Investigate possible parallelisation
				//for( uint i = 0; i < result.centers.size(); i++ )
				tbb::parallel_for<uint>(0, result.centers.size(), 1, [this,&result,&oldCentroids]( const uint i )
					{
						float numerator = 0.0;				//分子
						float denominator = 0.0;//分母

						for( int j = 0; j < this->m_bins.rows; j++ )
							{
								//当前m_bins(j)属于centers(i)
								if( static_cast< uint >( result.labels[ j ] == i ) )
									{
										//
										numerator += j * this->m_bins( j );
										denominator += this->m_bins( j );
									}
							}
						//当前聚类centers(i) 下 所有点的新的中心点centers(i)
						result.centers[ i ] = numerator / denominator;
						//判断当前数据是否有效，确保有效，否则不更新，
						if( !std::isfinite( result.centers[ i ] ) )

							{
								result.centers[ i ] = oldCentroids[ i ];
							}
					});

				//Find the maximum that we moved the centroids
				//找出中心点移动的最大值，以此来判断 聚类 是否已经收敛
				dC = 0.0;
				for (uint i = 0; i < result.centers.size(); i++)
					{
						dC = std::max(dC, std::abs(oldCentroids[i] - result.centers[i]));
					}

			}

		//fill the label Center Vector
		//先填充，其实没用,就是为了保持非空，之后再重新clear 重新 push_back
		for (int i = 0; i < result.labelsC.size(); i++)
			result.labelsC[i] = i + 1;

		return result;
	}

const int DepthHistogram::depthToBin(const double depth) const
	{
		float stepH = this->estStep() / 2;
		double histDepth = this->m_bins.rows * ((depth - this->m_minimum - stepH) / (this->m_maximum - this->m_minimum));

		return std::max(0, std::min(this->m_bins.rows - 1, cvRound(histDepth)));
	}

const double DepthHistogram::binToDepth(const float bin) const
	{

		float stepH = this->estStep() / 2;
		return ((bin * estStep()) + this->m_minimum + stepH);
	}

const int DepthHistogram::depthToLabel(const double depth, const std::vector<int> & labels) const
	{
		int bin = this->depthToBin(depth);

		return labels[bin];
	}

const int DepthHistogram::depthToPeak(const double depth, const std::vector<int> & peaks) const
	{
		//double histDepth = this->m_bins.rows * ( ( depth - this->m_minimum ) / ( this->m_maximum - this->m_minimum ) );
		double histDepth = this->depthToBin(depth);
		std::vector<double> peakTranslated(peaks.size());

		for (uint i = 0; i < peaks.size(); i++)
			{
				peakTranslated[i] = std::abs(peaks[i] - histDepth);
			}

		return std::min_element(peakTranslated.begin(), peakTranslated.end()) - peakTranslated.begin();
	}

const bool DepthHistogram::empty() const
	{
		return this->m_bins.empty();
	}

const size_t DepthHistogram::size() const
	{
		return this->m_bins.rows;
	}

const double DepthHistogram::minimum() const
	{
		return this->m_minimum;
	}

const float DepthHistogram::estStep() const
	{
		return this->estimatedStep;
	}

const double DepthHistogram::maximum() const
	{
		return this->m_maximum;
	}

const float DepthHistogram::operator[](const uint i) const
	{
		return this->m_bins(i);
	}

const DepthHistogram DepthHistogram::createHistogram(const uint step, const cv::Mat & region, const cv::Mat1b & mask)
	{
		cv::Mat1f region32f;
		DepthHistogram result;
		int histogramBinCount = 0;

		region.convertTo(region32f, CV_32F);
		std::cout << "step == " << step << "  mm    " << std::endl;
		//region32f= region32f.mul(mask);
		//std::cout << "region == " << std::endl << region << std::endl;
		cv::minMaxLoc(region32f, &result.m_minimum, &result.m_maximum, nullptr, nullptr, mask);

		//std::cout << "region32f == " << std::endl << region32f << std::endl;
		if (step == 0)
			{
				histogramBinCount = std::max(1, cvRound((result.m_maximum - result.m_minimum) / 50.0) + 1);    //modified here
				result.m_minimum -= 25;
				result.m_maximum += 25;
			}
		else
			{
				histogramBinCount = std::max(1, cvRound((result.m_maximum - result.m_minimum) / static_cast<double>(step)) + 1);    //modified here
				result.m_minimum -= static_cast<double>(step) / 2;
				result.m_maximum += static_cast<double>(step) / 2;
			}

		//change this bit with incrementing the number of bin
		//result.m_maximum += modelNoise( result.m_maximum, 0 );
		histogramBinCount++;
		//	std::cout << "histogramBinCount == " << histogramBinCount << std::endl;
		std::cout << "m_maximum == " << result.m_maximum << "  m_minimum" << result.m_minimum << std::endl;
		std::cout << "steps == " << step << "  mm    toatl is " << histogramBinCount << "  steps" << std::endl;
		int channels = 0;
		//上下限
		float hist_range[] = { static_cast<float>(result.m_minimum), static_cast<float>(result.m_maximum) };
		const float * hist_ranges[] = { hist_range };

		//cv::calcHist(&region32f, 1, &channels, cv::Mat(), result.m_bins, 1, &histogramBinCount, hist_ranges);
		cv::calcHist(&region32f, 1, &channels, mask, result.m_bins, 1, &histogramBinCount, hist_ranges);
		//std::cout << "result.m_bins == "  << result.m_bins ;
		/*   		std::cout << "patch size is " << region32f.cols * region32f.rows << std::endl;
		 std::cout << "result.m_bins size == " << result.m_bins.rows << std::endl;
		 std::cout << "step == " << step << "  Hist step is " << histogramBinCount << std::endl;
		 std::cout << "m_maximum == " << result.m_maximum << "  m_minimum" << result.m_minimum << std::endl;*/

		std::cout << std::endl;
/*		for (int a = 0; a < result.m_bins.rows; a++)
			{
				std::cout << result.m_bins(a, 0) << "  ";
			}
		std::cout << std::endl;*/
		result.estimatedStep = (1.0 / (float) result.size()) * (result.m_maximum - result.m_minimum);
		std::cout << "result.estimatedStep == " << result.estimatedStep << std::endl;
		return result;
	}

void DepthHistogram::visualise(const std::string & string)
	{
		visualiseHistogram(string, this->m_bins);
	}

const int DepthHistogram::depthToCentroid(const double depth, const std::vector<float> & centroids) const
	{
		//double histDepth = this->m_bins.rows * ( ( depth - this->m_minimum ) / ( this->m_maximum - this->m_minimum ) );
		double histDepth = this->depthToBin(depth);
		std::vector<double> peakTranslated(centroids.size());

		for (uint i = 0; i < centroids.size(); i++)
			{
				peakTranslated[i] = std::abs(centroids[i] - histDepth);
			}

		return std::min_element(peakTranslated.begin(), peakTranslated.end()) - peakTranslated.begin();
	}
