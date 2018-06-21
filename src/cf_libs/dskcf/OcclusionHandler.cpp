#include "OcclusionHandler.hpp"

#include <tbb/concurrent_vector.h>

OcclusionHandler::OcclusionHandler(KcfParameters paras, std::shared_ptr<Kernel> & kernel, std::shared_ptr<FeatureExtractor> & featureExtractor,
		std::shared_ptr<FeatureChannelProcessor> & featureProcessor)
	{
		this->m_paras = paras;
		this->m_kernel = kernel;
		this->m_featureExtractor = featureExtractor;
		this->m_featureProcessor = featureProcessor;
		this->m_depthSegmenter = std::make_shared<DepthSegmenter>();
		this->m_scaleAnalyser = std::make_shared < ScaleAnalyser > (this->m_depthSegmenter.get(), paras.padding);

		for (int i = 0; i < 2; i++)
			{
				this->m_targetTracker[i] = std::make_shared < DepthWeightKCFTracker > (paras, kernel);
			}

		this->m_occluderTracker = std::make_shared < KcfTracker > (paras, kernel);

		this->m_lambdaOcc = 0.35;
		this->m_lambdaR1 = 0.4;
		this->m_lambdaR2 = 0.2;
		this->m_isOccluded = false;

		this->singleFrameProTime = std::vector<int64>(8, 0);
	}

OcclusionHandler::~OcclusionHandler()
	{
		this->m_depthSegmenter = nullptr;
	}

void OcclusionHandler::init(const std::array<cv::Mat, 2> & frame, const Rect & target)
	{
		std::vector<std::shared_ptr<FC> > features(2);
		this->m_isOccluded = false;
		this->m_initialSize = target.size();

		//先清除 然后把 当前类 OcclusionHandle 和 两个 DepthWeightKCFTracker 类 登记到 ScaleAnalyser
		// 因为两个类 都继承了 ScaleChangeObserver 类
		this->m_scaleAnalyser->clearObservers();
		this->m_scaleAnalyser->registerScaleChangeObserver(this);
		this->m_scaleAnalyser->registerScaleChangeObserver(this->m_targetTracker[0].get());
		this->m_scaleAnalyser->registerScaleChangeObserver(this->m_targetTracker[1].get());

		this->m_depthSegmenter->init(frame[1], target);


		this->m_scaleAnalyser->init(frame[1], target);

		Point position = centerPoint(target);
		Rect window = boundingBoxFromPointSize(position, this->m_windowSize);



		std::cout << "target size is " << target.width << " * " << target.height << std::endl;
		std::cout << "this->m_targetSize  " << this->m_targetSize.width << " * " << this->m_targetSize.height << std::endl;
		std::cout << "this->m_windowSize  " << this->m_windowSize.width << " * " << this->m_windowSize.height << std::endl;
		std::cout << "this->m_cosineWindow  " << this->m_cosineWindow.cols << " * " << this->m_cosineWindow.rows << std::endl;
		//std::cout<<" objectMask  =="<<this->m_depthSegmenter->_ObjectMask<<std::endl;
//////***********************************************************************************//////
		int left_x = (int) ((this->m_windowSize.width - this->m_depthSegmenter->_ObjectMask.cols) / 2);
		int right_x = this->m_windowSize.width - left_x - 1;
		int top_y = (int) ((this->m_windowSize.height - this->m_depthSegmenter->_ObjectMask.rows) / 2);
		int down_y = this->m_windowSize.height - top_y - 1;

		cv::Mat weight_pre((int) this->m_windowSize.height, (int) this->m_windowSize.width, this->m_cosineWindow.type(), cv::Scalar::all(0));
		//	std::cout << "weight mat is " << weight_pre << std::endl;
		cv::Mat weight_for_show = frame[0].clone();
		cv::Mat weight_for_show_b = frame[0].clone();
		//	std::cout << "this->m_depthSegmenter->_ObjectMask == " << std::endl << this->m_depthSegmenter->_ObjectMask << std::endl;
		for (int y = 0; y < weight_pre.rows; y++)
			{
				for (int x = 0; x < weight_pre.cols; x++)
					{
						if (x >= left_x && x < right_x && y >= top_y && y < down_y)
							{
								weight_pre.at<double>(y, x) = (double) this->m_depthSegmenter->_ObjectMask(y - top_y, x - left_x);
							}
						if (weight_pre.at<double>(y, x) > 0)
							{
								// cv::Point p = cv::Point(window.y+4*cell_y+1,window.x+4*cell_y+1);
								weight_for_show.at<cv::Vec3b>(window.y + y, window.x + x)[2] = 255;
							}
					}
			}
		cv::namedWindow("weight_for_show", 0);
		cv::imshow("weight_for_show", weight_for_show);
		//	std::cout << "weight mat is " << weight_pre << std::endl;
		//
		int cell_width = this->m_cosineWindow.cols;
		int cell_height = this->m_cosineWindow.rows;
		cv::Mat weight(cell_height, cell_width, this->m_cosineWindow.type(), cv::Scalar::all(1));
		for (int cell_y = 0; cell_y < cell_height; cell_y++)
			{
				for (int cell_x = 0; cell_x < cell_width; cell_x++)
					{
						for (int i = 0; i < 4; i++)
							{
								for (int j = 0; j < 4; j++)
									{
										weight.at<double>(cell_y, cell_x) += weight_pre.at<double>(cell_y * 4 + i, cell_x * 4 + j) / 16;
									}
							}
						if (weight.at<double>(cell_y, cell_x) > 1)
							{
								// cv::Point p = cv::Point(window.y+4*cell_y+1,window.x+4*cell_y+1);
								weight_for_show_b.at<cv::Vec3b>(window.y + cell_y * 4 + 3, window.x + cell_x * 4 + 3)[2] = 255;
							}

					}
			}
		cv::namedWindow("weight_for_show_b", 0);
		cv::imshow("weight_for_show_b", weight_for_show_b);
		std::cout << "weight cols*rows" << weight.cols << " * " << weight.rows << std::endl;
		//	std::cout<<"finnaly weight == "<<std::endl<<weight<<std::endl;
		this->m_weight = weight;

///*********************************************************///
		//Extract features
		for (int i = 0; i < 2; i++)
			{
				//提取HOG特征，然后加cos窗
				features[i] = this->m_featureExtractor->getFeatures(frame[i], window);
				FC::mulFeatures(features[i], this->m_cosineWindow);

				FC::mulFeatures(features[i], this->m_weight);
			}
		//

		features = this->m_featureProcessor->concatenate(features);

		for (uint i = 0; i < features.size(); i++)
			{

				this->m_targetTracker[i]->init(frame[i], features[i], position);
			}

		this->m_filter.initialise(position);

	}

const Rect OcclusionHandler::detect(const std::array<cv::Mat, 2> & frame, const Point & position)
	{
		if (this->m_isOccluded)
			{
				return this->occludedDetect(frame, position);
			}
		else
			{
				return this->visibleDetect(frame, position);
			}
	}

void OcclusionHandler::update(const std::array<cv::Mat, 2> & frame, const Point & position)
	{
		if (this->m_isOccluded)
			{
				return this->occludedUpdate(frame, position);
			}
		else
			{
				return this->visibleUpdate(frame, position);
			}
	}

const Rect OcclusionHandler::visibleDetect(const std::array<cv::Mat, 2> & frame, const Point & position)
	{

		int64 tStartDetection = cv::getTickCount();
		std::vector<double> responses;
		std::vector<std::shared_ptr<FC> > features(2);
		std::vector<Point> positions;

		Rect target = boundingBoxFromPointSize(position, this->m_targetSize);
		Rect window = boundingBoxFromPointSize(position, this->m_windowSize);

		tbb::parallel_for<uint>(0, 2, 1, [this,&frame,&features,&window]( uint index ) -> void
			{
				features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
				FC::mulFeatures( features[ index ], this->m_cosineWindow );		// this->m_weight
				FC::mulFeatures( features[ index ], this->m_weight );

			});

		features = this->m_featureProcessor->concatenate(features);
		std::vector<cv::Mat> frames_ = this->m_featureProcessor->concatenate(std::vector<cv::Mat>(frame.begin(), frame.end()));

		for (uint i = 0; i < features.size(); i++)
			{
				DetectResult result = this->m_targetTracker[i]->detect(frames_[i], features[i], position, this->m_depthSegmenter->getTargetDepth(), this->m_depthSegmenter->getTargetSTD());
				positions.push_back(result.position);
				responses.push_back(result.maxResponse);
			}
		//here the maximun response is calculated....
		int64 tStopDetection = cv::getTickCount();
		this->singleFrameProTime[0] = tStopDetection - tStartDetection;

		int64 tStartSegment = tStopDetection;
		//TO BE CHECKED IN CASE OF MULTIPLE MODELS...LINEAR ETC....WORKS ONLY FOR SINGLE (or concatenate) features
		//取出的是 利用深度图的KCF估计出的位置 即positions[1]
		target = boundingBoxFromPointSize(positions.back(), this->m_targetSize);
		//*****************************
		//Point position_new = centerPoint(target);
		//Rect window_new = boundingBoxFromPointSize(position_new, this->m_windowSize);
		//int bin = this->m_depthSegmenter->update(frame[1], window_new);
		//****************************
		int bin = this->m_depthSegmenter->update(frame[1], target);

		DepthHistogram histogram = this->m_depthSegmenter->getHistogram();

		double totalArea = target.area() * 1.05;

		if (this->evaluateOcclusion(histogram, bin, this->m_featureProcessor->concatenate(responses), totalArea))
			{
				std::cout << "*********Occlusionhandle funtion think into occlusion****************" << std::endl;
				//here the maximun response is calculated....
				int64 tStopSegment = cv::getTickCount();
				this->singleFrameProTime[1] = tStopSegment - tStartSegment;

				int64 tStartNewTracker = tStopSegment;

				const Rect retRect = this->onOcclusion(frame, features, target);

				int64 tStopNewTracker = cv::getTickCount();
				this->singleFrameProTime[2] = tStopNewTracker - tStartNewTracker;

				return retRect;

			}
		else
			{
				//here the maximun response is calculated....
				int64 tStopSegment = cv::getTickCount();
				this->singleFrameProTime[1] = tStopSegment - tStartSegment;
				//Is the object entirely unoccluded? 目标没有被完全遮挡
				if (!this->m_depthSegmenter->isOccluded())
					{

					}
				//判定没有遮挡之后使用KCF估计出的位置
				//取出的是RGB图的KCF估计出的位置 即positions[0]
				Point estimate = this->m_featureProcessor->concatenate(positions);

				estimate.x = (estimate.x - this->m_targetSize.width / 2) < frame[0].cols ? estimate.x : this->m_targetSize.width;
				estimate.y = (estimate.y - this->m_targetSize.height / 2) < frame[0].rows ? estimate.y : this->m_targetSize.height;
				estimate.x = (estimate.x + this->m_targetSize.width / 2) > 0 ? estimate.x : 1;
				estimate.y = (estimate.y + this->m_targetSize.height / 2) > 0 ? estimate.y : 1;
				return boundingBoxFromPointSize(estimate, this->m_initialSize * this->m_scaleAnalyser->getScaleFactor());

			}
	}

void OcclusionHandler::visibleUpdate(const std::array<cv::Mat, 2> & frame, const Point & position)
	{
		//EVALUATE CHANGE OF SCALE....
		int64 tStartScaleCheck = cv::getTickCount();
		std::vector<std::shared_ptr<FC> > features(2);
		Rect window = boundingBoxFromPointSize(position, this->m_windowSize);

		//给 OcclusionHandler::onScaleChange 传入了参数
		this->m_scaleAnalyser->update(frame[1], window);

		int64 tStopScaleCheck = cv::getTickCount();
		this->singleFrameProTime[5] = tStopScaleCheck - tStartScaleCheck;

		int64 tStartModelUpdate = tStopScaleCheck;
		window = boundingBoxFromPointSize(position, this->m_windowSize);
		//////***********************************************************************************//////

		int left_x = (int) ((this->m_windowSize.width - this->m_depthSegmenter->_ObjectMask.cols) / 2);
		int right_x = this->m_windowSize.width - left_x - 1;
		int top_y = (int) ((this->m_windowSize.height - this->m_depthSegmenter->_ObjectMask.rows) / 2);
		int down_y = this->m_windowSize.height - top_y - 1;

		cv::Mat weight_pre((int) this->m_windowSize.height, (int) this->m_windowSize.width, this->m_cosineWindow.type(), cv::Scalar::all(0));
		//	std::cout << "weight mat is " << weight_pre << std::endl;
		cv::Mat weight_for_show = frame[0].clone();
		cv::Mat weight_for_show_b = frame[0].clone();
		//	std::cout << "this->m_depthSegmenter->_ObjectMask == " << std::endl << this->m_depthSegmenter->_ObjectMask << std::endl;
		for (int y = 0; y < weight_pre.rows; y++)
			{
				for (int x = 0; x < weight_pre.cols; x++)
					{
						if (x >= left_x && x < right_x && y >= top_y && y < down_y)
							{
								weight_pre.at<double>(y, x) = (double) this->m_depthSegmenter->_ObjectMask(y - top_y, x - left_x);
							}
						if (weight_pre.at<double>(y, x) > 0)
							{
								// cv::Point p = cv::Point(window.y+4*cell_y+1,window.x+4*cell_y+1);
								weight_for_show.at<cv::Vec3b>(window.y + y, window.x + x)[2] = 255;
							}

					}
			}

		cv::namedWindow("weight_for_show", 0);
		cv::imshow("weight_for_show", weight_for_show);
		//	std::cout << "weight mat is " << weight_pre << std::endl;
		//
		int cell_width = this->m_cosineWindow.cols;
		int cell_height = this->m_cosineWindow.rows;
		cv::Mat weight(cell_height, cell_width, this->m_cosineWindow.type(), cv::Scalar::all(1));
		for (int cell_y = 0; cell_y < cell_height; cell_y++)
			{
				for (int cell_x = 0; cell_x < cell_width; cell_x++)
					{
						for (int i = 0; i < 4; i++)
							{
								for (int j = 0; j < 4; j++)
									{
										weight.at<double>(cell_y, cell_x) += weight_pre.at<double>(cell_y * 4 + i, cell_x * 4 + j) / 16;
									}
							}
						if (weight.at<double>(cell_y, cell_x) > 1)
							{
								// cv::Point p = cv::Point(window.y+4*cell_y+1,window.x+4*cell_y+1);
								weight_for_show_b.at<cv::Vec3b>(window.y + cell_y * 4 + 3, window.x + cell_x * 4 + 3)[2] = 255;
							}
					}
			}
		cv::namedWindow("weight_for_show_b", 0);
		cv::imshow("weight_for_show_b", weight_for_show_b);
		std::cout << "weight cols*rows" << weight.cols << " * " << weight.rows << std::endl;
		std::cout << "this->m_cosineWindow  " << this->m_cosineWindow.cols << " * " << this->m_cosineWindow.rows << std::endl;
		//		std::cout<<"finnaly weight == "<<std::endl<<weight<<std::endl;
		this->m_weight = weight;

		///*********************************************************///
		std::cout << "crrent m_cosineWindow is " << this->m_cosineWindow.cols << " * " << this->m_cosineWindow.rows << std::endl;
		tbb::parallel_for<uint>(0, 2, 1, [this,&frame,&features,&window]( uint index ) -> void
			{
				features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
				FC::mulFeatures( features[ index ], this->m_cosineWindow );
				FC::mulFeatures( features[ index ], this->m_weight );

			});

		features = this->m_featureProcessor->concatenate(features);

		for (size_t i = 0; i < features.size(); i++)
			{
				this->m_targetTracker[i]->update(frame[i], features[i], position);
			}

		int64 tStopModelUpdate = cv::getTickCount();
		this->singleFrameProTime[6] = tStopModelUpdate - tStartModelUpdate;

	}

const Rect OcclusionHandler::occludedDetect(const std::array<cv::Mat, 2> & frame, const Point & position, float smallAreaFraction)
	{
		int64 tStartTrackOccluder = cv::getTickCount();

		Rect imageRect(0, 0, frame[1].cols, frame[1].rows);
		Rect window = rectRound(boundingBoxFromPointSize(position, this->m_occluderWindowSize));
		Rect target = rectRound(boundingBoxFromPointSize(position, this->m_occluderSize)); //old occluder position

		Point prediction = position;

		//Extract features
		auto features = this->m_featureExtractor->getFeatures(frame[0], window);
		FC::mulFeatures(features, m_occluderCosineWindow);

		Rect oldSearchWindow = this->m_searchWindow; //old occluder search window
		//mix it with the predicted position
		Rect newSearchWindow = extremeRect(boundingBoxFromPointSize(prediction, this->m_targetSize), extremeRect(oldSearchWindow, target));

		//move here the response of the tracker....then test the
		Rect result = boundingBoxFromPointSize(this->m_occluderTracker->detect(frame[0], features, position).position, this->m_occluderSize);
		int64 tStopTrackOccluder = cv::getTickCount();
		this->singleFrameProTime[3] = tStopTrackOccluder - tStartTrackOccluder;

		//START COUNTER FOR SOLVING OCCLUSIONS....
		int64 tStartSolveOcclusions = tStopTrackOccluder;

		this->m_searchWindow = extremeRect(newSearchWindow, result) & imageRect;
		this->m_searchWindow = resizeBoundingBox(this->m_searchWindow, this->m_searchWindow.size()) & imageRect;
		Rect areaToSegment = resizeBoundingBox(this->m_searchWindow, this->m_searchWindow.size() * 1.05) & imageRect;

		//substitute the next three lines with a better segmentation
		cv::Mat1w area(frame[1], areaToSegment);

		cv::Size size = this->m_targetSize;
		int tmpWidth = static_cast<int>(size.height);
		int tmpHeight = static_cast<int>(size.height);
		int minimumArea = cvRound(smallAreaFraction * tmpWidth * tmpHeight);
		cv::Mat1b objectMask(frame[1].rows, frame[1].cols);
		objectMask.setTo(0);
		cv::Mat1b tmpObjectMask = objectMask(areaToSegment);
		cv::Rect_<double> objMaskRect;

		std::vector<float> centersCandidate;

		std::vector<cv::Point_<double> > candidates = this->m_depthSegmenter->segmentOccluder(area, Rect(0, 0, area.cols, area.rows), minimumArea, tmpObjectMask, centersCandidate, objMaskRect);

		//re-update here the window search with the segmented occluder....
		objMaskRect.x += areaToSegment.x;
		objMaskRect.y += areaToSegment.y;
		this->m_searchWindow = extremeRect(objMaskRect, this->m_searchWindow) & imageRect;

		std::for_each(candidates.begin(), candidates.end(), [ this, areaToSegment ]( cv::Point_< double > & candidate ) -> void
			{	candidate += areaToSegment.tl();});  //this->m_searchWindow.tl(); } );

		auto itr = this->findBestCandidateRegion(frame, candidates, centersCandidate);
		if (itr == candidates.end())
			{ //仍在遮挡
			  //MOVED UP!!!!!!

				//Rect result = boundingBoxFromPointSize( this->m_occluderTracker->detect( frame[ 0 ], features, position ).position, this->m_occluderSize );
				//this->m_searchWindow = extremeRect( extremeRect( this->m_searchWindow, result ), boundingBoxFromPointSize( prediction, this->m_targetSize ) ) & imageRect;

				int64 tStopSolveOcclusions = cv::getTickCount();
				this->singleFrameProTime[4] = tStopSolveOcclusions - tStartSolveOcclusions;
				return result;
			}
		else
			{				//
							//check now if the bounding box belong to the occluder mask or not
				cv::Point shiftedPoint = pointRound(*itr);
				bool onMaskPixel = (objectMask(shiftedPoint) > 0);
				objectMask.release();
				if (onMaskPixel)
					{
						this->m_isOccluded = true;
						int64 tStopSolveOcclusions = cv::getTickCount();
						this->singleFrameProTime[4] = tStopSolveOcclusions - tStartSolveOcclusions;
						return result;
					}

				return boundingBoxFromPointSize(*itr, this->m_targetSize);
			}
	}

void OcclusionHandler::occludedUpdate(const std::array<cv::Mat, 2> & frame, const Point & position)
	{
		int64 tStartTrackOccluder = cv::getTickCount();

		Size paddedSize = sizeRound(this->m_occluderSize * this->m_paras.padding);
		Rect window = boundingBoxFromPointSize(position, paddedSize);

		auto features = this->m_featureExtractor->getFeatures(frame[0], window);
		FC::mulFeatures(features, m_occluderCosineWindow);

		this->m_occluderTracker->update(frame[0], features, position);

		int64 tStopTrackOccluder = cv::getTickCount();
		int64 newInterval = tStopTrackOccluder - tStartTrackOccluder;
		this->singleFrameProTime[3] += newInterval;
	}

const Rect OcclusionHandler::onOcclusion(const std::array<cv::Mat, 2> & frame, std::vector<std::shared_ptr<FC> > & features, const Rect & boundingBox)
	{
		cv::Mat im_show = frame[0].clone();

		this->m_isOccluded = true;
		cv::Scalar mean, stddev;
		Rect imageRect(0, 0, frame[0].cols, frame[0].rows);

		cv::rectangle(im_show, boundingBox.tl(), boundingBox.br(), cv::Scalar(0, 255, 0));
		cv::imshow("im_show", im_show);
		//cv::waitKey(0);

		Rect boundingBoxModified = getSubWindowRounding(boundingBox);

		Rect window = resizeBoundingBox(boundingBoxModified, this->m_windowSize);

		cv::rectangle(im_show, window.tl(), window.br(), cv::Scalar(0, 255, 0));
		cv::imshow("im_show", im_show);
		//cv::waitKey(0);

		//Find the region of the window belonging to the occluder
		auto L = this->m_depthSegmenter->getLabeledImage();

		//You must now select the closest and the biggest area!!!!
		DepthHistogram::Labels labels = this->m_depthSegmenter->getLabelsResults();
		int indexCenter = this->m_depthSegmenter->selectClosestObject(labels.centers);
		cv::Mat occluder = createMask<uchar>(L, labels.labelsC[indexCenter], false);

		//std::cout<<"occluder is "<<std::endl<<occluder<<std::endl;
		cv::Mat occluder_show = occluder.clone();
		occluder_show = occluder_show * 255;
		cv::namedWindow("occluder_show", 0);
		cv::imshow("occluder_show", occluder_show);
		//	cv::waitKey(0);

		//cv::meanStdDev( cv::Mat( frame[ 1 ], boundingBoxModified & imageRect ), mean, stddev );
		cv::meanStdDev(cv::Mat(frame[1], boundingBoxModified & imageRect), mean, stddev, occluder);
		stddev.val[0] = modelNoise(mean.val[0], stddev.val[0]);

		//从整张深度图上标记出那些点的深度值在目标深度的标准差范围内
		occluder = getRegion<ushort>(cv::Mat(frame[1], imageRect), cvFloor(mean.val[0] - stddev.val[0]), cvCeil(mean.val[0] + stddev.val[0]));

		//now filter out small regions and keep the one with maximum overlap....
		//把所有满足的点连接起来，把矩形块存到 occlusionCandidates里
		auto occlusionCandidates = connectedComponents<uchar>(occluder);

		//Find the bounding box that has the largest overlapping region with the target, excluding the background.
		//在所有的矩形中找到跟目标区域重叠面积最大的矩形 occluderBBCandidate
		cv::Rect_<int> occluderBBCandidate = *(std::max_element(occlusionCandidates.begin() + 1, occlusionCandidates.end(),
				[ boundingBoxModified ]( const cv::Rect_< int > & a, const cv::Rect_< int > & b ) -> bool
					{
						cv::Rect_< int > bb = rectRound( boundingBoxModified );
						return ( a & bb ).area() < ( b & bb ).area();
					}));

		cv::rectangle(im_show, occluderBBCandidate.tl(), occluderBBCandidate.br(), cv::Scalar(255, 0, 0));
		cv::imshow("im_show", im_show);
		//cv::waitKey(0);

		//找到 遮挡物和 目标调整后 的 重叠区域
		cv::Rect_<int> occluderBB = rectRound(window) & occluderBBCandidate;

		cv::rectangle(im_show, occluderBB.tl(), occluderBB.br(), cv::Scalar(0, 255, 0));
		cv::imshow("im_show", im_show);
		//cv::waitKey(0);

		//找到包含 矩形occluderBB 和 矩形boundingBox 最小的矩形
		// this->m_searchWindow 遮挡之后的搜索窗口
		this->m_searchWindow = extremeRect<double>(rectCast<double>(occluderBB), boundingBox);

		cv::rectangle(im_show, this->m_searchWindow.tl(), this->m_searchWindow.br(), cv::Scalar(0, 0, 255));
		cv::imshow("im_show", im_show);
		//cv::waitKey(0);

		if (occluderBB.area() > 0)
			//Initialise the occluder tracker with the new bounding box
			//初始化的是 RGB 的 KCF，跟踪 遮挡物和目标区域2.5倍的重叠部分
			//为什么是重叠部分不是整个遮挡物？？？
			this->initialiseOccluder(frame[0], occluderBB);
		else
			{
				//printf("ERROR\n");
				//遮挡物体的面积小于0，，认为没有遮挡
				this->m_isOccluded = false;
			}
		//Store the target object's depth and std
		this->m_targetDepthSTD = this->m_depthSegmenter->getTargetSTD();
		this->m_targetDepthMean = this->m_depthSegmenter->getTargetDepth();

		//Output the occluders bounding box
		//
		return occluderBB;
	}

void OcclusionHandler::onVisible(const std::array<cv::Mat, 2> & frame, std::vector<std::shared_ptr<FC> > & features, const Point & position)
	{
		this->m_isOccluded = false;
	}

std::vector<Point> OcclusionHandler::findCandidateRegions(const cv::Mat1d & depth, const double targetDepth, const double targetSTD, const cv::Mat1b occluderMask)
	{
		cv::Mat1b candidates = cv::Mat1b::zeros(depth.rows, depth.cols);
		std::vector<Point> result;

		if (targetSTD != 0.0)
			{
				for (int row = 0; row < depth.rows; row++)
					{
						for (int col = 0; col < depth.cols; col++)
							{
								if ((depth(row, col) > (targetDepth - 2 * targetSTD)) && (depth(row, col) < (targetDepth + 2 * targetSTD)) && (occluderMask(row, col) != 0))
									{
										candidates(row, col) = 1;
									}
								else
									{
										candidates(row, col) = 0;
									}
							}
					}

				auto rects = connectedComponents<uchar>(candidates);

				std::for_each(rects.begin(), rects.end(), [&result]( const cv::Rect_< double > & rect )
					{	result.push_back( centerPoint( rect ) );});
			}

		return result;
	}

std::vector<cv::Point_<double> >::iterator OcclusionHandler::findBestCandidateRegion(const std::array<cv::Mat, 2> & frame, std::vector<cv::Point_<double> > & candidates)
	{
		if (candidates.size() > 0)
			{
				std::vector<std::vector<cv::Point_<double> >::iterator> iterators;
				for (auto itr = candidates.begin(); itr != candidates.end(); itr++)
					{
						iterators.push_back(itr);
					}

				tbb::concurrent_vector<ThreadResult> scoredCandidates;
				tbb::parallel_for_each(iterators.begin(), iterators.end(), [this, &frame, &scoredCandidates]( std::vector< cv::Point_< double > >::iterator & candidate ) -> void
					{
						scoredCandidates.push_back( this->scoreCandidate( frame, candidate ) );
					});

				auto max_elem = *std::max_element(scoredCandidates.begin(), scoredCandidates.end());

				if ((max_elem.score > this->m_lambdaR2))
					{
						this->m_isOccluded = false;
						return max_elem.value;
					}
			}

		return candidates.end();
	}

std::vector<cv::Point_<double> >::iterator OcclusionHandler::findBestCandidateRegion(const std::array<cv::Mat, 2> & frame, std::vector<cv::Point_<double> > & candidates,
		std::vector<float> &centersCandidate)
	{
		if (candidates.size() > 0)
			{
				std::vector<std::vector<cv::Point_<double> >::iterator> iterators;
				for (auto itr = candidates.begin(); itr != candidates.end(); itr++)
					{
						iterators.push_back(itr);
					}

				tbb::concurrent_vector<ThreadResult> scoredCandidates;

				tbb::parallel_for<int>(0, iterators.size(), 1, [ this, &frame, &scoredCandidates, &iterators, &centersCandidate ]( int i ) -> void
					{
						scoredCandidates.push_back( this->scoreCandidate( frame, iterators[ i ], centersCandidate[ i ] ) );
					});

				auto max_elem = *std::max_element(scoredCandidates.begin(), scoredCandidates.end());

				if ((max_elem.score > this->m_lambdaR2))
					{
						this->m_isOccluded = false;
						return max_elem.value;
					}
			}

		return candidates.end();
	}

bool OcclusionHandler::evaluateOcclusion(const DepthHistogram & histogram, const int objectBin, const double maxResponse)
	{
		// ( f(z)_max < λ_r1 ) ∧ ( Φ( Ω_obj ) > λ_occ l)
		//std::cout << "Φ( Ω_obj )  == " << phi(histogram, objectBin) << "  KCF max Response ==  " << maxResponse << std::endl;
		return ((maxResponse < this->m_lambdaR1) && (this->phi(histogram, objectBin) > this->m_lambdaOcc));
	}

bool OcclusionHandler::evaluateOcclusion(const DepthHistogram & histogram, const int objectBin, const double maxResponse, const double totalArea)
	{
		// ( f(z)_max < λ_r1 ) ∧ ( Φ( Ω_obj ) > λ_occ l)
		//	std::cout << "Φ( Ω_obj )  == " << phi(histogram, objectBin, totalArea) << "  KCF max maxResponse ==  " << maxResponse << std::endl;
		return ((maxResponse < this->m_lambdaR1) && (this->phi(histogram, objectBin, totalArea) > this->m_lambdaOcc));
	}

bool OcclusionHandler::evaluateVisibility(const DepthHistogram & histogram, const int objectBin, const double maxResponse) const
	{
		//( f(z)_n > λ_r2 ) ∧ ( Φ( Ω_Tbc ) < λ_occ )
		return ((maxResponse > this->m_lambdaR2) && (this->phi(histogram, objectBin) < this->m_lambdaOcc));
	}

double OcclusionHandler::phi(const DepthHistogram & histogram, const int objectBin) const
	{
		double totalArea = 0.0;
		double occluderArea = 0.0;

		for (uint i = 0; i < histogram.size(); i++)
			{
				if (i < objectBin)
					{
						occluderArea += histogram[i];
					}

				totalArea += histogram[i];
			}
		//std::cout<<"select object Bin  == "<<objectBin<<std::endl;
		//	std::cout << "occluderArea == " << occluderArea << "   totalArea == " << totalArea << std::endl;

		return occluderArea / totalArea;
	}

double OcclusionHandler::phi(const DepthHistogram & histogram, const int objectBin, const double totalArea) const
	{

		double occluderArea = 0.0;

		for (uint i = 0; i < histogram.size(); i++)
			{
				if (i < objectBin)
					{
						occluderArea += histogram[i];
					}
				else
					break;
			}
		//std::cout<<"select object Bin == "<<objectBin<<std::endl;
		//	std::cout << "occluderArea == " << occluderArea << "   totalArea == " << totalArea << std::endl;
		return occluderArea / totalArea;
	}
void OcclusionHandler::onScaleChange(const Size & targetSize, const Size & windowSize, const cv::Mat2d & yf, const cv::Mat1d & cosineWindow)
	{
		if (!this->m_isOccluded)
			{
				this->m_targetSize = targetSize;
				this->m_windowSize = windowSize;
				this->m_cosineWindow = cosineWindow;
			}
	}

void OcclusionHandler::initialiseOccluder(const cv::Mat & frame, const Rect boundingBox)
	{
		cv::Mat2d yf;

		this->m_occluderSize = boundingBox.size();
		this->m_occluderWindowSize = sizeRound(boundingBox.size() * this->m_paras.padding);
		Rect window = resizeBoundingBox(boundingBox, this->m_occluderWindowSize);
		Point position = centerPoint(boundingBox);

		//Create all of the parameters normally created by the scale analyser
		double outputSigma = sqrt(boundingBox.area()) * this->m_paras.outputSigmaFactor / this->m_paras.cellSize;

		cv::dft(gaussianShapedLabelsShifted2D(outputSigma, sizeFloor(window.size() * (1.0 / static_cast<double>(this->m_paras.cellSize)))), yf, cv::DFT_COMPLEX_OUTPUT);

		this->m_occluderCosineWindow = hanningWindow<double>(yf.rows) * hanningWindow<double>(yf.cols).t();

		//Extract features
		auto features = this->m_featureExtractor->getFeatures(frame, window);
		FC::mulFeatures(features, this->m_occluderCosineWindow);

		//Setup the occluder tracker
		this->m_occluderTracker = std::make_shared < KcfTracker > (this->m_paras, this->m_kernel);
		this->m_occluderTracker->onScaleChange(boundingBox.size(), window.size(), yf, this->m_occluderCosineWindow);
		this->m_occluderTracker->init(frame, features, position);
	}

const bool OcclusionHandler::isOccluded() const
	{
		return this->m_isOccluded;
	}

ThreadResult OcclusionHandler::scoreCandidate(const std::array<cv::Mat, 2> & frame, std::vector<cv::Point_<double> >::iterator candidate) const
	{
		cv::Mat patch[2];
		ThreadResult result;
		result.score = 0.0;
		result.value = candidate;

		std::vector<std::shared_ptr<FC> > features(2);
		std::vector<double> maxResponses;
		std::vector<Point> positions;
		Rect window = boundingBoxFromPointSize(*result.value, this->m_windowSize);

		tbb::parallel_for<uint>(0, 2, 1, [this,&frame,&features,&window]( uint index ) -> void
			{
				features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
				FC::mulFeatures( features[ index ], this->m_cosineWindow );
			});

		//Concat features and images here...
		features = this->m_featureProcessor->concatenate(features);
		std::vector<cv::Mat> frames_ = this->m_featureProcessor->concatenate(std::vector<cv::Mat>(frame.begin(), frame.end()));

		//Calculate the response of the target tracker at the candidate point
		for (uint i = 0; i < features.size(); i++)
			{
				DetectResult detection = this->m_targetTracker[i]->detect(frames_[i], features[i], *result.value, this->m_targetDepthMean, this->m_targetDepthSTD);
				positions.push_back(detection.position);
				maxResponses.push_back(detection.maxResponse);
			}

		*result.value = this->m_featureProcessor->concatenate(positions);
		double maxResponse = this->m_featureProcessor->concatenate(maxResponses);

		if (getSubWindow(frame[0], patch[0], this->m_targetSize, *result.value) && getSubWindow(frame[1], patch[1], this->m_targetSize, *result.value))
			{
				cv::Mat1b mask = createMask<ushort>(patch[1], 0);
				DepthHistogram histogram = DepthHistogram::createHistogram(cvRound(modelNoise(this->m_targetDepthMean, this->m_targetDepthSTD)), patch[1], mask);
				std::vector<int> peaks = histogram.getPeaks();

				//modification here to fit the matlab version

				int peak = histogram.depthToPeak(this->m_targetDepthMean, peaks);

				//Check to see if the response and depth are good enough to be the target objects
				if ((peak <= 1) && this->evaluateVisibility(histogram, histogram.depthToBin(this->m_targetDepthMean - this->m_targetDepthSTD), maxResponse))
					{
						result.score = maxResponse;
					}
			}

		return result;
	}

ThreadResult OcclusionHandler::scoreCandidate(const std::array<cv::Mat, 2> & frame, std::vector<cv::Point_<double> >::iterator candidate, float &candidateCenterDepth) const
	{
		cv::Mat patch[2];
		ThreadResult result;
		result.score = 0.0;
		result.value = candidate;

		std::vector<std::shared_ptr<FC> > features(2);
		std::vector<double> maxResponses;
		std::vector<Point> positions;
		Rect window = boundingBoxFromPointSize(*result.value, this->m_windowSize);

		tbb::parallel_for<uint>(0, 2, 1, [this,&frame,&features,&window]( uint index ) -> void
			{
				features[ index ] = this->m_featureExtractor->getFeatures( frame[ index ], window );
				FC::mulFeatures( features[ index ], this->m_cosineWindow );
			});

		//Concat features and images here...
		features = this->m_featureProcessor->concatenate(features);
		std::vector<cv::Mat> frames_ = this->m_featureProcessor->concatenate(std::vector<cv::Mat>(frame.begin(), frame.end()));

		//Calculate the response of the target tracker at the candidate point
		for (uint i = 0; i < features.size(); i++)
			{
				DetectResult detection = this->m_targetTracker[i]->detect(frames_[i], features[i], *result.value, this->m_targetDepthMean, this->m_targetDepthSTD);
				positions.push_back(detection.position);
				maxResponses.push_back(detection.maxResponse);
			}

		*result.value = this->m_featureProcessor->concatenate(positions);
		double maxResponse = this->m_featureProcessor->concatenate(maxResponses);

		if (getSubWindow(frame[0], patch[0], this->m_targetSize, *result.value) && getSubWindow(frame[1], patch[1], this->m_targetSize, *result.value))
			{
				cv::Mat1b mask = createMask<ushort>(patch[1], 0);
				DepthHistogram histogram = DepthHistogram::createHistogram(cvRound(modelNoise(this->m_targetDepthMean, this->m_targetDepthSTD)), patch[1], mask);
				std::vector<int> peaks = histogram.getPeaks();

				//modification here to fit the matlab version
				int peak = histogram.depthToPeak((double) candidateCenterDepth, peaks);

				//Check to see if the response and depth are good enough to be the target objects
				if ((peak <= 1) && this->evaluateVisibility(histogram, histogram.depthToBin((double) candidateCenterDepth - this->m_targetDepthSTD), maxResponse))
					{
						result.score = maxResponse;
					}
			}

		return result;
	}

const bool ThreadResult::operator<(const ThreadResult & rval) const
	{
		return this->score < rval.score;
	}
