#ifndef _SCALEANALYSIS_HPP_
#define _SCALEANALYSIS_HPP_


/*
This class represents a C++ implementation of the DS-KCF Tracker [1]. In particular
the scale change module presented in  [1] is performed within this class

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/

#include <array>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/core.hpp>

#include "DepthSegmenter.hpp"
#include "ScaleChangeObserver.hpp"
//This include file creates circular dependencies
//FIX ME!
//#include "kcf_tracker.hpp"
class KcfTracker;

/**
 * ScaleAnalyser implements the detection and handling of scale changes as describe in section 3.2 of \cite DSKCF.
 */
class ScaleAnalyser
{
public:
	ScaleAnalyser( DepthSegmenter * depthSegmenter, double padding );
	ScaleAnalyser( const std::vector< double > & scales, const double outputSigmaFactor, const int cellSize, double padding );

	cv::Rect_< double > init( const cv::Mat & image, const cv::Rect_< double > & boundingBox );
	cv::Rect_< double > update( const cv::Mat & image, const cv::Rect_< double > & boundingBox );

	double getScaleFactor() const;

	void registerScaleChangeObserver( ScaleChangeObserver * observer );
	void clearObservers();

	std::vector< std::shared_ptr< KcfTracker > > createModelScales( std::shared_ptr< KcfTracker > tracker );

	static cv::Mat2d scaleImageFourier( const cv::Mat2d & image, const cv::Size2i & size );
	static cv::Mat2d scaleImageFourierShift( const cv::Mat2d & image, const cv::Size2i & size );
private:
	size_t m_i;
	int m_cellSize;
	double m_padding;
	double m_outputSigmaFactor;
	double m_minStep;
	double m_step;
	double m_currentDepth;
	double m_initialDepth;
	double m_scaleFactor;

	DepthSegmenter * m_depthSegmenter;

	std::vector< double > m_scales;
	std::vector< cv::Size_< double > > m_windowSizes;
	std::vector< cv::Size_< double > > m_targetSizes;
	std::vector< cv::Point_< double > > m_targetPositions;
	std::vector< double > m_outputSigmas;
	std::vector< cv::Mat2d > m_yfs;
	std::vector< cv::Mat1d > m_cosineWindows;
	std::vector< ScaleChangeObserver* > m_observers;
};

#endif
