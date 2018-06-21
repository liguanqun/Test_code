#ifndef _HOGFEATURES_HPP_
#define _HOGFEATURES_HPP_



/*
 // Original file: https://github.com/klahaag/cf_tracking/blob/master/src/cf_libs/kcf/kcf_tracker.hpp
 // + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
 // * We modified the original code of  Klaus Haag, such that different classes are used for the different KCF components
 //   In this case this class is used to calculate the HOG features as in [1] for the KCF core filtering

 It is implemented closely to the Matlab implementation by the original authors:
 http://home.isr.uc.pt/~henriques/circulant/
 However, some implementation details differ and some difference in performance
 has to be expected.

 This specific implementation features the scale adaption, sub-pixel
 accuracy for the correlation response evaluation and a more robust
 filter update scheme [2] used by Henriques, et al. in the VOT Challenge 2014.

 As default scale adaption, the tracker uses the 1D scale filter
 presented in [3]. The scale filter can be found in scale_estimator.hpp.
 Additionally, target loss detection is implemented according to [4].

 Every complex matrix is as default in CCS packed form:
 see : https://software.intel.com/en-us/node/504243
 and http://docs.opencv.org/modules/core/doc/operations_on_arrays.html

 References:
 [1] J. Henriques, et al.,
 "High-Speed Tracking with Kernelized Correlation Filters,"
 PAMI, 2015.

 [2] M. Danelljan, et al.,
 �Adaptive Color Attributes for Real-Time Visual Tracking,�
 in Proc. CVPR, 2014.

 [3] M. Danelljan,
 "Accurate Scale Estimation for Robust Visual Tracking,"
 Proceedings of the British Machine Vision Conference BMVC, 2014.

 [4] D. Bolme, et al.,
 �Visual Object Tracking using Adaptive Correlation Filters,�
 in Proc. CVPR, 2010.
 */

#include "FeatureExtractor.hpp"
#include "feature_channels.hpp"

class HOGFeatureExtractor: public FeatureExtractor
	{
	public:
		HOGFeatureExtractor();
		virtual ~HOGFeatureExtractor();

		virtual std::shared_ptr<FC> getFeatures(const cv::Mat & image, const cv::Rect_<double> & boundingBox) const;
	private:
		int m_cellSize;
	};

#endif
