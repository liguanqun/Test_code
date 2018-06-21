

/*
This class represents a C++ implementation of the DS-KCF Tracker [1]. In particular
the weighted response based on depth data presented in [1] is performed within this class

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao, 
    DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/
#ifndef CFTRACKING_DEPTHWEIGHTKCFTRACKER_H
#define CFTRACKING_DEPTHWEIGHTKCFTRACKER_H

#include "kcf_tracker.hpp"

class DepthWeightKCFTracker : public KcfTracker
{
public:
  DepthWeightKCFTracker( KcfParameters paras, std::shared_ptr< Kernel > kernel );
  virtual ~DepthWeightKCFTracker();

  const DetectResult detect( const cv::Mat & image, const std::shared_ptr< FC > & features, const cv::Point_< double > & position, const double depth, const double std ) const;
private:
  int m_cellSize;
};


#endif //CFTRACKING_DEPTHWEIGHTKCFTRACKER_H
