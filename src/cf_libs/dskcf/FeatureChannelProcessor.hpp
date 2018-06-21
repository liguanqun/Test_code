#ifndef _FEATURECHANNELPROCESSOR_HPP_
#define _FEATURECHANNELPROCESSOR_HPP_


/*
This class represents a C++ implementation of the DS-KCF Tracker [1]. In particular
this class is a template class with virtual methods only. Other classes are derived from this class, such as
ColourFeatureChannelProcessor

References:
[1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
*/

#include <memory>
#include <vector>
#include <opencv2/core/core.hpp>

#include "feature_channels.hpp"

/**
 * FeatureChannelProcessor is responsible for taking a collection of feature channels and processing them.
 */
class FeatureChannelProcessor
{
public:
  /**
   * Concatenate the feature channels.
   *
   * @param featureChannels The input collection of feature channels to be processed.
   *
   * @returns A new collection of processed feature channels.
   */
  virtual const std::vector< std::shared_ptr< FC > > concatenate(
      const std::vector< std::shared_ptr< FC > > & featureChannels ) const = 0;

  /**
   * Process the frame so that each element of the resulting collection is associated with the element in the feature channel collection.
   *
   * @param frame The collection of images for the current frame.
   *
   * @returns A collection ordered so that each element matches its associated feature channel.
   */
  virtual const std::vector< cv::Mat > concatenate( const std::vector< cv::Mat > & frame ) const = 0;

  /**
   * Combine the maximum responses of each feature channel.
   *
   * @param maxResponses The maximum responses for each feature channel.
   *
   * @returns The combined maximum response.
   */
  virtual const double concatenate( const std::vector< double > & maxResponses ) const = 0;

  /**
   * Combine the positions of each tracker
   *
   * @param positions The positions given my each tracker's detect call
   *
   * @returns The combined position
   */
  virtual const cv::Point_< double > concatenate( const std::vector< cv::Point_< double > > & positions ) const = 0;
};

#endif
