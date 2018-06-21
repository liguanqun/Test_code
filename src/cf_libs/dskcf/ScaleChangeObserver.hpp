#ifndef _SCALECHANGEOBSERVER_HPP_
#define _SCALECHANGEOBSERVER_HPP_

/*
 This class represents a C++ implementation of the DS-KCF Tracker [1]. In particular
 the scaling system presented in [1] is implemented within this class

 References:
 [1] S. Hannuna, M. Camplani, J. Hall, M. Mirmehdi, D. Damen, T. Burghardt, A. Paiement, L. Tao,
 DS-KCF: A ~real-time tracker for RGB-D data, Journal of Real-Time Image Processing
 */

#include "Typedefs.hpp"

/**
 * ScaleChangeObserver is an abstract class.
 * This class is designed to be derived by any class which needs to observe
 * a ScaleAnalyser. An instance of ScaleChangeObserver should only be
 * registered to observe a single ScaleAnalyser.
 */
class ScaleChangeObserver
	{
	public:
		/**
		 * onScaleChange is called whenever a scale change has been detected.
		 * @param targetSize The new size of the target object's bounding box.
		 * @param windowSize The new padded size of the bounding box around the target.
		 * @param yf The new gaussian shaped labels for this scale.
		 * @param cosineWindow The new cosine window for this scale.
		 *
		 * @warning If an instance of this class is registered to observe multiple
		 *   ScaleAnalyser, then this method will likely cause a crash.
		 */
		virtual void onScaleChange(const Size & targetSize, const Size & windowSize, const cv::Mat2d & yf, const cv::Mat1d & cosineWindow) = 0;
	};

#endif
