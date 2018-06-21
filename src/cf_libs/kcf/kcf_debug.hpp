

/*
// Original file: https://github.com/klahaag/cf_tracking/blob/master/src/cf_libs/kcf/kcf_debug.hpp
// + Authors: Jake Hall, Massimo Camplan, Sion Hannuna
// * We modified the original code of  Klaus Haag, such that different classes are used for the different KCF components
//   in a more modular way, to support DS-KCF code
*/

#ifndef KCF_DEBUG_HPP_
#define KCF_DEBUG_HPP_

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>

#include "tracker_debug.hpp"

template<typename T>
class KcfDebug : public TrackerDebug
{
public:
  KcfDebug() :
    _maxResponse(0),
    _psrClamped(0),
   _RESPONSE_TITLE( "Response" )
  {
    _SUB_WINDOW_TITLE[ 0 ] = "SUB WINDOW RGB";
    _SUB_WINDOW_TITLE[ 1 ] = "SUB WINDOW DEPTH";
  }

  virtual ~KcfDebug()
  {
    if (_outputFile.is_open())
    {
      _outputFile.close();
    }
  }

  virtual void init(std::string outputFilePath)
  {
    namedWindow(_SUB_WINDOW_TITLE[ 0 ], cv::WINDOW_NORMAL);
    namedWindow(_SUB_WINDOW_TITLE[ 1 ], cv::WINDOW_NORMAL);
    namedWindow(_RESPONSE_TITLE, cv::WINDOW_NORMAL);
    _outputFile.open(outputFilePath.c_str());
  }

  virtual void printOnImage(cv::Mat& image)
  {
    _ss.str("");
    _ss.clear();
    _ss << "Max Response: " << _maxResponse;
    putText(image, _ss.str(), cv::Point(20, 40), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));

    _ss.str("");
    _ss.clear();
    _ss << "PSR Clamped: " << _psrClamped;
    putText(image, _ss.str(), cv::Point(20, 80), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(255, 0, 0));
  }

  virtual void printConsoleOutput()
  {
  }

  virtual void printToFile()
  {
    _outputFile << _maxResponse << "," << _psrClamped << std::endl;
  }

  void showPatch(const cv::Mat& patchResized)
  {
    imshow(_SUB_WINDOW_TITLE[ _windowIndex ], patchResized);
    _windowIndex = ( _windowIndex + 1 ) % 2;
  }

  void setPsr(T psrClamped)
  {
    _psrClamped = psrClamped;
    std::cout << "PSR: " << psrClamped << std::endl;
  }

  void showResponse(const cv::Mat& response, T maxResponse)
  {
    cv::Mat responseOutput = response.clone();
    _maxResponse = maxResponse;
    imshow(_RESPONSE_TITLE, responseOutput);
  }

private:
  std::array<std::string, 2 > _SUB_WINDOW_TITLE;
  const std::string _RESPONSE_TITLE;
  T _maxResponse;
  T _psrClamped;
  std::stringstream _ss;
  std::ofstream _outputFile;
  int _windowIndex;
};

#endif
