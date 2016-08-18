/**
 * File: FClass.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: generic FClass to instantiate templated classes
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_DESCMANIP__
#define __D_T_DESCMANIP__

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include "exports.h"

namespace DBoW2 {

/// Class to manipulate descriptors (calculating means, differences and IO routines)
class DBOW_API DescManip
{
public:
  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
   static void meanValue(const std::vector<cv::Mat> &descriptors,
    cv::Mat &mean)  ;
  
  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
  static double distance(const cv::Mat &a, const cv::Mat &b);
  
  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const cv::Mat &a);
  
  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(cv::Mat &a, const std::string &s);

  /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
  static void toMat32F(const std::vector<cv::Mat> &descriptors,
    cv::Mat &mat);

private:
  /**Returns the number of bytes of the descriptor
   * used for binary descriptors only*/
  static size_t getnBytes(const cv::Mat & d){return d.cols* d.elemSize();}
};

} // namespace DBoW2

#endif
