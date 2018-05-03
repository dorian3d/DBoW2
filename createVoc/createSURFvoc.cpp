/**
 * File: createSURFvoc.cpp
 */
#include <stdio.h>
#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines CNNVocabulary and CNNDatabase

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>
//#include "FCNN.h"
#include "FClass.h"
#include "FSurf64.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>


#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"


#include "opencv2/core/ocl.hpp"
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace cv::xfeatures2d;
using namespace DBoW2;
using namespace DUtils;



// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
struct SURFDetector
{
    Ptr<Feature2D> surf;
    SURFDetector(double hessian = 800.0)
    {
        surf = SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(std::vector<std::vector<FSurf64::TDescriptor > > &features);
void changeStructure(const cv::Mat &plain, std::vector<FSurf64::TDescriptor> &out);
void createVoc(const std::vector<std::vector<FSurf64::TDescriptor > > &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 65047;

// ----------------------------------------------------------------------------

int main()
{
  std::cout << "Running createCNNvoc" << std::endl;

  //std::vector<std::vector<FCNN::TDescriptor > > features;
  std::vector<std::vector<FSurf64::TDescriptor > > features;

  loadFeatures(features);

  createVoc(features);

  return 0;
}

// ----------------------------------------------------------------------------


void loadFeatures(std::vector<std::vector<FSurf64::TDescriptor > > &vDescriptors)
{
  vDescriptors.clear();
  vDescriptors.reserve(NIMAGES);

  SURFDetector surf;
  std::cout << "Extracting Surf descriptors..." << std::endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    // Print status
    if (i % 1000 == 0)
    {
      std::cout <<std::endl<<"["<<i<<"/"<<NIMAGES<<"]";
    }
    if (i % 20 == 0)
    {
      std::cout <<"#"<< std::flush;
    }

    std::stringstream ss;
    ss << "../TrainingImages/FRONTAL/FRONTAL" << i << ".png";


    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat mDescriptors;

    surf(image, mask, keypoints, mDescriptors);
    vDescriptors.push_back(std::vector<FSurf64::TDescriptor >());
    changeStructure(mDescriptors, vDescriptors.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, std::vector<FSurf64::TDescriptor> &out)
{
    std::vector<std::vector<float>> features;
    features.reserve(plain.rows);
    for (int j=0;j<plain.rows;j++)
    {
        // Pointer to the i-th row
        const float* p = plain.ptr<float>(j);

        // Copy data to a vector.  Note that (p + plain.cols) points to the
        // end of the row.
        std::vector<float> vec(p, p + plain.cols);
        features.push_back(vec);
    }
    out = features;
    
}
// ----------------------------------------------------------------------------

void createVoc(const std::vector<std::vector<FSurf64::TDescriptor > > &features)
{
  // branching factor and depth levels 
  const int k = 10; // 10 used in ORB-SLAM
  const int L = 6; // 6 used in ORB-SLAM
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  SURF64Vocabulary voc(k, L, weight, score);

  std::cout<<std::endl << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
  voc.create(features);
  std::cout << "... done!" << std::endl;

  std::cout << "Vocabulary information: " << std::endl
  << voc << std::endl << std::endl;

  // lets do something with this vocabulary
  std::cout << "Matching 4 first images against themselves (0 low, 1 high): " << std::endl;
  BowVector v1, v2;
  for(int i = 0; i < 4; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < 4; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
    }
  }

  // save the vocabulary to disk
  
  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  voc.save("surf_voc.yml.gz");

  voc.saveToTextFile("surf_voc.txt");
  
  std::cout << "Done" << std::endl;

}

// ----------------------------------------------------------------------------