/**
 * File: createCNNVocabulary.cpp
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
void testVocCreation(const std::vector<std::vector<FSurf64::TDescriptor > > &features);
void testDatabase(const std::vector<std::vector<FSurf64::TDescriptor > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;//65047;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  std::cout << std::endl << "Press enter to continue" << std::endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
  std::cout << "Running example" << std::endl;
  //std::vector<std::vector<FCNN::TDescriptor > > features;
  std::vector<std::vector<FSurf64::TDescriptor > > features;
  loadFeatures(features);

  testVocCreation(features);

  //wait();

  //testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------


void loadFeatures(std::vector<std::vector<FSurf64::TDescriptor > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  //cv::Ptr<cv::SURF> surf = cv::SURF::create();
  //cv::Ptr<cv::Feature2D> mSurfDetector = cv::xfeatures2d::SURF::create();
  
  //Ptr<SURF> detector = SURF::create();
  SURFDetector surf;
  std::cout << "Extracting Surf features..." << std::endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    if (i % 1000 == 0)
    {
      std::cout <<std::endl<<"["<<i<<"/"<<NIMAGES<<"]";
    }
    if (i % 20 == 0)
    {
      std::cout <<"#"<< std::flush;
    }

    std::stringstream ss;
    //ss << "../TrainingImages/FRONTAL/FRONTAL" << i << ".png";
    ss << "../demo/images_old/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    //detector->detectAndCompute( image, mask, keypoints, descriptors);
    surf(image, mask, keypoints, descriptors);
    features.push_back(std::vector<FSurf64::TDescriptor >());
    changeStructure(descriptors, features.back());

    //std::cout<< "Features detected: " << features[i].size()<<"x"<< 
    //features[i][0].size() << std::endl;
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

void testVocCreation(const std::vector<std::vector<FSurf64::TDescriptor > > &features)
{
  // branching factor and depth levels 
  const int k = 9; // 10 used in ORB-SLAM
  const int L = 3; // 6 used in ORB-SLAM
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  SURF64Vocabulary voc(k, L, weight, score);

  std::cout<<std::endl << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
  voc.create(features);
  std::cout << "... done!" << std::endl;

  std::cout << "Vocabulary information: " << std::endl
  << voc << std::endl << std::endl;

  // lets do something with this vocabulary
  std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;
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
  voc.save("small_voc.yml.gz");

  voc.saveToTextFile("small_voc.txt");
  
  std::cout << "Done" << std::endl;

}

// ----------------------------------------------------------------------------

void testDatabase(const std::vector<std::vector<FSurf64::TDescriptor > > &features)
{
  std::cout << "Creating a small database..." << std::endl;

  // load the vocabulary from disk
  //SURF64Vocabulary voc("small_voc.yml.gz");

  SURF64Vocabulary voc("small_voc.txt");

  
  SURF64Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  std::cout << "... done!" << std::endl;

  std::cout << "Database information: " << std::endl << db << std::endl;

  // and query the database
  std::cout << "Querying the database: " << std::endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    std::cout << "Searching for Image " << i << ". " << ret << std::endl;
  }

  std::cout << std::endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  std::cout << "Saving database..." << std::endl;
  db.save("small_db.yml.gz");
  std::cout << "... done!" << std::endl;
  
  // once saved, we can load it again  
  std::cout << "Retrieving database once again..." << std::endl;
  OrbDatabase db2("small_db.yml.gz");
  std::cout << "... done! This is: " << std::endl << db2 << std::endl;
}

// ----------------------------------------------------------------------------


