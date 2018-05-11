/**
 * File: createCNNvoc.cpp
 */

// Read descriptors from file
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <array>
#include <unistd.h>

// standard
#include <stdio.h>
#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines CNNVocabulary and CNNDatabase

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>
#include "FClass.h"
#include "FCNN.h"

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

void loadFeatures(std::vector<std::vector<FCNN::TDescriptor > > &features);
void changeStructure(const cv::Mat &plain, std::vector<FCNN::TDescriptor> &out);
void createVoc(const std::vector<std::vector<FCNN::TDescriptor > > &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 65047;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
  std::cout << "Running createCNNvoc" << std::endl;

  std::vector<std::vector<FCNN::TDescriptor > > features;

  loadFeatures(features);

  createVoc(features);

  return 0;
}

// ----------------------------------------------------------------------------


void loadFeatures(std::vector<std::vector<FCNN::TDescriptor > > &descriptors)
{ 
    descriptors.clear();
    //descriptors.reserve(NIMAGES);
    std::cout<<"About to load descriptors of " << NIMAGES<< " images."<<std::endl;

    std::string s_cwd(getcwd(NULL,0));
    //std::cout << "CWD is: " << s_cwd << std::endl;

    std::string filename = s_cwd + "/../TrainingImages/descriptors.dat";
    std::cout << "From file: " << filename << std::endl;
    // Read from file
    std::ifstream f{filename, std::ios::binary};
    if (!f) { throw std::runtime_error{std::strerror(errno)}; }

    int nDescriptors = 0;
    int nFrames = 0;
    std::cout << "With CNN descriptor size: "<<FCNN::L << std::endl;
    // Size of CNN descriptor is set in FCNN.h
    bool stop = false;
    wait();
    std::cout << "Loading descriptors..."<< std::endl;
    while( !f.eof() )
    {
      std::vector<FCNN::TDescriptor> vDescriptorsFromOneFrame;
      while(!f.eof())
      {

        // Read (x,y)-coordinate of feature
        int x;
        int y;
        f.read(reinterpret_cast<char*>(&x), sizeof(std::int32_t));
        f.read(reinterpret_cast<char*>(&y), sizeof(std::int32_t));

        if ( x==-1 || y==-1)
        {
          break;
        }


        // Read descriptor of feature
        FCNN::TDescriptor descriptor;
        descriptor.reserve(FCNN::L);

        for (auto i = 0; i < FCNN::L; ++i) {
            float t;
            f.read(reinterpret_cast<char *>(&t), sizeof(t));
            if (!f) { throw std::runtime_error{std::strerror(errno)}; }
            descriptor.push_back(t);
        }
        nDescriptors +=1;


        // Save feature in vector of features
        vDescriptorsFromOneFrame.push_back(descriptor);
      }
      nFrames +=1;
      nDescriptors = 0;
      if (!vDescriptorsFromOneFrame.empty())
      {
        descriptors.push_back(vDescriptorsFromOneFrame);
      }

      // Print status
      if (nFrames % 4000 == 0)
      {
        std::cout <<std::endl<<"["<<nFrames<<"/"<<NIMAGES<<"]";
      }
      if (nFrames % 200 == 0)
      {
        std::cout <<"#"<< std::flush;
      }
    }

    std::cout << "Done"<< std::endl;
}

// ----------------------------------------------------------------------------

void createVoc(const std::vector<std::vector<FCNN::TDescriptor > > &features)
{
  // branching factor and depth levels 
  const int k = 10; // 10 used in ORB-SLAM
  const int L = 6; // 6 used in ORB-SLAM
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  CNNVocabulary voc(k, L, weight, score);

  std::cout<<std::endl << "Creating a small " << k << "^" << L << " vocabulary:" << std::endl;
  voc.create(features);
  std::cout << "Done" << std::endl;

  std::cout << "Vocabulary information: " << std::endl
  << voc << std::endl << std::endl;

  // save the vocabulary to disk
  std::string vocName = "cnn_voc";

  std::cout << std::endl << "Saving vocabulary as " + vocName + ".txt" << std::endl;
  try{
    voc.saveToTextFile(vocName + ".txt");
  } catch (const std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  } catch (...) {
    std::cout << "An unknown exception occurred" << std::endl;
  }
  
  std::cout << std::endl << "Saving vocabulary as " + vocName + ".voc" << std::endl;
  try{
    voc.save(vocName + ".voc");
  } catch (const std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  } catch (...) {
    std::cout << "An unknown exception occurred" << std::endl;
  }

  std::cout << std::endl << "Saving vocabulary as " + vocName + ".yml.gz" << std::endl;
  try{
    voc.save(vocName + ".yml.gz");
  } catch (const std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  } catch (...) {
    std::cout << "An unknown exception occurred" << std::endl;
  }
  
  std::cout << "Done" << std::endl;


  // lets do something with this vocabulary
  std::cout << std::endl << "Small test: Matching 4 first images against themselves (0 low, 1 high): " << std::endl;
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

}

// ----------------------------------------------------------------------------