/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif


using namespace DBoW2;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}





void loadFeatures(vector<vector<cv::Mat> > &features,string path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    features.clear();
    features.reserve(NIMAGES);

    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    cout << "Extracting   features..." << endl;
    for(int i = 0; i < NIMAGES; ++i)
    {
        stringstream ss;
        ss << path_to_images<<"image" << i << ".png";
        cout<<"reading image: "<<ss.str()<<endl;
        cv::Mat image = cv::imread(ss.str(), 0);
        if(image.empty())throw std::runtime_error("Could not open image"+ss.str());
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        cout<<"done detecting features"<<endl;

        //turn descriptor into a set of matries, one per row and store in the vector
        features.push_back(vector<cv::Mat >(descriptors.rows));
        for(int i = 0; i < descriptors.rows; i++) features.back()[i]=descriptors.rowRange(i,i+1);
    }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  DBoW2::Vocabulary voc(k, L, weight, score);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

//// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
   Vocabulary voc("small_voc.yml.gz");
  
  Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  Database db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
    CmdLineParser cml(argc,argv);
    if (cml["-h"] || argc==1){
        cerr<<"Usage: [-i path_to_image_dir] [-d descriptor_name] \n\t descriptors:brisk,surf,orb(default),akaze(only if using opencv 3)"<<endl;
        return -1;
    }
  vector<vector<cv::Mat > > features;
  string path_to_images=cml("-i","");
  string descriptor=cml("-d","orb");

  loadFeatures(features,path_to_images,descriptor);

  testVocCreation(features);

  wait();

  testDatabase(features);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

  return 0;
}
