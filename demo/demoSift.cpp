/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <dirent.h>
#include <locale>

// DBoW2
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database
// defines SiftVocabulary and SiftDatabase

#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX
#include <DVision/DVision.h>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#if CV24
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/core/core.hpp>
#endif

// Execution Time
#include <sys/time.h>

using namespace DBoW2;
using namespace DUtils;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void loadFeatures(vector<vector<vector<float> > > &features,
  string sDatasetImagesDirectory, string sOutDirectory,
  vector<string> imagesNames);
void storeImages(const char* imagesDirectory, vector<string>& imagesNames);
void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
  int L);
void testVocCreation(const vector<vector<vector<float> > > &features,
  string& sOutDirectory, string& vocName, int k, int L);
void testDatabase(const vector<vector<vector<float> > > &datasetFeatures,
  const vector<vector<vector<float> > > &queryFeatures,
  vector<string>& datasetImagesNames, vector<string>& queryImagesNames,
  string& sOutDirectory, string& vocName);
bool fileAlreadyExists(string& fileName, string& sDirectory);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// number of training images
int NIMAGES_DATASET;
int NIMAGES_QUERY;


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

const char* keys =
  "{h | help	| false  | print this message                               }"
  "{d | dataset | images/  | path to the directory containing dataset images}"
  "{q | query	| images/  | path to the directory containing query images  }"
  "{o | output  | ./       | path to the output directory                   }"
  "{v | vocName | small_voc.yml.gz  | name of the vocabulary file           }"
  "{k |         | 9 | max number of sons of each node                       }"
  "{L |         | 3 | max depth of the vocabulary tree                      }"
;

// ----------------------------------------------------------------------------

int main(int argc, const char **argv)
{
  cv::CommandLineParser parser(argc, argv, keys);
  if( parser.get<bool>( "h" ) || parser.get<bool>( "help" ) )
  {
    parser.printParams();
    return EXIT_SUCCESS;
  }

  parser.printParams();
  string sDatasetImagesDirectory = parser.get<string>("d");
  string sQueryImagesDirectory   = parser.get<string>("q");
  string sOutDirectory = parser.get<string>("o");
  string vocName = parser.get<string>("vocName");

  int k = parser.get<int>("k");
  int L = parser.get<int>("L");

  vector<string> datasetImagesNames;
  vector<string> queryImagesNames;

  storeImages(sDatasetImagesDirectory.c_str(), datasetImagesNames);
  storeImages(sQueryImagesDirectory.c_str()  , queryImagesNames);

  NIMAGES_DATASET = datasetImagesNames.size();
  NIMAGES_QUERY   = queryImagesNames.size();

  cout << "Dataset images: " << endl;
  for (unsigned int i = 0; i < datasetImagesNames.size(); i++)
  {
      cout << "image " << i << " = " << datasetImagesNames[i] << endl;
  }

  cout << endl;

  cout << "Query images: " << endl;
  for (unsigned int i = 0; i < queryImagesNames.size(); i++)
  {
      cout << "image " << i << " = " << queryImagesNames[i] << endl;
  }

  cout << endl;
  cout << "Extracting SIFT features..." << endl;
  vector<vector<vector<float> > > datasetFeatures;
  vector<vector<vector<float> > > queryFeatures;
  loadFeatures(datasetFeatures, sDatasetImagesDirectory, sOutDirectory, datasetImagesNames);
  loadFeatures(queryFeatures  , sQueryImagesDirectory  , sOutDirectory, queryImagesNames  );

  testVocCreation(datasetFeatures, sOutDirectory, vocName, k, L);

  wait();

  testDatabase(datasetFeatures, queryFeatures, datasetImagesNames,
    queryImagesNames, sOutDirectory, vocName);

  return 0;
}

// ----------------------------------------------------------------------------

void storeImages(const char* imagesDirectory, vector<string>& imagesNames)
{
  // image extensions
  vector<string> extensions;
  extensions.push_back("png");
  extensions.push_back("jpg");

  DIR * repertoire = opendir(imagesDirectory);

  if ( repertoire == NULL)
  {
    cout << "The images directory: " << imagesDirectory
      << " cannot be found" << endl;
  }
  else
  {
    struct dirent * ent;
    while ( (ent = readdir(repertoire)) != NULL)
    {
      string file_name = ent->d_name;
      string extension = file_name.substr(file_name.find_last_of(".") +1);
      locale loc;
      for (std::string::size_type j = 0; j < extension.size(); j++)
      {
        extension[j] = std::tolower(extension[j], loc);
      }
      for (unsigned int i = 0; i < extensions.size(); i++)
      {
        if (extension == extensions[i])
        {
          imagesNames.push_back(file_name);
        }
      }
    }
  closedir(repertoire);
  }
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<vector<float> > > &features,
  string sDatasetDirectory, string sOutDirectory, vector<string> imagesNames)
{
  features.clear();
  features.reserve(imagesNames.size());

  cv::SIFT sift(15000, 3, 0.04, 10, 1.6);

  for(int i = 0; i < imagesNames.size(); ++i)
  {
    stringstream ss;
    ss << sDatasetDirectory << "/" << imagesNames[i];

    string descFileName = imagesNames[i].substr(0, imagesNames[i].find_last_of(".")) + ".desc";
    vector<float> descriptors;

    if (!fileAlreadyExists(descFileName, sOutDirectory))
    {
      cout << "File " << sOutDirectory + "/" + descFileName << " does not exist" << endl;
      BinaryFile binFile(sOutDirectory + "/" + descFileName, WRITE);
      cv::Mat image = cv::imread(ss.str(), 0);
      cv::Mat mask;
      vector<cv::KeyPoint> keypoints;
      cv::Mat matDescriptors;
      sift(image, mask, keypoints, matDescriptors);
      cout << keypoints.size() << " keypoints found on " << sDatasetDirectory
        << "/" << imagesNames[i] << endl;

      descriptors.assign((float*)matDescriptors.datastart,
                         (float*)matDescriptors.dataend);

      std::cout << "descriptor size is " << descriptors.size()/128 << std::endl;
      binFile << static_cast<int>(descriptors.size()/128);
      for (unsigned int j = 0; j < descriptors.size(); j++)
      {
        binFile << descriptors[j];
      }
      binFile.Close();
    }
    else // descFileName already exists, just load it
    {
      BinaryFile binFile;
      binFile.OpenForReading(sOutDirectory + "/" + descFileName);
      int descriptorSize;
      binFile >> descriptorSize;
      std::cout << "descriptor size is " << descriptorSize << std::endl;
      descriptors.resize(128*descriptorSize);
      for (unsigned int j = 0; j < 128*descriptorSize; j++)
      {
        binFile >> descriptors[j];
      }
    }

    features.push_back(vector<vector<float> >());
    changeStructure(descriptors, features.back(), sift.descriptorSize());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
  int L)
{
  out.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    out[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, out[j].begin());
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<vector<float> > > &features,
  string& sOutDirectory, string& vocName, int k, int L)
{
  if (fileAlreadyExists(vocName, sOutDirectory))
  {
    return;
  }

  // branching factor and depth levels
  const WeightingType weight = TF_IDF;
  const ScoringType score = L2_NORM;

  SiftVocabulary voc(k, L, weight, score);

  cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // let's do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES_DATASET; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES_DATASET; j++)
    {
      voc.transform(features[j], v2);

      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save(sOutDirectory + "/" + vocName);
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<float> > > &datasetFeatures,
  const vector<vector<vector<float> > > &queryFeatures,
  vector<string>& datasetImagesNames, vector<string>& queryImagesNames,
  string& sOutDirectory, string& vocName)
{
  cout << "Creating a database..." << endl;

  // load the vocabulary from disk
  SiftVocabulary voc(sOutDirectory + "/" + vocName);

  SiftDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES_DATASET; i++)
  {
    db.add(datasetFeatures[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  int nbBestMatchesToKeep = 4;

  QueryResults ret;
  for(int i = 0; i < NIMAGES_QUERY; i++)
  {
    struct timeval tbegin, tend;
    gettimeofday(&tbegin, NULL);
    db.query(queryFeatures[i], ret, nbBestMatchesToKeep);
    gettimeofday(&tend,NULL);
    double texec = (double) (1000.0*(tend.tv_sec - tbegin.tv_sec) + (tend.tv_usec - tbegin.tv_usec)/1000.0);

    // ret[0] is always the same image in this case, because we added it to the
    // database. ret[1] is the second best match.

    for (int j = 0; j < nbBestMatchesToKeep; j++)
    {
      cout << "Searching for Image " << i << ": " << queryImagesNames[i]
        << "... Found: " << datasetImagesNames[ret[j].Id]
        << " with score " << ret[j].Score << endl;
    }
    cout << "in " << texec << " ms" <<  endl;
    cout << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save(sOutDirectory + "/db.yml.gz");
  cout << "... done!" << endl;

  // once saved, we can load it again
  cout << "Retrieving database once again..." << endl;
  SiftDatabase db2(sOutDirectory + "/db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------

bool fileAlreadyExists(string& fileName, string& sDirectory)
{
  DIR * repertoire = opendir(sDirectory.c_str());

  if (repertoire == NULL)
  {
    cout << "The output directory: " << sDirectory
      << " cannot be found" << endl;
  }
  else
  {
    struct dirent * ent;
    while ( (ent = readdir(repertoire)) != NULL)
    {
      if (strncmp(ent->d_name, fileName.c_str(), fileName.size()) == 0)
      {
        cout << "The file " << fileName
          << " already exists, there is no need to recreate it" << endl;
        return true;
      }
    }
  closedir(repertoire);
  }
  return false;
}

