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
#include <algorithm>
#include <iterator>

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

bool writeDescToBinFile(string filename, vector<float> descriptors)
{
    ofstream file(filename.c_str(), ofstream::out | ofstream::binary);
    // write the number of descriptors
    const size_t numDesc = descriptors.size()/128;
    file.write((const char*) &numDesc, sizeof(std::size_t));
    for (int i = 0; i < descriptors.size(); ++i)
    {
        file.write( (char*) &descriptors[i], sizeof(float));
    }
    bool isOk = file.good();
    file.close();
    return isOk;
}

vector<float> readDescFromBinFile(const char* path)
{
    fstream fs;
    size_t ndesc;

    // Open file and get the number of descriptors
    fs.open(path, ios::in | ios::binary);

    // get the number of descriptors
    fs.read((char*) &ndesc, sizeof(size_t));

    vector<float> res;

    // Fill the matrix in the air
    for (int i = 0; i < ndesc*128; i++)
    {
        float cur;
        fs.read((char*) &cur, sizeof(float));
        res.push_back(cur);
    }

    // Close file and return
    fs.close();
    return res;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

bool loadFeatures(vector<vector<vector<float> > > &features,
        string sDatasetImagesDirectory, string sOutDirectory,
        vector<string> imagesNames, bool root, bool justDesc);
void storeImages(const char* imagesDirectory, vector<string>& imagesNames, bool desc);
void changeStructure(const vector<float> &plain, vector<vector<float> > &out,
        int L);
void testVocCreation(const vector<vector<vector<float> > > &features,
        string& sOutDirectory, string& vocName, int k, int L, bool isOK, bool matchingTest);
void testDatabase(const vector<vector<vector<float> > > &datasetFeatures,
        const vector<vector<vector<float> > > &queryFeatures,
        vector<string>& datasetImagesNames, vector<string>& queryImagesNames,
        string& sOutDirectory, string& vocName, string& dbName,
        const int numImagesQuery, int diLvl);
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
"{h | help    | false  | print this message                               }"
"{d | dataset | images/  | path to the directory containing dataset images}"
"{q | query   | images/  | path to the directory containing query images  }"
"{o | output  | ./       | path to the output directory                   }"
"{v | vocName | small_voc.yml.gz  | name of the vocabulary file           }"
"{b | dbName  | db.yml.gz  | name of the database file                    }"
"{k |         | 9 | max number of sons of each node                       }"
"{L |         | 3 | max depth of the vocabulary tree                      }"
"{r | rootSift| false | use rootSift instead of SIFT                      }"
"{n | nBest   | 4 | number of best matches to keep                        }"
"{t | testVoc | false | print the score of all pairs after voc creation   }"
"{w | wait    | false | wait between voc creation and database creation   }"
"{i | direct  | -1 | direct index level (if -1, do not use direct index)  }"
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
    string dbName = parser.get<string>("dbName");

    int k = parser.get<int>("k");
    int L = parser.get<int>("L");
    int diLvl = parser.get<int>("i");
    const int numImagesQuery = parser.get<int>("n");

    bool root = parser.get<bool>("r");
    bool matchingTest = parser.get<bool>("t");
    bool waitAfterVocCreation = parser.get<bool>("w");
    bool justDesc = false;

    vector<string> datasetImagesNames;
    vector<string> queryImagesNames;

    vector<double> timers;

    struct timeval tglobalStart, t1, t2, tglobalEnd;

    cout << endl;
    cout << "Starting the global timer..." << endl;
    cout << endl;

    gettimeofday(&tglobalStart, NULL);
    gettimeofday(&t1, NULL);

    storeImages(sDatasetImagesDirectory.c_str(), datasetImagesNames, false);
    if (datasetImagesNames.size() == 0)
    {
        cout << "There is no image in the dataset directory " << sDatasetImagesDirectory
                << "... The program is looking for .desc file in the output directory "
                << sOutDirectory << endl;

        justDesc = true;
        storeImages(sOutDirectory.c_str(), datasetImagesNames, true);
        if (datasetImagesNames.size() == 0)
        {
            cout << "There is no image in the dataset directory " << sDatasetImagesDirectory
                << " and no .desc file in the output directory " << sOutDirectory
                << "... The program cannot work." << endl;
            return EXIT_FAILURE;
        }

    }

    gettimeofday(&t2, NULL);
    double texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);
    timers.push_back(texec);

    gettimeofday(&t1, NULL);

    storeImages(sQueryImagesDirectory.c_str()  , queryImagesNames, false);

    gettimeofday(&t2, NULL);
    texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);
    timers.push_back(texec);

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

    gettimeofday(&t1, NULL);

    bool isOK = loadFeatures(datasetFeatures, sDatasetImagesDirectory, sOutDirectory, datasetImagesNames, root, justDesc);

    gettimeofday(&t2, NULL);
    texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);
    timers.push_back(texec);
    gettimeofday(&t1, NULL);

    loadFeatures(queryFeatures  , sQueryImagesDirectory  , sOutDirectory, queryImagesNames, root, false);

    gettimeofday(&t2, NULL);
    texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);
    timers.push_back(texec);
    gettimeofday(&t1, NULL);

    testVocCreation(datasetFeatures, sOutDirectory, vocName, k, L, isOK, matchingTest);

    gettimeofday(&t2, NULL);
    texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);
    timers.push_back(texec);

    if (waitAfterVocCreation) wait();

    gettimeofday(&t1, NULL);

    testDatabase(datasetFeatures, queryFeatures, datasetImagesNames,
            queryImagesNames, sOutDirectory, vocName, dbName, numImagesQuery, diLvl);

    gettimeofday(&t2, NULL);
    texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);
    timers.push_back(texec);

    gettimeofday(&tglobalEnd, NULL);
    texec = (double) (1000.0*(tglobalEnd.tv_sec - tglobalStart.tv_sec) + (tglobalEnd.tv_usec - tglobalStart.tv_usec)/1000.0);

    cout << endl;
    cout << "Dataset images storage took " << timers[0] << " ms" << endl;
    cout << "Query images storage took " << timers[1] << " ms" << endl;
    cout << "Dataset features loading took " << timers[2] << " ms" << endl;
    cout << "Query features loading took " << timers[3] << " ms" << endl;
    cout << "Vocabulary testing took " << timers[4] << " ms" << endl;
    cout << "Database testing took " << timers[5] << " ms" << endl;
    cout << endl;
    cout << "Global algorithm took " << texec << " ms" << endl;

    return 0;
}

// ----------------------------------------------------------------------------

void storeImages(const char* imagesDirectory, vector<string>& imagesNames, bool desc)
{
    // image extensions
    vector<string> extensions;
    if (desc)
    {
        extensions.push_back("desc");
    }
    else
    {
        extensions.push_back("png");
        extensions.push_back("jpg");
    }

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

bool loadFeatures(vector<vector<vector<float> > > &features,
        string sDatasetDirectory, string sOutDirectory, vector<string> imagesNames,
        bool root, bool justDesc)
{
    bool goodDescType = true;
    features.clear();
    features.reserve(imagesNames.size());

    cv::SIFT sift(15000, 3, 0.04, 10, 1.6);

    bool loadFile = false;

    for(int i = 0; i < imagesNames.size(); ++i)
    {
        stringstream ss;
        ss << sDatasetDirectory << "/" << imagesNames[i];

        string descFileName = imagesNames[i].substr(0, imagesNames[i].find_last_of(".")) + ".desc";
        vector<float> descriptors;

        if (!fileAlreadyExists(descFileName, sOutDirectory))
        {
            cout << "File " << sOutDirectory + "/" + descFileName << " does not exist" << endl;
            cv::Mat image = cv::imread(ss.str(), 0);
            cv::Mat mask;
            vector<cv::KeyPoint> keypoints;
            cv::Mat matDescriptors;
            sift(image, mask, keypoints, matDescriptors);

            descriptors.assign((float*)matDescriptors.datastart,
                    (float*)matDescriptors.dataend);

            int descriptorSize = static_cast<int>(descriptors.size()/128);

            if (root)
            {
                for (unsigned int j = 0; j < descriptorSize; ++j)
                {
                    float sum = 0;
                    for (unsigned int k = 0; k < 128; ++k)
                    {
                        sum += descriptors[128*j+k];
                    }
                    for (unsigned int k = 0; k < 128; ++k)
                    {
                        descriptors[128*j+k] = sqrt(descriptors[128*j+k]/sum);
                    }
                }
            }
            writeDescToBinFile(sOutDirectory+"/"+descFileName, descriptors);
        }

        vector<float> firstDescriptor;
        string path = sOutDirectory + "/" + descFileName;
        firstDescriptor = readDescFromBinFile(path.c_str());

        if (firstDescriptor.size() >= 128)
        {
            float firstDescriptorNorm = 0;

            for (unsigned int k = 0; k < 128; k++)
            {
                firstDescriptorNorm += firstDescriptor[k]*firstDescriptor[k];
            }

            if ((firstDescriptorNorm < 1.01 ) == root)
            {
                loadFile = true;
            }
            else
            {
                goodDescType = false;
            }
        }

        else // descriptor is empty so it can be loaded with sift or rootSift the same way
        {
            loadFile = true;
        }

        if (loadFile) // descFileName already exists, just load it
        {
            string path = sOutDirectory + "/" + descFileName;
            descriptors = readDescFromBinFile(path.c_str());
            cout << "Descriptor size is " << descriptors.size()/128 << endl;
            cout << endl;
        }
        else // a rootSift descriptor file has been loaded with -r false
             // or a sift descriptor file has been loaded with -r true
             // descriptor has to be recomputed from zero
        {
            if (root)
            {
                cout << "File " << sOutDirectory + "/" + descFileName
                     << " represents a sift descriptor but the program has been called with rootSift descriptors option : it will be recomputed"
                     << endl;
            }
            else
            {
                cout << "File " << sOutDirectory + "/" + descFileName
                     << " represents a rootSift descriptor but the program has been called with sift descriptors option : it will be recomputed"
                     << endl;
            }
            cv::Mat image = cv::imread(ss.str(), 0);
            cv::Mat mask;
            vector<cv::KeyPoint> keypoints;
            cv::Mat matDescriptors;
            sift(image, mask, keypoints, matDescriptors);
            cout << keypoints.size() << " keypoints found on " << sDatasetDirectory
                << "/" << imagesNames[i] << endl;
            cout << endl;

            descriptors.assign((float*)matDescriptors.datastart,
                               (float*)matDescriptors.dataend);

            int descriptorSize = static_cast<int>(descriptors.size()/128);

            if (root)
            {
                for (unsigned int j = 0; j < descriptorSize; ++j)
                {
                    float sum = 0;
                    for (unsigned int k = 0; k < 128; k++)
                    {
                        sum += descriptors[128*j+k];
                    }
                    for (unsigned int k = 0; k < 128; k++)
                    {
                        descriptors[128*j+k] = sqrt(descriptors[128*j+k]/sum);
                    }
                }
            }
            writeDescToBinFile(descFileName, descriptors);

        }

        features.push_back(vector<vector<float> >());
        changeStructure(descriptors, features.back(), sift.descriptorSize());
    }
    return goodDescType;
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
        string& sOutDirectory, string& vocName, int k, int L, bool isOK, bool matchingTest)
{
    if (fileAlreadyExists(vocName, sOutDirectory) && isOK)
    {
        return;
    }

    // branching factor and depth levels
    const WeightingType weight = TF_IDF;
    const ScoringType score = L2_NORM;

    SiftVocabulary voc(k, L, weight, score);


    cout << "Creating a " << k << "^" << L << " vocabulary..." << endl;

    struct timeval tbegin, tend;
    gettimeofday(&tbegin, NULL);

    voc.create(features);

    gettimeofday(&tend,NULL);
    double texec = (double) (1000.0*(tend.tv_sec - tbegin.tv_sec) + (tend.tv_usec - tbegin.tv_usec)/1000.0);

    cout << "... done!" << endl;
    cout << " in " << texec << " ms" << endl;
    cout << endl;

    cout << "Vocabulary information: " << endl
        << voc << endl;

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save(sOutDirectory + "/" + vocName);
    cout << endl;

    if (matchingTest)
    {
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
    }
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<vector<float> > > &datasetFeatures,
        const vector<vector<vector<float> > > &queryFeatures,
        vector<string>& datasetImagesNames, vector<string>& queryImagesNames,
        string& sOutDirectory, string& vocName, string& dbName,
        const int numImagesQuery, int diLvl)
{
    cout << "Creating a database..." << endl;
    cout << endl;

    // load the vocabulary from disk
    cout << "Loading the vocabulary..." << endl;
    cout << endl;
    SiftVocabulary voc(sOutDirectory + "/" + vocName);
    cout << "Vocabulary loaded!" << endl;

    cout << endl;
    cout << "Database creation..." << endl;

    struct timeval t1, t2;
    gettimeofday(&t1, NULL);

    bool useDirectIndex = diLvl >= 0;
    bool dbAlreadyExists = fileAlreadyExists(dbName, sOutDirectory);

    SiftDatabase db;


    if (!dbAlreadyExists)
    {
        db = SiftDatabase(voc, useDirectIndex, diLvl); // false = do not use direct index
        // (so ignore the last param)
        // The direct index is useful if we want to retrieve the features that
        // belong to some vocabulary node.
        // db creates a copy of the vocabulary, we may get rid of "voc" now

        gettimeofday(&t2, NULL);
        double texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);

        cout << "Database created in " << texec << " ms! Adding the features to the database..." << endl;
        gettimeofday(&t1, NULL);

        // add images to the database
        for(int i = 0; i < NIMAGES_DATASET; i++)
        {
            db.add(datasetFeatures[i]);
        }

        gettimeofday(&t2, NULL);
        texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);
        cout << "... done in " << texec << " ms!" << endl;
    }
    else
    {
        db.load(sOutDirectory + "/" + dbName);
    }

    cout << endl;
    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << endl;
    cout << "Querying the database: " << endl;
    cout << endl;

    int nbBestMatchesToKeep = numImagesQuery;

    QueryResults ret;
    for(int i = 0; i < NIMAGES_QUERY; i++)
    {
        gettimeofday(&t1, NULL);
        db.query(queryFeatures[i], ret, nbBestMatchesToKeep);
        gettimeofday(&t2,NULL);
        double texec = (double) (1000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000.0);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        for (int j = 0; j < fmin(nbBestMatchesToKeep, ret.size()); j++)
        {
            cout << "Searching for Image " << i << ": " << queryImagesNames[i]
                << "... Found: " << datasetImagesNames[ret[j].Id]
                << " with score " << ret[j].Score << endl;
        }
        cout << " in " << texec << " ms" <<  endl;
        cout << endl;
    }

    cout << endl;

    if (!dbAlreadyExists)
    {
        // we can save the database. The created file includes the vocabulary
        // and the entries added
        cout << "Saving database..." << endl;
        db.save(sOutDirectory + "/db.yml.gz");
        cout << "... done!" << endl;
    }
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
                closedir(repertoire);
                return true;
            }
        }
    }
    closedir(repertoire);
    return false;
}

