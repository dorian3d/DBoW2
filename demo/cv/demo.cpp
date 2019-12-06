/**************************************************************************
 * Copyright (c) 2019 Chimney Xu. All Rights Reserve.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **************************************************************************/
/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

// TDBoW and CV2Eigen transformer
#include <TDBoW/CVBridge.h> // this must be included before OpenCV

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <chrono>

using std::cout; using std::endl;
typedef TDBoW::Orb256Vocabulary Vocabulary;
typedef TDBoW::Orb256Database Database;
typedef Vocabulary::ConstDataSet ConstDataSet;
typedef std::vector<Vocabulary::DescriptorArray> DescriptorsSet;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

DescriptorsSet loadFeatures();
void testVocabCreation(const ConstDataSet& _DataSet, const DescriptorsSet& _Features);
void testDatabase(const DescriptorsSet& _Features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const size_t IMAGES_NUM = 4;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait() {
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

// ----------------------------------------------------------------------------

int main() {
    // Load files, calculate ORB descriptors and do transform
    // DescriptorArray and Descriptor are recommended types when query
    DescriptorsSet features = loadFeatures();

    // In this simple case, we had not prepare too many data, so we
    // use the same data for both create and query.
    // `make_shared`(inner method) will drop the original data, so make copy
    auto copy = features;
    // DataSet type is only used in vocabulary create.
    auto dataset = Vocabulary::util::make_const(copy);
    // Vocabulary testing
    testVocabCreation(dataset, features);

    wait();

    // Vocabulary testing
    testDatabase(features);

    return 0;
}

// ----------------------------------------------------------------------------

std::vector<Vocabulary::DescriptorArray> loadFeatures() {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cout << "Extracting ORB features..." << endl;
    DescriptorsSet features(IMAGES_NUM);
    std::stringstream ss;
    for(size_t i = 0; i < IMAGES_NUM; ++i) {
        ss << PKG_DIR << "/demo/cv/images/image" << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        std::vector<cv::KeyPoint> kp;
        cv::Mat descriptors;

        orb -> detectAndCompute(image, mask, kp, descriptors);

        features[i].resize(descriptors.rows, descriptors.cols);
        cv::cv2eigen(descriptors, features[i]);
        ss.clear(); ss.str("");
    }
    return features;
}

// ----------------------------------------------------------------------------

void testVocabCreation(const ConstDataSet& _DataSet, const DescriptorsSet& _Features) {
    using namespace TDBoW;
    // branching factor and depth levels
    const int k = 9;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    Vocabulary voc(k, L, weight, score);

    size_t count = 0;
    for(const auto& image : _DataSet) {
        count += image.size();
    }
    cout << "Features size: " << count << endl;

    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    using namespace std::chrono;
    auto start = system_clock::now();
    voc.create(_DataSet);
    auto end = system_clock::now();
    std::cout << "Spent time: " << duration_cast<milliseconds>(end - start).count() << " ms." << endl;
    cout << "... done!" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    std::vector<BowVector> vec(_Features.size());
    for(size_t i = 0; i < _Features.size(); i++) {
        voc.transform(_Features[i], vec[i]);
    // cout << "Bow vector " << i << ": " << vec[i] << endl;
    }
    cout << "=================================" << endl;
    for(size_t i = 0; i < _Features.size(); i++) {
        for(size_t j = 0; j < _Features.size(); j++) {
            cout << "Image " << i << " vs Image " << j << ": " << voc.score(vec[i], vec[j]) << endl;
        }
        cout << "=================================" << endl;
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.bin.qp");
    // voc.save("small_voc.yml", false); // save in YAML format
    cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const DescriptorsSet& _Features) {
    cout << "Creating a small database..." << endl;
    // load the vocabulary from disk
    Database db("small_voc.bin.qp", false); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(const auto& feature : _Features) {
        db.add(feature);
    }
    cout << "... done!" << endl;
    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;
    cout << "==============================" << endl;
    for(size_t i = 0; i < _Features.size(); i++) {
        // Ignore the bow vector result and don't limit result number
        auto ret = db.query(_Features[i], nullptr, 0);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
        cout << "==============================" << endl;
    }
}

// ----------------------------------------------------------------------------
