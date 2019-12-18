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
/* *************************************************************************
   * File Name     : demo.cpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-12-05 16:55:11
   * Last Modified : smallchimney
   * Modified Time : 2019-12-06 13:37:25
************************************************************************* */
// TDBoW and template typedef
#include <TDBoW/PCBridge.h>

// PCL
#include <pcl/io/pcd_io.h>
#pragma GCC diagnostic ignored "-Wpedantic"
#include <pcl/keypoints/harris_3d.h>
#ifdef FOUND_OPENMP
#include <pcl/features/fpfh_omp.h>
#else
#include <pcl/features/fpfh.h>
#endif

#include <chrono>

using std::cout; using std::endl;
typedef TDBoW::FPFH33Database    Database;
typedef TDBoW::FPFH33Vocabulary  Vocabulary;
typedef Vocabulary::Descriptor   Descriptor;
typedef Vocabulary::ConstDataSet ConstDataSet;
typedef Vocabulary::DescriptorsSet DescriptorsSet;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

std::vector<std::string> loadFeatures(DescriptorsSet&);
void testVocabCreation(const ConstDataSet&, const DescriptorsSet&, const std::vector<std::string>&);
void testDatabase(const DescriptorsSet&);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait() {
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

// ----------------------------------------------------------------------------

int main() {
    // Load files, calculate FPFH-33 descriptors and do transform
    // DescriptorArray and Descriptor are recommended types when query
    DescriptorsSet features;
    auto files = loadFeatures(features);

    // In this simple case, we had not prepare too many data, so we
    // use the same data for both create and query.
    // `make_shared`(inner method) will drop the original data, so make copy
    auto copy = features;
    // DataSet type is only used in vocabulary create.
    auto dataset = Vocabulary::util::make_const(copy);
    // Vocabulary testing
    testVocabCreation(dataset, features, files);

    wait();

    // Vocabulary testing
    testDatabase(features);

    return 0;
}

// ----------------------------------------------------------------------------

std::vector<std::string> loadFeatures(DescriptorsSet& _Features) {
    using namespace boost::filesystem;
    const auto resourceDir = path(PKG_DIR)/"demo/pc/pointclouds";
    // Automatically find all `.pcd` files
    std::vector<path> files;
    for(const auto& file : recursive_directory_iterator(resourceDir)) {
        if(!is_regular_file(file))continue;
        const auto& filePath = file.path();
        auto extension = filePath.extension().native();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        if(extension != ".pcd")continue;
        files.emplace_back(filePath);
    }
    // Reserve the features space
    _Features.clear();
    _Features.shrink_to_fit();
    _Features.resize(files.size());
    // Iterator each file
    typedef pcl::PointCloud<pcl::PointXYZ>    PointCloudXYZ;
    typedef pcl::PointCloud<pcl::PointXYZI>   PointCloudXYZI;
    typedef pcl::PointCloud<pcl::PointNormal> PointCloudNormal;
    pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI, pcl::Normal> harris;
#ifdef FOUND_OPENMP
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
    fpfh.setNumberOfThreads(std::thread::hardware_concurrency());
#else
    pcl::FPFHEstimation<PointXYZ, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
#endif
    std::vector<std::string> names(0);
    names.reserve(files.size());
    for(size_t i = 0; i < files.size(); i++) {
        // Load pointcloud
        auto input = boost::make_shared<PointCloudXYZ>();
        if(pcl::io::loadPCDFile(files[i].native(), *input) == -1)continue;
        auto output = boost::make_shared<PointCloudXYZI>();
        // Calculate keypoints using 3D harris
        harris.setInputCloud(input);
        harris.setNonMaxSupression(true);
        harris.setRadius(0.5f);
        harris.setThreshold(0.01f);
        harris.compute(*output);
        auto indices = harris.getKeypointsIndices();
        // Compute normal for each points
        auto normals = boost::make_shared<PointCloudNormal>();
        auto kdTree = boost::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
        pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
        ne.setInputCloud(input);
        ne.setSearchMethod(kdTree);
        ne.setKSearch(10);
        ne.compute(*normals);
        // Compute descriptors for each points
        auto descriptors = boost::make_shared<pcl::PointCloud<pcl::FPFHSignature33>>();
        fpfh.setInputCloud(input);
        fpfh.setInputNormals(normals);
        fpfh.setSearchMethod(kdTree);
        fpfh.setRadiusSearch(0.5f);
        fpfh.compute(*descriptors);
        // Collect the descriptors
        typedef Vocabulary::Descriptor Descriptor;
        auto& feature = _Features[i];
        feature.resize(indices -> indices.size());
        for(size_t j = 0; j < indices -> indices.size(); j++) {
            const auto& index = static_cast<size_t>(indices -> indices[j]);
            const auto& descriptor = descriptors -> at(index);
            feature[j] = Descriptor::Map(
                    descriptor.histogram, 1, pcl::FPFHSignature33::descriptorSize());
        }
        names.emplace_back(files[i].filename().native());
    }
    return names;
}

// ----------------------------------------------------------------------------

void testVocabCreation(const ConstDataSet& _DataSet,
        const DescriptorsSet& _Features, const std::vector<std::string>& _Names) {
    // branching factor and depth levels
    using namespace TDBoW;
    const int k = 6;
    const int L = 3;
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    Vocabulary voc(k, L, weight, score);

    assert(_Names.size() == _Features.size());
    size_t PRINT_LEN = 0;
    for(const auto& name : _Names) {
        PRINT_LEN = std::max(PRINT_LEN, name.length());
    }
    PRINT_LEN *= 2;

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

    voc.stopWords(0.5);
    cout << "Vocabulary stop words by weight: 0.5" << endl;

    cout << "Vocabulary information: " << endl
         << voc << endl << endl;

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    std::vector<BowVector> vec(_Features.size());
    for(size_t i = 0; i < _Features.size(); i++) {
        voc.transform(_Features[i], vec[i]);
        // cout << "Bow vector " << i << ": " << vec[i] << endl;
    }
    cout << "======================================================" << endl;
    for(size_t i = 0; i < _Features.size(); i++) {
        for(size_t j = 0; j < _Features.size(); j++) {
            cout << _Names[i] << " vs " << _Names[j];
            size_t space = _Names[i].length() + _Names[j].length();
            while(space++ <= PRINT_LEN) {
                cout << ' ';
            }
            cout << ": " << voc.score(vec[i], vec[j]) << endl;
        }
        cout << "======================================================" << endl;
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
