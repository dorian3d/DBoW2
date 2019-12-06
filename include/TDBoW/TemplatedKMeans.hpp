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
   * File Name     : TemplatedKMeans.hpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-11-27 14:01:21
   * Last Modified : smallchimney
   * Modified Time : 2019-12-02 19:24:48
************************************************************************* */
#ifndef __ROCKAUTO_TEMPLATED_K_MEANS_HPP__
#define __ROCKAUTO_TEMPLATED_K_MEANS_HPP__

#include <random>
#include <functional>

#include "TemplatedDescriptor.hpp"

namespace TDBoW {

// local methods

/**
 * @brief  Random float point value from [_Left, _Right)
 * @author smallchimney
 * @tparam Scalar  The scalar should be float point type.
 * @param  _Left   The min value of the range.
 * @param  _Right  The max value of the range.
 * @return         A random float point value
 */
template <typename Scalar>
static Scalar randomReal(const Scalar& _Left, const Scalar& _Right);

/**
 * @brief  Random integral value from [_Left, _Right]
 * @author smallchimney
 * @tparam Scalar  The scalar should be integral type.
 * @param  _Left   The min value of the range.
 * @param  _Right  The max value of the range.
 * @return         A random float point value
 */
template <typename Scalar>
static Scalar randomInt(const Scalar& _Left, const Scalar& _Right);

/**
 * @brief  Actually this is a robust k-means++ implement, can easily reused
 *         by define the template parameter as data type and distance type.
 *         e.g. {@code template <typename DataType, typename distance_type>}
 *         And using the function as parameter:
 *         {@code std::function<distance_type(const DataType&, const DataType&)>}
 * @author smallchimney
 * @tparam DescriptorUtil  The template parameters set of data type, distance
 *                         type, `meanValue()` and `distance()`
 */
template <class DescriptorUtil>
class TemplatedKMeans {
protected:
    // Load the typename from DescriptorUtil
    typedef typename DescriptorUtil::distance_type       distance_type;
    typedef typename DescriptorUtil::Descriptor          Descriptor;
    typedef typename DescriptorUtil::DescriptorPtr       DescriptorPtr;
    typedef typename DescriptorUtil::DescriptorConstPtr  DescriptorConstPtr;
    typedef typename DescriptorUtil::MeanCallback        MeanCallback;
    typedef typename DescriptorUtil::DistanceCallback    DistanceCallback;

public:
    typedef std::function<void(const size_t&, const std::vector<DescriptorConstPtr>&,
            std::vector<Descriptor>&, DistanceCallback, MeanCallback)> InitMethods;

    TemplatedKMeans() = delete;
    TemplatedKMeans(const size_t& _K): m_ulK(_K) {}
    ~TemplatedKMeans() = default;

    /**
     * @brief  Run k-means++ on the descriptors
     * @author smallchimney
     * @param  _Descriptors     All descriptors to be grouped.
     * @param  _Centers  (out)  Centers value of each result cluster.
     * @param  _Clusters (out)  Descriptors of each cluster.
     * @param  _Init            Initial seed selected methods, default
     *                          by K-Means++.
     * @param  _DistF           Distance function.
     * @param  _MeanF           Mean value function.
     */
    void process(const std::vector<DescriptorConstPtr>& _Descriptors,
                 std::vector<Descriptor>& _Centers,
                 std::vector<std::vector<DescriptorConstPtr>>& _Clusters,
                 InitMethods _Init = initiateClustersKM2nd,
                 DistanceCallback _DistF = &DescriptorUtil::distance,
                 MeanCallback _MeanF = &DescriptorUtil::meanValue) noexcept(false);

protected:

    /**
     * @breif  Found k clusters' center from the given descriptor sets
     *         by running the initial step of k-means
     * @author smallchimney
     * @param  _Descriptors   Input descriptors.
     * @param  _Centers (out) resulting clusters.
     * @param  _DistF         Distance function.
     */
    static void initiateClustersKM(const size_t& _K,
            const std::vector<DescriptorConstPtr>& _Descriptors,
            std::vector<Descriptor>& _Centers, DistanceCallback _F, MeanCallback _M);

    /**
     * @breif  Found k clusters' center from the given descriptor sets
     *         by running the initial step of k-means++
     * @author smallchimney
     * @param  _Descriptors   Input descriptors.
     * @param  _Centers (out) resulting clusters.
     * @param  _DistF         Distance function.
     */
    static void initiateClustersKMpp(const size_t& _K,
            const std::vector<DescriptorConstPtr>& _Descriptors,
            std::vector<Descriptor>& _Centers, DistanceCallback _F, MeanCallback _M);

    /**
     * @breif  Found k clusters' center from the given descriptor sets
     *         by running the initial step of k-meansⅡ
     * @author smallchimney
     * @param  _Descriptors   Input descriptors.
     * @param  _Centers (out) resulting clusters.
     * @param  _DistF         Distance function.
     */
    static void initiateClustersKM2nd(const size_t& _K,
            const std::vector<DescriptorConstPtr>& _Descriptors,
            std::vector<Descriptor>& _Centers, DistanceCallback _F, MeanCallback _M);


private:
    size_t m_ulK;
};

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::process(
        const std::vector<DescriptorConstPtr>& _Descriptors,
        std::vector<Descriptor>& _Centers,
        std::vector<std::vector<DescriptorConstPtr>>& _Clusters,
        InitMethods _Init, DistanceCallback _DistF, MeanCallback _MeanF) noexcept(false) {
    _Centers.clear();_Centers.shrink_to_fit();
    _Centers.reserve(m_ulK);
    _Clusters.clear();_Clusters.shrink_to_fit();
    _Clusters.reserve(m_ulK);
    // No need for run k-means
    if(_Descriptors.size() <= m_ulK) {
        // Trivial case: one cluster per feature
        _Clusters.assign(_Descriptors.size(), std::vector<DescriptorConstPtr>());
        for(size_t i = 0; i < _Descriptors.size(); i++) {
            _Clusters[i].emplace_back(_Descriptors[i]);
            _Centers.emplace_back(*_Descriptors[i]);
        }
        return;
    }
    // select clusters and groups with k-means
    bool firstTime = true;
    // to check if clusters move after iterations
    std::vector<size_t> currentBelong, previousBelong;
    while(true) {
        // 1. Calculate clusters
        if(firstTime) {
            // random sample
            _Init(m_ulK, _Descriptors, _Centers, _DistF, _MeanF);
            firstTime = false;
        } else {
            // calculate cluster centres
            for(size_t i = 0; i < _Centers.size(); i++) {
                auto& cluster = _Clusters[i];
                if(cluster.empty()) {
                    firstTime = true;
                    break;  // re-run the k-means
                }
                std::vector<DescriptorConstPtr> descSet(0);
                descSet.reserve(cluster.size());
                for(const auto& descriptor : cluster) {
                    descSet.emplace_back(descriptor);
                }
                _Centers[i] = _MeanF(descSet);
            }
            if(firstTime)continue;
        }

        // 2. Associate features with clusters
        // calculate distances to cluster centers
        _Clusters.assign(_Centers.size(), std::vector<DescriptorConstPtr>());
        currentBelong.resize(_Descriptors.size());

        for(size_t i = 0; i < _Descriptors.size(); i++) {
            const auto& descriptor = *_Descriptors[i];
            distance_type min = _DistF(descriptor, _Centers[0]);
            size_t minIdx = 0;
            for(size_t idx = 1; idx < _Centers.size(); idx++) {
                distance_type distance = DescriptorUtil::distance(descriptor, _Centers[idx]);
                if(distance < min) {
                    min = distance;
                    minIdx = idx;
                }
            }
            _Clusters[minIdx].emplace_back(_Descriptors[i]);
            currentBelong[i] = minIdx;
        }
        // k-means++ ensures all the clusters has any feature associated with them

        // 3. check convergence
        if(currentBelong == previousBelong)break;
        previousBelong = currentBelong;
    }
}

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::initiateClustersKM(const size_t& _K,
        const std::vector<DescriptorConstPtr>& _Descriptors,
        std::vector<Descriptor>& _Centers, DistanceCallback _F, MeanCallback _M) {
    // Random K seeds
    _Centers.clear();
    _Centers.reserve(_K);
    for(size_t i = 0; i < _K; i++) {
        auto featureIdx = randomInt<size_t>(0, _Descriptors.size() - 1);
        _Centers.emplace_back(*_Descriptors[featureIdx]);
    }
}

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::initiateClustersKMpp(const size_t& _K,
        const std::vector<DescriptorConstPtr>& _Descriptors,
        std::vector<Descriptor>& _Centers, DistanceCallback _F, MeanCallback _M) {
    // Implements k-means++ seeding algorithm
    // Algorithm:
    // 1. Choose one center uniformly at random from among the data points.
    // 2. For each data point x, compute D(x), the distance between x and the nearest
    //    center that has already been chosen.
    // 3. Add one new data point as a center. Each point x is chosen with probability
    //    proportional to D(x)^2.
    // 4. Repeat Steps 2 and 3 until k centers have been chosen.
    // 5. Now that the initial centers have been chosen, proceed using standard k-means
    //    clustering.
    _Centers.clear();
    _Centers.reserve(_K);

    // 1.
    // create first cluster
    auto featureIdx = randomInt<size_t>(0, _Descriptors.size() - 1);
    _Centers.emplace_back(*_Descriptors[featureIdx]);

    // compute the initial distances
    std::vector<distance_type> minDist(0);
    minDist.reserve(_Descriptors.size());
    for(const auto& descriptor : _Descriptors) {
        minDist.emplace_back(_F(*descriptor, _Centers.back()));
    }

    while(_Centers.size() < _K) {
        // 2.
        const auto& center = _Centers.back();
        for(size_t i = 0; i < _Descriptors.size(); i++) {
            auto& distance = minDist[i];
            if(distance > 0) {
                distance = std::min(distance, _F(*_Descriptors[i], center));
            }
        }

        // 3.
        distance_type sum = std::accumulate(minDist.begin(), minDist.end(), 0.);
        if(sum == 0) {
            // Trivial case: one cluster per feature
            return;
        }
        if(sum < 0) {
            throw std::runtime_error(TDBOW_LOG("get negative sum, please check the `distance()`"));
        }

        auto limit = randomReal<distance_type>(0, sum);
        size_t idx = 0; sum = 0;
        while(idx < minDist.size() && sum <= limit) {
            sum += minDist[idx++];
        }
        _Centers.emplace_back(*_Descriptors[idx - 1]);
    }
}

template <class DescriptorUtil>
void TemplatedKMeans<DescriptorUtil>::initiateClustersKM2nd(const size_t& _K,
        const std::vector<DescriptorConstPtr>& _Descriptors,
        std::vector<Descriptor>& _Centers, DistanceCallback _F, MeanCallback _M) {
    // Implements k-meansⅡ seeding algorithm
    const auto LIMIT = static_cast<size_t>(log(_Descriptors.size()) / log(2)) * _K;

    if(_Descriptors.size() <= LIMIT) {
        initiateClustersKMpp(_K, _Descriptors, _Centers, _F, _M);
        return;
    }

    std::vector<size_t> choices(_Descriptors.size());
    for(size_t i = 1; i < _Descriptors.size(); i++) {
        choices[i] = i;
    }
    std::vector<DescriptorConstPtr> seeds;
    seeds.reserve(LIMIT);
    for(size_t i = 0; i < LIMIT; i++) {
        auto idx = randomInt<size_t>(0, choices.size() - 1);
        seeds.emplace_back(std::make_shared<Descriptor>(*_Descriptors[choices[idx]]));
        choices.erase(choices.begin() + idx);
    }
    std::vector<std::vector<DescriptorConstPtr>> ignored;
    TemplatedKMeans<DescriptorUtil>(_K).process(seeds, _Centers, ignored, initiateClustersKMpp, _F, _M);
}

/* ********************************************************************************
 *                               RANDOM METHODS                                   *
 ******************************************************************************** */

template <typename Scalar>
Scalar randomReal(const Scalar& _Left, const Scalar& _Right) {
    static_assert(std::is_floating_point<Scalar>::value,
                  "result_type must be a floating point type");
    static std::default_random_engine e(std::random_device().operator()());
    typedef std::uniform_real_distribution<Scalar> Uniform;
    typedef typename Uniform::param_type param;
    static Uniform uniform;
    uniform.param(param(_Left, _Right));
    return uniform(e);
}

template <typename Scalar>
Scalar randomInt(const Scalar& _Left, const Scalar& _Right) {
    static_assert(std::is_integral<Scalar>::value,
                  "template argument must be an integral type");
    static std::default_random_engine e(std::random_device().operator()());
    typedef std::uniform_int_distribution<Scalar> Uniform;
    typedef typename Uniform::param_type param;
    static Uniform uniform;
    uniform.param(param(_Left, _Right));
    return uniform(e);
}


}   // namespace TDBoW

#endif //__ROCKAUTO_TEMPLATED_K_MEANS_HPP__
