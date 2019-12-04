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

//DBoW2: bag-of-words library for C++ with generic descriptors
//
//Copyright (c) 2015 Dorian Galvez-Lopez. http://doriangalvez.com
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions
//are met:
//1. Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//3. The original author of the work must be notified of any
//   redistribution of source code or in binary form.
//4. Neither the name of copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS
//BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//POSSIBILITY OF SUCH DAMAGE.

/* *************************************************************************
   * File Name     : TemplatedDescriptor.hpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-11-20 10:37:21
   * Last Modified : smallchimney
   * Modified Time : 2019-12-03 11:21:06
************************************************************************* */
#ifndef __ROCKAUTO_DESCRIPTOR_HPP__
#define __ROCKAUTO_DESCRIPTOR_HPP__

#include <iostream>
#include <vector>

#include <eigen3/Eigen/Core>
#include <boost/filesystem.hpp>
#include "traits.h"

namespace TDBoW {

// local methods
#define LINE_LOG(content) __line_log(__FILE__, __LINE__, (content))
static std::string __line_log(const std::string&, const size_t&, const std::string&);

std::string toHex(const uint8_t& _Data) noexcept;
uint8_t fromHex(const std::string& _Str) noexcept(false);

// Generic class to encapsulate functions to manage descriptors.
template <typename TScalar, size_t L>
class TemplatedDescriptorUtil {
protected:
    static_assert(L != 0, "The parameter L cannot be zero!");

    template <typename T>
    using trait = traits::basic_traits<T>;

    template <typename Scalar, int Rows = 1>
    using Matrix = Eigen::Matrix<Scalar, Rows, L, Eigen::RowMajor>;

public:
    typedef Matrix<TScalar>                            Descriptor;
    typedef typename trait<Descriptor>::Ptr            DescriptorPtr;
    typedef typename trait<Descriptor>::ConstPtr       DescriptorConstPtr;

    typedef Matrix<TScalar, Eigen::Dynamic>            DescriptorArray;
    typedef typename trait<DescriptorArray>::Ptr       DescriptorArrayPtr;
    typedef typename trait<DescriptorArray>::ConstPtr  DescriptorArrayConstPtr;

    typedef Matrix<uint8_t>                            BinaryDescriptor;
    typedef typename trait<BinaryDescriptor>::Ptr      BinaryDescriptorPtr;
    typedef typename trait<BinaryDescriptor>::ConstPtr BinaryDescriptorConstPtr;

    typedef Matrix<uint8_t, Eigen::Dynamic>            BinaryDescriptorArray;
    typedef typename trait<BinaryDescriptor>::Ptr      BinaryDescriptorArrayPtr;
    typedef typename trait<BinaryDescriptor>::ConstPtr BinaryDescriptorArrayConstPtr;

    typedef double_t distance_type;
    typedef std::function<Descriptor(
            const std::vector<DescriptorConstPtr>&)> MeanCallback;
    typedef std::function<distance_type(
            const Descriptor& _A, const Descriptor& _B)> DistanceCallback;

    /** @brief The "images by features" data */
    typedef std::vector<std::vector<DescriptorPtr>>      DataSet;
    typedef std::vector<std::vector<DescriptorConstPtr>> ConstDataSet;

    // Dataset mean value and distance

    /**
     * @breif  Make shared of the original dataset
     * @author smallchimney
     * @param  _Input  The original dataset, will be cleared after process.
     * @return         The dataset can be shared
     */
    static DataSet make_shared(std::vector<std::vector<Descriptor>>& _Input) {
        DataSet ret(0);
        return make_shared(_Input, ret);
    }

    /**
     * @breif  Make shared of the original dataset
     * @author smallchimney
     * @param  _Input  The original dataset, will be cleared after process.
     * @return         The dataset can be shared
     */
    static DataSet make_shared(std::vector<DescriptorArray>& _Input) {
        DataSet ret(0);
        return make_shared(_Input, ret);
    }

    /**
     * @breif  Make shared of the original dataset
     * @author smallchimney
     * @param  _Input  The original dataset, will be cleared after process.
     * @param  _Output The dataset can be shared
     * @return _Output
     */
    static DataSet& make_shared(std::vector<std::vector<Descriptor>>& _Input, DataSet& _Output);

    /**
     * @breif  Make shared of the original dataset
     * @author smallchimney
     * @param  _Input  The original dataset, will be cleared after process.
     * @param  _Output The dataset can be shared
     * @return _Output
     */
    static DataSet& make_shared(std::vector<DescriptorArray>& _Input, DataSet& _Output);

    /**
     * @breif  make vector<vector<Ptr>> to vector<vector<const Ptr>>
     * @author smallchimney
     * @param  _Input  The shared dataset
     * @return         The const shared dataset
     */
    static ConstDataSet make_const(const DataSet& _Input);

    /**
     * @breif  Make const shared of the original dataset
     * @author smallchimney
     * @param  _Input  The original dataset, will be cleared after process.
     * @param  _Output The dataset can be shared
     * @return _Output
     */
    static ConstDataSet& make_const(std::vector<std::vector<Descriptor>>& _Input) {
        return make_const(make_shared(_Input));
    }

    /**
     * @breif  Make const shared of the original dataset
     * @author smallchimney
     * @param  _Input  The original dataset, will be cleared after process.
     * @return         The dataset can be shared
     */
    static ConstDataSet make_const(std::vector<DescriptorArray>& _Input) {
        return make_const(make_shared(_Input));
    }

    // Descriptor mean value and distance

    static Descriptor _meanValue(
            const std::vector<DescriptorConstPtr>& _Descriptors,
            const MeanCallback& _Callback) noexcept(false) {
        if(_Descriptors.empty()) {
            throw std::runtime_error(LINE_LOG("Cannot calculate mean value for empty set."));
        }
        return _Callback(_Descriptors);
    }

    static Descriptor meanValue(const std::vector<DescriptorConstPtr>& _Descriptors) noexcept(false) {
        if(typeid(TScalar) == typeid(uint8_t)) {
            return _meanValue(_Descriptors, binaryMean);
        } else if(typeid(TScalar) == typeid(float_t)) {
            return _meanValue(_Descriptors, valueMean);
        } else if(typeid(TScalar) == typeid(double_t)) {
            return _meanValue(_Descriptors, valueMean);
        } else {
            throw std::runtime_error(LINE_LOG("The scalar type is not support automatically, "
                                     "a processing function must be explicitly specified."));
        }
    }

    static distance_type _distance(
            const Descriptor& _A, const Descriptor& _B,
            const DistanceCallback& _Callback) {
        return _Callback(_A, _B);
    }

    static distance_type distance(const Descriptor& _A, const Descriptor& _B) noexcept(false) {
        if(typeid(TScalar) == typeid(uint8_t)) {
            return _distance(_A, _B, binaryDistance);
        } else if(typeid(TScalar) == typeid(float_t)) {
            return _distance(_A, _B, valueDistance);
        } else if(typeid(TScalar) == typeid(double_t)) {
            return _distance(_A, _B, valueDistance);
        } else {
            throw std::runtime_error(LINE_LOG("The scalar type is not support automatically, "
                                     "a processing function must be explicitly specified."));
        }
    }

    template <typename Matrix>
    static void toBinary(const Matrix& _Mat, std::ostream& _Out);

    template <typename Matrix>
    static void fromBinary(std::istream& _In, Matrix& _Mat);

    static std::string toString(const Descriptor& _Desc);

    static void fromString(const std::string& _In, Descriptor& _Desc) noexcept(false);

    static std::ostream& visualBinary(std::ostream& _Out,
            const BinaryDescriptor& _Descriptor);

    static std::ostream& visualBinary(std::ostream& _Out,
            const BinaryDescriptorArray& _Descriptor);

protected:
    static BinaryDescriptor binaryMean(const std::vector<BinaryDescriptorConstPtr>& _Descriptors);
    static Descriptor valueMean(const std::vector<DescriptorConstPtr>& _Descriptors);

    static distance_type binaryDistance(const BinaryDescriptor& _A, const BinaryDescriptor& _B);
    static distance_type valueDistance(const Descriptor& _A, const Descriptor& _B);

};


template <typename TScalar, size_t DescL>
typename TemplatedDescriptorUtil<TScalar, DescL>::DataSet&
TemplatedDescriptorUtil<TScalar, DescL>::make_shared(
        std::vector<std::vector<Descriptor>>& _Input, DataSet& _Output) {
    if(_Input.empty()) {
        throw std::runtime_error(LINE_LOG("Empty dataset."));
    }
    _Output.clear();
    _Output.shrink_to_fit();
    _Output.resize(_Input.size());
    size_t idx = _Input.size() - 1;
    while(!_Input.empty()) {
        const auto& image = _Input.back();
        auto& output = _Output[idx--];
        output.reserve(image.size());
        // todo: make a memory pool, and execute methods like `memcpy()`, or even reuse the original memory
        for(const auto& feature : image) {
            output.emplace_back(std::make_shared<Descriptor>(feature));
        }
        _Input.pop_back();
    }
    return _Output;
}

template <typename TScalar, size_t DescL>
typename TemplatedDescriptorUtil<TScalar, DescL>::DataSet&
TemplatedDescriptorUtil<TScalar, DescL>::make_shared(
        std::vector<DescriptorArray>& _Input, DataSet& _Output) {
    if(_Input.empty()) {
        throw std::runtime_error(LINE_LOG("Empty dataset."));
    }
    _Output.clear();
    _Output.shrink_to_fit();
    _Output.resize(_Input.size());
    size_t idx = _Input.size() - 1;
    while(!_Input.empty()) {
        const auto& image = _Input.back();
        auto& output = _Output[idx--];
        const auto rows = image.rows();
        output.reserve(rows);
        // todo: make a memory pool, and execute methods like `memcpy()`, or even reuse the original memory
        for(typename Descriptor::Index i = 0; i < rows; i++) {
            output.emplace_back(std::make_shared<Descriptor>());
            DescriptorArray::Map(output.back() -> data(), 1, DescL) = image.row(i);
        }
        _Input.pop_back();
    }
    return _Output;
}

template <typename TScalar, size_t DescL>
typename TemplatedDescriptorUtil<TScalar, DescL>::ConstDataSet
TemplatedDescriptorUtil<TScalar, DescL>::make_const(const DataSet& _Input) {
    ConstDataSet dataset;
    dataset.resize(_Input.size());
    for(size_t i = 0; i < _Input.size(); i++) {
        const auto& features = _Input[i];
        auto& copy = dataset[i];
        copy.resize(features.size());
        for(size_t j = 0; j < features.size(); j++) {
            copy[j] = features[j];
        }
    }
    return dataset;
}

template<typename TScalar, size_t L>
typename TemplatedDescriptorUtil<TScalar, L>::BinaryDescriptor
TemplatedDescriptorUtil<TScalar, L>::binaryMean(const std::vector<BinaryDescriptorConstPtr>& _Descriptors) {
    BinaryDescriptor mean = BinaryDescriptor::Zero();
    std::vector<size_t> sum(L * 8, 0);
    if(_Descriptors.size() == 1) {
        return *_Descriptors[0];
    }
    for(const auto& descriptor : _Descriptors) {
        const TScalar* d = descriptor -> data();
        for(size_t i = 0; i < L; i++, d++) {
            if(*d & (1 << 7)) ++sum[i * 8    ];
            if(*d & (1 << 6)) ++sum[i * 8 + 1];
            if(*d & (1 << 5)) ++sum[i * 8 + 2];
            if(*d & (1 << 4)) ++sum[i * 8 + 3];
            if(*d & (1 << 3)) ++sum[i * 8 + 4];
            if(*d & (1 << 2)) ++sum[i * 8 + 5];
            if(*d & (1 << 1)) ++sum[i * 8 + 6];
            if(*d & (1))      ++sum[i * 8 + 7];
        }
    }
    TScalar* p = mean.data();
    const auto N2 = _Descriptors.size() / 2 + _Descriptors.size() % 2;
    for(size_t i = 0; i < sum.size(); ++i) {
        if(sum[i] >= N2) {
            *p |= 1 << (7 - (i % 8));   // set bit
        }
        if(i % 8 == 7) ++p;
    }
    return mean;
}

template<typename TScalar, size_t L>
typename TemplatedDescriptorUtil<TScalar, L>::Descriptor
TemplatedDescriptorUtil<TScalar, L>::valueMean(const std::vector<DescriptorConstPtr>& _Descriptors) {
    Descriptor mean;
    auto num = _Descriptors.size();
    for(const auto& p : _Descriptors) {
        for(size_t i = 0; i < L; i++)
            mean[i] += (*p)[i] / num;
    }
    return mean;
}

template<typename TScalar, size_t L>
typename TemplatedDescriptorUtil<TScalar, L>::distance_type
TemplatedDescriptorUtil<TScalar, L>::binaryDistance(
        const BinaryDescriptor& _A, const BinaryDescriptor& _B) {
    // Bit count function got from:
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
    // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0
    assert(L % sizeof(uint64_t) == 0);

    const uint64_t *pa, *pb;
    pa = (uint64_t*)_A.data();
    pb = (uint64_t*)_B.data();

    uint64_t v, ret = 0;
    for(size_t i = 0; i < L / sizeof(uint64_t); ++i, ++pa, ++pb) {
        v = *pa ^ *pb;
        v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
        v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) &
                (uint64_t)~(uint64_t)0/15*3);
        v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
        ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >>
                (sizeof(uint64_t) - 1) * CHAR_BIT;
    }
    return ret;
}

template<typename TScalar, size_t L>
typename TemplatedDescriptorUtil<TScalar, L>::distance_type
TemplatedDescriptorUtil<TScalar, L>::valueDistance(
        const Descriptor& _A, const Descriptor& _B) {
    const auto& a = _A.row(0);
    const auto& b = _B.row(0);
    double sqd = 0.;
    for(size_t i = 0; i < L; i++) {
        sqd += pow(a[i] - b[i], 2);
    }
    return sqd;
}


template<typename TScalar, size_t L>
template <typename TMatrix>
void TemplatedDescriptorUtil<TScalar, L>::toBinary(const TMatrix& _Mat, std::ostream& _Out) {
    typedef typename TMatrix::Index Index;
    typedef typename TMatrix::Scalar Scalar;
    Index rows = _Mat.rows(), cols = _Mat.cols();
    _Out.write((char*)(&rows), sizeof(Index));
    _Out.write((char*)(&cols), sizeof(Index));
    _Out.write((char*)_Mat.data(), rows * cols * sizeof(Scalar) );
}

template<typename TScalar, size_t L>
template <typename TMatrix>
void TemplatedDescriptorUtil<TScalar, L>::fromBinary(std::istream& _In, TMatrix& _Mat) {
    typedef typename TMatrix::Index Index;
    typedef typename TMatrix::Scalar Scalar;
    Index rows = 0, cols = 0;
    _In.read((char*)(&rows), sizeof(Index));
    _In.read((char*)(&cols), sizeof(Index));
    _Mat.resize(rows, cols);
    _In.read((char*)_Mat.data(), rows * cols * sizeof(Scalar));
}

template<typename TScalar, size_t L>
std::string TemplatedDescriptorUtil<TScalar, L>::toString(const Descriptor& _Desc) {
    // confirm the string is valid
    static std::stringstream ss;
    ss.clear(); ss.str("");
    const auto* p = reinterpret_cast<const uint8_t*>(_Desc.data());
    ss << toHex(*p++);
    constexpr auto LEN = L * sizeof(TScalar);
    for(size_t i = 1; i < LEN; i++) {
        ss << ' ' << toHex(*p++);
    }
    return ss.str();
}

template<typename TScalar, size_t L>
void TemplatedDescriptorUtil<TScalar, L>::fromString(const std::string& _In, Descriptor& _Desc) noexcept(false) {
    // confirm the string is valid
    static std::stringstream ss;
    ss.str(_In);
    auto* p = reinterpret_cast<uint8_t*>(_Desc.data());
    constexpr auto LEN = L * sizeof(TScalar);
    std::string hex;
    for(size_t i = 0; i < LEN; i++, p++) {
        ss >> hex;
        *p = fromHex(hex);
    }
    if(ss.fail()) {
        throw std::runtime_error(LINE_LOG("Invalid data."));
    }
    ss.clear();
    ss.str("");
}

template<typename TScalar, size_t L>
std::ostream& TemplatedDescriptorUtil<TScalar, L>::visualBinary(
        std::ostream& _Out, const BinaryDescriptor& _Descriptor) {
    auto* p = _Descriptor.data();
    _Out << '<'
         << (*p & (1 << 7) ? '1' : '0')
         << (*p & (1 << 6) ? '1' : '0')
         << (*p & (1 << 5) ? '1' : '0')
         << (*p & (1 << 4) ? '1' : '0')
         << (*p & (1 << 3) ? '1' : '0')
         << (*p & (1 << 2) ? '1' : '0')
         << (*p & (1 << 1) ? '1' : '0')
         << (*p & (1 << 0) ? '1' : '0');
    p++;
    for(size_t i = 1; i < L; i++, p++) {
        _Out << ' '
             << (*p & (1 << 7) ? '1' : '0')
             << (*p & (1 << 6) ? '1' : '0')
             << (*p & (1 << 5) ? '1' : '0')
             << (*p & (1 << 4) ? '1' : '0')
             << (*p & (1 << 3) ? '1' : '0')
             << (*p & (1 << 2) ? '1' : '0')
             << (*p & (1 << 1) ? '1' : '0')
             << (*p & (1 << 0) ? '1' : '0');
    }
    _Out << '>';
    return _Out;
}

template<typename TScalar, size_t L>
std::ostream& TemplatedDescriptorUtil<TScalar, L>::visualBinary(
        std::ostream& _Out, const BinaryDescriptorArray& _Descriptor) {
    auto* p = _Descriptor.data();
    _Out << '<';
    auto rows = static_cast<size_t>(_Descriptor.rows());
    for(size_t i = 0; i < rows; i++) {
        _Out << (*p & (1 << 7) ? '1' : '0')
             << (*p & (1 << 6) ? '1' : '0')
             << (*p & (1 << 5) ? '1' : '0')
             << (*p & (1 << 4) ? '1' : '0')
             << (*p & (1 << 3) ? '1' : '0')
             << (*p & (1 << 2) ? '1' : '0')
             << (*p & (1 << 1) ? '1' : '0')
             << (*p & (1 << 0) ? '1' : '0');
        p++;
        for(size_t j = 1; j < L; j++, p++) {
            _Out << ' '
                 << (*p & (1 << 7) ? '1' : '0')
                 << (*p & (1 << 6) ? '1' : '0')
                 << (*p & (1 << 5) ? '1' : '0')
                 << (*p & (1 << 4) ? '1' : '0')
                 << (*p & (1 << 3) ? '1' : '0')
                 << (*p & (1 << 2) ? '1' : '0')
                 << (*p & (1 << 1) ? '1' : '0')
                 << (*p & (1 << 0) ? '1' : '0');
        }
        if(i != static_cast<size_t>(_Descriptor.rows() - 1)) {
            _Out << std::endl;
        }
    }
    return _Out << '>';
}

/* ********************************************************************************
 *                                LOCAL METHODS                                   *
 ******************************************************************************** */

char int2hex(const unsigned& _Val) noexcept(false) {
    switch(_Val) {
    case 0: case 1: case 2: case 3: case 4:
    case 5: case 6: case 7: case 8: case 9:
        return static_cast<char>(_Val + '0');
    case 10: case 11: case 12: case 13: case 14: case 15:
        return static_cast<char>(_Val + 'A' - 10);
    default:
        throw std::runtime_error(LINE_LOG("Invalid value."));
    }
}

unsigned hex2int(const char& _Ch) noexcept(false) {
    switch(_Ch) {
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            return static_cast<unsigned>(_Ch - '0');
        case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
            return static_cast<unsigned>(_Ch - 'a' + 10);
        case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
            return static_cast<unsigned>(_Ch - 'A' + 10);
        default:
            throw std::runtime_error(LINE_LOG("Invalid value."));
    }
}

std::string toHex(const uint8_t& _Data) noexcept {
    static std::stringstream ss;
    ss.clear(); ss.str("");
    ss << int2hex(static_cast<unsigned>(_Data / 16));
    ss << int2hex(static_cast<unsigned>(_Data % 16));
    return ss.str();
}

uint8_t fromHex(const std::string& _Str) noexcept(false) {
    if(_Str.length() != 2) {
        throw std::runtime_error(LINE_LOG("Not TDBoW default descriptor"
                                 "string(hex), please use custom fromString()"));
    }
    return static_cast<uint8_t>(hex2int(_Str[0]) * 16 + hex2int(_Str[1]));
}

#if ROCKAUTO_RELATIVE_LOG
#ifndef FOUND_BOOST_1_60
// since Ubuntu 16.04 install the boost with version 1.58, we don't use boost::filesystem::relative
boost::filesystem::path relativeTo(
        const boost::filesystem::path& _From,
        const boost::filesystem::path& _To) {
    using boost::filesystem::path;

    path::const_iterator fromIter = _From.begin();
    path::const_iterator toIter = _To.begin();

    while (fromIter != _From.end() && toIter != _To.end() && (*toIter) == (*fromIter)) {
        ++toIter;
        ++fromIter;
    }

    path finalPath = "";
    while (fromIter != _From.end()) {
        finalPath /= "..";
        ++fromIter;
    }

    while (toIter != _To.end()) {
        finalPath /= *toIter;
        ++toIter;
    }

    return finalPath;
}
#endif
#endif

std::string __line_log(const std::string& _File, const size_t& _Line, const std::string& _Content) {
    boost::filesystem::path file(_File);
    static std::stringstream ss;
    ss.clear();
    ss.str("");
#if ROCKAUTO_RELATIVE_LOG
    static const boost::filesystem::path ROOT(PKG_DIR);
#ifndef FOUND_BOOST_1_60
    const auto relPath = relativeTo(ROOT, file);
#else
    const auto relPath = boost::filesystem::relative(file, ROOT);
#endif
    ss << relPath.native() << ':' << _Line << ": " << _Content << std::endl;
#else
    ss << file.native() << ':' << _Line << ": " << _Content << std::endl;
#endif
    return ss.str();
}

#undef LINE_LOG

}   // namespace TDBoW

#endif //__ROCKAUTO_DESCRIPTOR_HPP__
