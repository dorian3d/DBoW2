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

/**
 * File: FeatureVector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __ROCKAUTO_TDBOW_FEATURE_VECTOR_H__
#define __ROCKAUTO_TDBOW_FEATURE_VECTOR_H__

#include "BowVector.h"
#include "traits.h"

namespace TDBoW {

/**
 * @brief  Vector of nodes with indexes of local features.
 *          Standard map with two thread-safety function,
 *         note that the implement is not complete thread_safety.
 *         Take care when using original std::map's methods
 *         in multiply threads context.
 * @author smallchimney
 */
class FeatureVector: public std::map<NodeId, std::vector<size_t>> {
protected:
    template <typename T>
    using trait = traits::basic_traits<T>;

public:
    typedef trait<FeatureVector>::Ptr      Ptr;
    typedef trait<FeatureVector>::ConstPtr ConstPtr;

    /**
     * Constructor
     */
    FeatureVector(): m_bLocked(false) {};

    FeatureVector(const FeatureVector& _Obj);

    /**
     * Destructor
     */
    ~FeatureVector() = default;

    /**
     * @brief Adds a feature to an existing node, or adds a new node with an initial
     * feature
     * @param _ID         node id to add or to modify
     * @param _FeatureIdx index of feature to add to the given node
     */
    void addFeature(NodeId _ID, size_t _FeatureIdx);

    /**
     * @brief Sends a string versions of the feature vector through the stream
     * @param _Out stream
     * @param _Vec feature vector
     */
    friend std::ostream& operator<<(std::ostream& _Out, const FeatureVector& _Vec);

    /**
     * @brief Sends a string versions of the feature vector through the stream
     * @param _Out stream
     * @param pair value type of feature vector
     */
    friend std::ostream& operator<<(std::ostream& _Out, const FeatureVector::value_type& pair);

private:
    /** @brief set as a label who get the control */
    mutable std::atomic_bool m_bLocked;
};
typedef FeatureVector::Ptr      FeatureVectorPtr;
typedef FeatureVector::ConstPtr FeatureVectorConstPtr;

} // namespace TDBoW

#endif  // __ROCKAUTO_TDBOW_FEATURE_VECTOR_H__

