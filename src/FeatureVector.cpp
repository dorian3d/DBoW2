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
 * File: FeatureVector.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see the LICENSE.txt file
 *
 */

#include <FeatureVector.h>

namespace TDBoW {

FeatureVector::FeatureVector(const FeatureVector& _Obj) : m_bLocked(false) {
    if(_Obj.m_bLocked) {
        while(true) {
            bool excepted = false;
            if(_Obj.m_bLocked.compare_exchange_strong(excepted, true)) {
                break;
            }
        }
    }
    for(const auto& pair : _Obj) {
        insert(end(), pair);
    }
    _Obj.m_bLocked = false;
}

// ---------------------------------------------------------------------------

void FeatureVector::addFeature(NodeId _ID, unsigned int _FeatureIdx) {
    SpinLock locker(m_bLocked);
    auto iter = lower_bound(_ID);
    if(iter != end() && iter -> first == _ID) {
        iter -> second.emplace_back(_FeatureIdx);
    } else {
        iter = insert(iter, value_type(_ID, std::vector<unsigned>()));
        iter -> second.emplace_back(_FeatureIdx);
    }
}

// ---------------------------------------------------------------------------

std::ostream& operator <<(std::ostream& _Out,
        const FeatureVector::value_type& pair) {
    _Out << '<' << pair.first << ": [";
    const auto& data = pair.second;
    auto iter = data.begin();
    if(!data.empty()) {
        _Out << *iter++;
    }
    while(iter != data.end()) {
        _Out << ", " << *iter++;
    }
    return _Out << "]>";
}

// ---------------------------------------------------------------------------

std::ostream& operator <<(std::ostream& _Out, const FeatureVector& _Vec) {
    SpinLock locker(_Vec.m_bLocked);
    if(_Vec.empty())return _Out << "[empty]";
    auto iter = _Vec.begin();
    _Out << '[' << *iter++;
    while(iter != _Vec.end()) {
        _Out << ", " << *iter++;
    }
    return _Out << "]";
}

// ---------------------------------------------------------------------------

} // namespace DBoW2
