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
 * File: BowVector.cpp
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see the LICENSE.txt file
 *
 */

#include <fstream>
#include <cmath>

#include <TDBoW/BowVector.h>

namespace TDBoW {

// --------------------------------------------------------------------------

BowVector::BowVector(const BowVector& _Obj) : m_bLocked(false) {
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

BowVector& BowVector::operator =(const BowVector& _Obj) {
    if(_Obj.m_bLocked) {
        while(true) {
            bool excepted = false;
            if(_Obj.m_bLocked.compare_exchange_strong(excepted, true)) {
                break;
            }
        }
    }
    SpinLock locker(m_bLocked);
    for(const auto& pair : _Obj) {
        insert(end(), pair);
    }
    _Obj.m_bLocked = false;
    return *this;
}

/**
 * @brief Adds a value to a word value existing in the vector, or creates a new
 * word with the given value
 * @param id word id to look for
 * @param v value to create the word with, or to add to existing word
 */
void BowVector::addWeight(const WordId _ID, const WordValue _Val) {
    SpinLock locker(m_bLocked);
    auto iter = lower_bound(_ID);
    if(iter != end() && !(key_comp()(_ID, iter -> first))) {
        iter -> second += _Val;
    } else {
        insert(iter, BowVector::value_type(_ID, _Val));
    }
}

/**
 * @brief Adds a word with a value to the vector only if this does not exist yet
 * @param _ID   Word id to look for
 * @param _Val  Value to give to the word if this does not exist
 */
void BowVector::addIfNotExist(const WordId _ID, const WordValue _Val) {
    SpinLock locker(m_bLocked);
    auto vit = this->lower_bound(_ID);
    if(vit == this->end() || (this->key_comp()(_ID, vit->first))) {
        this->insert(vit, BowVector::value_type(_ID, _Val));
    }
}

/**
 * @brief L1-Normalizes the values in the vector
 * @param _NormType   L1 or L2
 */
void BowVector::normalize(const LNorm _NormType) {
    double norm = 0.0;
    SpinLock locker(m_bLocked);
    switch(_NormType) {
    case L1:
        for(const auto& pair : *this) {
            norm += fabs(pair.second);
        }
        break;

    case L2:
        for(const auto& pair : *this) {
            norm += pow(pair.second, 2);
        }
        norm = sqrt(norm);
        break;
    }

    if(norm > 0.0) {
        for(auto& pair : *this) {
            pair.second /= norm;
        }
    }
}

/**
 * @brief Prints the content of the bow vector
 * @param _Out stream
 * @param _Vec bow vector
 * @return     ostream
 */
std::ostream& operator<< (std::ostream& _Out, const BowVector& _Vec) {
    SpinLock locker(_Vec.m_bLocked);
    if(_Vec.empty())return _Out << "<empty>";
    auto iter = _Vec.begin();
    _Out << '<' << iter -> first << ", " << iter -> second << '>';
    while(iter != _Vec.end()) {
        _Out << ", <" <<  iter -> first << ", " << iter -> second << '>';
        iter++;
    }
    return _Out;
}
/**
 * @brief Saves the bow vector as a vector in a binary file
 * @param _Filename
 */
void BowVector::saveBinary(const std::string& _Filename) const {
    SpinLock locker(m_bLocked);
    std::fstream f(_Filename.c_str(), std::ios::out|std::ios::binary);
    auto len = size();
    f.write((char*)&len, sizeof(len));
    for(const auto& pair : *this) {
        f.write((char*)&pair, sizeof(pair));
    }
    f.close();
}
/**
 * @brief Load the bow vector as a vector in a binary file
 * @param _Filename
 */
void BowVector::loadBinary(const std::string& _Filename) {
    SpinLock locker(m_bLocked);
    std::fstream f(_Filename.c_str(), std::ios::in|std::ios::binary);
    clear();
    auto len = size();
    f.read((char*)&len, sizeof(len));
    BowVector::value_type pair;
    for(size_t i = 0; i <len; i++) {
        f.read((char*)&pair, sizeof(pair));
        insert(end(), pair);
    }
    f.close();
}

/**
 * @brief Saves the bow vector as a vector in a MatLab file
 * @param _Filename
 * @param _Width     number of words in the vocabulary
 */
void BowVector::saveM(const std::string& _Filename, size_t _Width) const {
    SpinLock locker(m_bLocked);
    std::fstream f(_Filename.c_str(), std::ios::out);
    WordId last = 0;
    BowVector::const_iterator bit;
    for(bit = this->begin(); bit != this->end(); ++bit) {
        for(; last < bit->first; ++last) {
            f << "0 ";
        }
        f << bit->second << " ";
        last++;
    }
    for(; last < (WordId)_Width; ++last)f << "0 ";
    f.close();
}

// --------------------------------------------------------------------------

} // namespace TDBoW
