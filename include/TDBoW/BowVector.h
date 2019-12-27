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
 * File: BowVector.h
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __ROCKAUTO_TDBOW_BOW_VECTOR_H__
#define __ROCKAUTO_TDBOW_BOW_VECTOR_H__

#include "traits.h"
#include "SpinLock.h"

#include <iostream>
#include <map>
#include <vector>

namespace TDBoW {

/// Id of words
typedef unsigned int WordId;

/// Value of a word
typedef double WordValue;

/// Id of nodes in the vocabulary tree
typedef unsigned int NodeId;

/// L-norms for normalization
enum LNorm {
    L1,
    L2
};

/// Weighting type
enum WeightingType {
    TF_IDF,
    TF,
    IDF,
    BINARY
};

/// Scoring type
enum ScoringType {
    L1_NORM,
    L2_NORM,
    CHI_SQUARE,
    KL,
    BHATTACHARYYA,
    DOT_PRODUCT
};

/**
 * @brief  Vector of words to represent images.
 *          Standard map with two thread-safety function,
 *         note that the implement is not complete thread_safety.
 *         Take care when using original std::map's methods
 *         in multiply threads context.
 * @author smallchimney
 */
class BowVector: public std::map<WordId, WordValue> {
protected:
	template <typename T>
	using trait = traits::basic_traits<T>;

	typedef std::map<WordId, WordValue> Base;

public:
	typedef trait<BowVector>::Ptr      Ptr;
	typedef trait<BowVector>::ConstPtr ConstPtr;

	// continue public some declaring
	typedef Base::iterator       iterator;
	typedef Base::const_iterator const_iterator;

	/** 
	 * Constructor
	 */
	BowVector() : m_bLocked(false) {}

	BowVector(const BowVector& _Obj);

	/**
	 * Destructor
	 */
	~BowVector() = default;

	BowVector& operator =(const BowVector& _Obj);

	/**
	 * @brief Adds a value to a word value existing in the vector, or creates a new
	 * word with the given value
	 * @param id word id to look for
	 * @param v value to create the word with, or to add to existing word
	 */
	void addWeight(WordId _ID, WordValue _Val);
	
	/**
	 * @brief Adds a word with a value to the vector only if this does not exist yet
	 * @param _ID   Word id to look for
	 * @param _Val  Value to give to the word if this does not exist
	 */
	void addIfNotExist(WordId _ID, WordValue _Val);

	/**
	 * @brief L1-Normalizes the values in the vector
	 * @param _NormType   L1 or L2
	 */
	void normalize(LNorm _NormType);
	
	/**
	 * @brief Prints the content of the bow vector
	 * @param _Out stream
	 * @param _Vec bow vector
	 * @return     ostream
	 */
	friend std::ostream& operator<<(std::ostream& _Out, const BowVector& _Vec);
	
	/**
	 * @brief Saves the bow vector as a vector in a binary file
	 * @param _Filename
	 */
    void saveBinary(const std::string& _Filename) const;

    /**
	 * @brief Load the bow vector as a vector in a binary file
	 * @param _Filename
	 */
    void loadBinary(const std::string& _Filename);

    /**
	 * @brief Saves the bow vector as a vector in a MatLab file
	 * @param _Filename
	 * @param _Width     number of words in the vocabulary
	 */
    void saveM(const std::string& _Filename, size_t _Width) const;

private:
    /** @brief set as a label who get the control */
    mutable std::atomic_bool m_bLocked;
};
typedef BowVector::Ptr      BowVectorPtr;
typedef BowVector::ConstPtr BowVectorConstPtr;

} // namespace TDBoW

#endif	// __ROCKAUTO_TDBOW_BOW_VECTOR_H__
