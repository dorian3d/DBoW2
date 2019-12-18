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
 * File: ScoringObject.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions to compute bow scores 
 * License: see the LICENSE.txt file
 *
 */

#ifndef __ROCKAUTO_TDBOW_SCORING_OBJECT_H__
#define __ROCKAUTO_TDBOW_SCORING_OBJECT_H__

#include "IndexedFile.h"
#include <functional>

namespace TDBoW {

/// Base class of scoring functions
class GeneralScoring {
public:
    /**
     * Computes the score between two vectors. Vectors must be sorted and
     * normalized if necessary
     * @param v (in/out)
     * @param w (in/out)
     * @return score
     */
    virtual WordValue score(const BowVector& _V1, const BowVector& _V2) const final;

    /**
     * @brief  Compute scores for all vectors in history, and return sorted result.
     * @author smallchimney
     * @param  _Vec           Vector to query
     * @param  _InvertedFile  Database.
     * @param  _MaxResults    Return size limit, {@code 0} stands no limit.
     * @param  _MinCommon     Entry common words selected limit, {@code 0} stands no limit.
     * @param  _MaxId         Entry ID selected limit, {@code 0} stands no limit.
     * @return                Selected and sorted entries results with scores.
     */
    virtual QueryResults score(const BowVector& _Vec, const InvertedFile& _InvertedFile,
            unsigned _MaxResults, unsigned _MinCommon, EntryId _MaxId) const final;

    /**
     * Returns whether a vector must be normalized before scoring according
     * to the scoring scheme
     * @param norm norm to use
     * @return true iff must normalize
     */
    virtual bool mustNormalize(LNorm &norm) const = 0;

    /// Log of epsilon
    static const WordValue LOG_EPS;

protected:
    typedef std::function<WordValue(
            const WordValue&, const WordValue&)> Calculator;

    typedef std::function<WordValue&(WordValue&)> Scaler;

    typedef std::function<void(WordValue&,
            BowVector::const_iterator&, BowVector::const_iterator&,
            const BowVector&, const BowVector&,
            const Calculator&, const Scaler&)> CompareCallback;

    typedef std::pair<WordValue, unsigned> Record;
    typedef std::map<EntryId, Record> Records;

    typedef std::function<void(const BowVector&,
            const InvertedFile&, Records::value_type&)> InvertedCallback;

public:
    GeneralScoring() = delete;
    explicit GeneralScoring(bool _SmallBetter,
            Calculator _Calculator, Scaler _Scalar = default_scaler,
            CompareCallback _AfterAllCallback = default_after_all_callback,
            CompareCallback _SmallerCallback = default_small_callback,
            InvertedCallback _InvertedCallback = default_inverted_callback,
            CompareCallback _BiggerCallback = default_bigger_callback,
            CompareCallback _EqualCallback = default_equal_callback)
            : m_bSmallBetter(_SmallBetter),
            m_cCalculator(std::move(_Calculator)),
            m_cScaler(std::move(_Scalar)),
            m_cEqCallback(std::move(_EqualCallback)),
            m_cSmallCallback(std::move(_SmallerCallback)),
            m_cBiggerCallback(std::move(_BiggerCallback)),
            m_cAfterAllCallback(std::move(_AfterAllCallback)),
            m_cInvertedCallback(std::move(_InvertedCallback)) {}
    virtual ~GeneralScoring() = default;

protected:

    /** @brief Do nothing */
    static WordValue& default_scaler(WordValue& _Val) { return _Val; }

    /** @brief Do nothing */
    static void default_inverted_callback(const BowVector&,
            const InvertedFile&, Records::value_type&) {}

    /** @brief Call calculator */
    static void default_equal_callback(WordValue& _Score,
            BowVector::const_iterator& _It1, BowVector::const_iterator& _It2,
            const BowVector& _V1, const BowVector& _V2,
            const Calculator& _Cal, const Scaler& _Scaler) {
        const auto& vi = _It1++ -> second;
        const auto& wi = _It2++ -> second;
        _Score += _Cal(vi, wi);
    }

    /** @brief Call scaler */
    static void default_after_all_callback(WordValue& _Score,
            BowVector::const_iterator& _It1, BowVector::const_iterator& _It2,
            const BowVector& _V1, const BowVector& _V2,
            const Calculator& _Cal, const Scaler& _Scaler) {
        _Scaler(_Score);
    }

    /**
     * @brief  Default process when {@code _It1->first < _It2->first}.
     *          Move V1 forward to {@code _It1 = (first element >= _It2->id)}
     */
    static void default_small_callback(WordValue& _Score,
            BowVector::const_iterator& _It1, BowVector::const_iterator& _It2,
            const BowVector& _V1, const BowVector& _V2,
            const Calculator& _Cal, const Scaler& _Scaler) {
        _It1 = _V1.lower_bound(_It2 -> first);
    }

    /**
     * @brief  Default process when {@code _It1->first > _It2->first}.
     *          Move V2 forward to {@code _It2 = (first element >= _It1->id)}
     */
    static void default_bigger_callback(WordValue& _Score,
            BowVector::const_iterator& _It1, BowVector::const_iterator& _It2,
            const BowVector& _V1, const BowVector& _V2,
            const Calculator& _Cal, const Scaler& _Scaler) {
        _It2 = _V2.lower_bound(_It1 -> first);
    }

private:
    /** {@true} if smaller is better */
    bool m_bSmallBetter;
    /** @breif Used to calculate weight from two value */
    Calculator m_cCalculator;
    /** @breif Used to scale the value */
    Scaler m_cScaler;

    CompareCallback m_cEqCallback, m_cSmallCallback, m_cBiggerCallback;
    CompareCallback m_cAfterAllCallback;

    InvertedCallback m_cInvertedCallback;
};

/** 
 * Macro for defining Scoring classes
 * @param NAME name of class
 * @param MUST_NORMALIZE if vectors must be normalized to compute the score
 * @param NORM type of norm to use when MUST_NORMALIZE
 */
#define __SCORING_CLASS(NAME, MUST_NORMALIZE, NORM)\
class NAME: public GeneralScoring {\
public:\
    NAME ();\
    /** \
     * Says if a vector must be normalized according to the scoring function \
     * @param norm (out) if true, norm to use
     * @return true iff vectors must be normalized \
     */ \
    bool mustNormalize(LNorm &norm) const override {\
        norm = NORM;\
        return MUST_NORMALIZE;\
    }\
}
  
/// L1 Scoring object
__SCORING_CLASS(L1Scoring, true, L1);

/// L2 Scoring object
__SCORING_CLASS(L2Scoring, true, L2);

/// Chi square Scoring object
__SCORING_CLASS(ChiSquareScoring, true, L1);

/// KL divergence Scoring object
__SCORING_CLASS(KLScoring, true, L1);

/// Bhattacharyya Scoring object
__SCORING_CLASS(BhattacharyyaScoring, true, L1);

/// Dot product Scoring object
__SCORING_CLASS(DotProductScoring, false, L1);

#undef __SCORING_CLASS
  
} // namespace TDBoW

#endif  // __ROCKAUTO_TDBOW_SCORING_OBJECT_H__
