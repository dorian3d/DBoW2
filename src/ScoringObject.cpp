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
 * File: ScoringObject.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions to compute bow scores 
 * License: see the LICENSE.txt file
 *
 */

#include <TDBoW/TemplatedVocabulary.hpp>

#include <numeric>

namespace TDBoW {

#define __CMP_CB [](\
    WordValue& _Score,\
    BowVector::const_iterator& _It1,\
    BowVector::const_iterator& _It2,\
    const BowVector& _V1, const BowVector& _V2,\
    const Calculator& _Cal, const Scaler&)

#define __IVT_CB [](\
    const BowVector& _Vec,\
    const InvertedFile& _InvertedFile,\
    Records::value_type& _Pair)

const WordValue GeneralScoring::LOG_EPS = log(std::numeric_limits<WordValue>::epsilon());

#define __CALL(NAME)\
NAME (ret, it1, it2, _V1, _V2, m_cCalculator, m_cScaler)
WordValue GeneralScoring::score(
        const BowVector& _V1, const BowVector& _V2) const {
    WordValue ret = 0;
    auto it1 = _V1.begin();
    auto it2 = _V2.begin();
    while(it1 != _V1.end() && it2 != _V2.end()) {
        if(it1 -> first == it2 -> first) {
            __CALL(m_cEqCallback);
        } else if(it1 -> first < it2 -> first) {
            __CALL(m_cSmallCallback);
        } else {
            __CALL(m_cBiggerCallback);
        }
    }
    __CALL(m_cAfterAllCallback);
    return ret;
}
#undef __CALL

QueryResults GeneralScoring::score(const BowVector& _Vec,
        const InvertedFile& _InvertedFile, const unsigned _MaxResults,
        const unsigned _MinCommon, const EntryId _MaxId) const {
    std::map<EntryId, Record> scores;
    for(const auto& word : _Vec) {
        const auto &wordId = word.first;
        const auto &qValue = word.second;
        // IFRows are sorted in ascending entry_id order
        for(const auto& pair : _InvertedFile[wordId]) {
            const auto& entryId = pair.entry_id;
            const auto& dValue  = pair.word_weight;
            if(_MaxId && entryId >= _MaxId)continue;
            WordValue value = m_cCalculator(qValue, dValue);
            if(value == 0)continue;
            auto iter = scores.lower_bound(entryId);
            if(iter != scores.end() && !(scores.key_comp()(entryId, iter -> first))) {
                iter -> second.first += value;
                iter -> second.second++;
            } else {
                scores.insert(iter, std::make_pair(entryId, std::make_pair(value, 1)));
            }
        } // for each inverted row
    } // for each query word
    QueryResults ret;
    ret.reserve(scores.size());
    for(auto& pair : scores) {
        if(pair.second.second < _MinCommon)continue;
        m_cInvertedCallback(_Vec, _InvertedFile, pair);
        ret.emplace_back(Result(pair.first, pair.second.first));
    }

    // Sort vector
    if(m_bSmallBetter) {
        std::sort(ret.begin(), ret.end(), std::less<Result>()); // NOLINT(modernize-use-transparent-functors)
    } else {
        std::sort(ret.begin(), ret.end(), std::greater<Result>()); // NOLINT(modernize-use-transparent-functors)
    }
    // (ret is sorted now [the best ... the worst])

    // Cut vector
    if(_MaxResults > 0 && ret.size() > _MaxResults) {
        ret.resize(_MaxResults);
    }
    ret.shrink_to_fit();
    // Scale
    for(auto& r : ret) {
        m_cScaler(r.Score);
    }
    return ret;
}

/* ********************************************************************************
 *                                L1 Scoring                                      *
 ******************************************************************************** */

/**
 * ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|)
 * for all i | v_i != 0 and w_i != 0
 * (Nister, 2006)
 * scaled_||v - w||_{L1} = 1 - 0.5 * ||v - w||_{L1}
 */
L1Scoring::L1Scoring() : GeneralScoring(true, // less is better
// Calculator
[](const WordValue& _V, const WordValue& _W) -> WordValue {
    return fabs(_V - _W) - fabs(_V) - fabs(_W);
},
// Scaler
[](WordValue& _Score) -> WordValue& {
    return _Score /= -2.;   // [0..1], greater is better
}) {}

/* ********************************************************************************
 *                                L2 Scoring                                      *
 ******************************************************************************** */

/**
 * ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )
 * for all i | v_i != 0 and w_i != 0 )
 * (Nister, 2006)
 */
L2Scoring::L2Scoring() : GeneralScoring(false, // greater is better
// Calculator
[](const WordValue& _V, const WordValue& _W) -> WordValue {
    return _V * _W;
},
// Scaler
[](WordValue& _Score) -> WordValue& {
    return _Score = _Score >= 1 ? 1.0 : 1.0 - sqrt(1.0 - _Score); // [0..1]
}) {}

/* ********************************************************************************
 *                             Chi Square Scoring                                 *
 ******************************************************************************** */

ChiSquareScoring::ChiSquareScoring() : GeneralScoring(false, // greater is better
// Calculator
[](const WordValue& _V, const WordValue& _W) -> WordValue {
    // (v-w)^2/(v+w) - v - w = -4 vw/(v+w)
    // we move the -4 out
    const auto tmp = _V + _W;
    return tmp != 0 ? _V * _W / tmp : 0;
},
// Scaler
[](WordValue& _Score) -> WordValue& {
    // this takes the -4 into account
    return _Score *= 2.; // [0..1]
}) {}

/* ********************************************************************************
 *                           KL Divergence Scoring                                *
 ******************************************************************************** */

KLScoring::KLScoring() : GeneralScoring(true, // less is better
// Calculator
[](const WordValue& _V, const WordValue& _W) -> WordValue {
    return _V * log(_V / _W);
},
default_scaler,
// After all callback
__CMP_CB {
    // sum rest of items of v
    for(; _It1 != _V1.end(); _It1++) {
        if(_It1 -> second == 0)continue;
        _Score += _It1 -> second * (log(_It1 -> second) - LOG_EPS);
    }
},
// Smaller callback
__CMP_CB {
    const auto& vi = _It1++ -> second;
    _Score += vi * (log(vi) - LOG_EPS);
},
// Inverted Callback
__IVT_CB {
    const auto& entryId = _Pair.first;
    auto& value = _Pair.second.first;
    for(const auto& word : _Vec) {
        const auto& vi = word.second;
        const auto& row = _InvertedFile[word.first];
        if(vi != 0 && std::find(row.begin(), row.end(), entryId) == row.end()) {
            value += vi * (log(vi) - GeneralScoring::LOG_EPS);
        }
    }
}) {}

/* ********************************************************************************
 *                           Bhattacharyya Scoring                                *
 ******************************************************************************** */

BhattacharyyaScoring::BhattacharyyaScoring() : GeneralScoring(true, // less is better
// Calculator
[](const WordValue& _V, const WordValue& _W) -> WordValue {
    return sqrt(_V * _W);
}) {}

/* ********************************************************************************
 *                            Dot Product Scoring                                 *
 ******************************************************************************** */

DotProductScoring::DotProductScoring() : GeneralScoring(true, // less is better
// Calculator
[](const WordValue& _V, const WordValue& _W) -> WordValue {
    return _V * _W;
}) {}

#undef __CMP_CB
#undef __IVT_CB

} // namespace TDBoW
