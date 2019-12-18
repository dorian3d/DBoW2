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
   * File Name     : IndexedFile.h
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-11-30 17:21:10
   * Last Modified : smallchimney
   * Modified Time : 2019-12-04 11:04:27
************************************************************************* */
#ifndef __ROCKAUTO_TDBOW_INDEXED_FILE_H__
#define __ROCKAUTO_TDBOW_INDEXED_FILE_H__

#include "FeatureVector.h"
#include "QueryResults.h"
#include <list>

namespace TDBoW {

/* Inverted file declaration */

/// Item of IFRow
struct IFPair {
    /// Entry id
    EntryId entry_id{};

    /// Word weight in this entry
    WordValue word_weight{};

    /**
     * Creates an empty pair
     */
    IFPair() = default;

    /** copy constructor */
    IFPair(const IFPair&) = default;

    /** move constructor */
    IFPair(IFPair&&) = default;

    /**
     * Creates an inverted file pair
     * @param eid entry id
     * @param wv word weight
     */
    IFPair(EntryId eid, WordValue wv): entry_id(eid), word_weight(wv) {}

    /**
     * Compares the entry ids
     * @param eid
     * @return true iff this entry id is the same as eid
     */
    bool operator==(const EntryId& _Eid) const { return entry_id == _Eid; }
};

/// Row of InvertedFile
typedef std::list<IFPair> IFRow;
// IFRows are sorted in ascending entry_id order

/// Inverted index
typedef std::vector<IFRow> InvertedFile;
// InvertedFile[word_id] --> inverted file of that word

/* Direct file declaration */

/// Direct index
typedef std::vector<FeatureVector> DirectFile;
// DirectFile[entry_id] --> [ directentry, ... ]

}

#endif //__ROCKAUTO_TDBOW_INDEXED_FILE_H__
