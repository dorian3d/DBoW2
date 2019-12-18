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
 * File: TemplatedDatabase.hpp
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated database of images
 * License: see the LICENSE.txt file
 *
 */
 
#ifndef __ROCKAUTO_TDBOW_TEMPLATED_DATABASE_HPP__
#define __ROCKAUTO_TDBOW_TEMPLATED_DATABASE_HPP__

#include "IndexedFile.h"
#include "TemplatedVocabulary.hpp"
#include "QueryResults.h"

namespace TDBoW {

/// @param Descriptor class of descriptor
template <class Vocabulary>
/// Generic Database
class TemplatedDatabase {
public:
    // typedef
    typedef Vocabulary VocabularyType;
    typedef std::unique_ptr<VocabularyType> VocabularyPtr;
    typedef typename Vocabulary::LoadMode LoadMode;
    typedef typename Vocabulary::util util;
    TDBOW_DESCRIPTOR_DEF(util)

    // Constructor

    /**
     * @brief Creates an empty database without vocabulary
     * @param _UseDirectIdx   A direct index is used to store feature
     *                        indices
     * @param _DirectIdxLevel Levels to go up the vocabulary tree to
     *                        select the node id to store in the direct
     *                        index when adding images
     * @param _Vocab          Unique pointer to the vocabulary, support
     *                        inherited class.
     */
    explicit TemplatedDatabase(
            bool _UseDirectIdx = true, unsigned _DirectIdxLevel = 0,
            VocabularyPtr&& _Vocab = nullptr);

    /**
     * @brief Copy constructor is deleted.
     * @param _DB  Another database.
     */
    TemplatedDatabase(const TemplatedDatabase<Vocabulary>& _DB) = delete;

    /**
     * @brief Move constructor. Move the vocabulary too
     * @param db object to copy
     */
    TemplatedDatabase(const TemplatedDatabase<Vocabulary>&& _DB) noexcept;

    /**
     * @brief For now I decide to drop the save/load function for database,
     *        because it's a temperature data, for query history only.
     * @param _Filename  Absolute/relative file path, will search
     *                   relative from the "vocab" direction
     */
    explicit TemplatedDatabase(const std::string& _Filename,
            bool _UseDirectIdx = true, unsigned _DirectIdxLevel = 0);

    /**
     * @brief For now I decide to drop the save/load function for database,
     *        because it's a temperature data, for query history only.
     * @param _Filename  Absolute/relative file path, will search
     *                   relative from the "vocab" direction
     */
    explicit TemplatedDatabase(const char* _Filename,
            bool _UseDirectIdx = true, unsigned _DirectIdxLevel = 0);

    virtual ~TemplatedDatabase() = default;

    TemplatedDatabase& operator =(
            const TemplatedDatabase<Vocabulary>& _DB) = delete;
    TemplatedDatabase& operator==(
            const TemplatedDatabase<Vocabulary>& _DB) = delete;

    // Functional methods

    /**
     * @breif  Adds an entry to the database and returns its index
     * @author smallchimney
     * @param  _Features      Features of the new entry.
     * @param  _BowVec  (out) If given, the bow vector of these features
     *                        is returned.
     * @param  _FeatVec (out) If given, the vector of nodes and feature
     *                       indexes is returned.
     * @return                ID of the new added entry
     */
    EntryId add(const DescriptorArray& _Features,
                const BowVectorPtr& _BowVec = nullptr,
                const FeatureVectorPtr& _FeatVec = nullptr) noexcept(false);

    /**
     * @breif  Adds an entry to the database and returns its index
     * @author smallchimney
     * @param  _Features      Features of the new entry.
     * @param  _BowVec  (out) If given, the bow vector of these features
     *                        is returned.
     * @param  _FeatVec (out) If given, the vector of nodes and feature
     *                       indexes is returned.
     * @return                ID of the new added entry
     */
    EntryId add(const Descriptors& _Features,
                const BowVectorPtr& _BowVec = nullptr,
                const FeatureVectorPtr& _FeatVec = nullptr) noexcept(false);

    /**
     * @brief Add an entry to the database and returns its index
     * @param _BowVec  Bow vector
     * @param _FeatVec Feature vector to add the entry. Only necessary if using the
     *                 direct index
     * @return         ID of the new added entry
     */
    EntryId add(const BowVector& _BowVec,
                const FeatureVectorConstPtr& _FeatVec = nullptr) noexcept(false);

    /**
     * @brief  Queries the database with some features
     * @author smallchimney
     * @param  _Features      Query image's descriptor
     * @param  _BowVec        If set, the bow vector value will be filled.
     * @param  _MaxResults    Return size limit, {@code 0} stands no limit.
     * @param  _MinCommon     Entry common words selected limit, {@code 0} stands no limit.
     * @param  _MaxId         Entry ID selected limit, {@code 0} stands no limit.
     * @return                Selected and sorted entries results with scores.
     */
    QueryResults query(const DescriptorArray& _Features,
               const BowVectorPtr& _BowVec = nullptr, unsigned _MaxResults = 1,
               unsigned _MinCommon = 5, EntryId _MaxId = 0) const noexcept(false);

    /**
    * @brief  Queries the database with some features
    * @author smallchimney
    * @param  _Features      Query image's descriptor
    * @param  _BowVec        If set, the bow vector value will be filled.
    * @param  _MaxResults    Return size limit, {@code 0} stands no limit.
    * @param  _MinCommon     Entry common words selected limit, {@code 0} stands no limit.
    * @param  _MaxId         Entry ID selected limit, {@code 0} stands no limit.
    * @return                Selected and sorted entries results with scores.
    */
    QueryResults query(const Descriptors& _Features,
                       const BowVectorPtr& _BowVec = nullptr, unsigned _MaxResults = 1,
                       unsigned _MinCommon = 5, EntryId _MaxId = 0) const noexcept(false);

    /**
     * @breif  Queries the database with a vector
     * @author smallchimney
     * @param  _Vec           Vector to query
     * @param  _InvertedFile  Database.
     * @param  _MaxResults    Return size limit, {@code 0} stands no limit.
     * @param  _MinCommon     Entry common words selected limit, {@code 0} stands no limit.
     * @param  _MaxId         Entry ID selected limit, {@code 0} stands no limit.
     * @return                Selected and sorted entries results with scores.
     */
    QueryResults query(const BowVector& _Vec, unsigned _MaxResults = 1,
                       unsigned _MinCommon = 5, EntryId _MaxId = 0) const noexcept(false);

    /**
     * Empties the database
     */
    void clear() {
        m_aIFile.clear();
        m_aIFile.resize(getVocabSize());
        m_aDFile.clear();
        m_aDFile.shrink_to_fit();
        m_ulNumEntries = 0;
    }

    /**
     * Stops those words whose weight is below minWeight.
     * Words are stopped by setting their weight to 0. There are not returned
     * later when transforming image features into vectors.
     * Note that when using IDF or TF_IDF, the weight is the idf part, which
     * is equivalent to -log(f), where f is the frequency of the word
     * (f = Ni/N, Ni: number of training images where the word is present,
     * N: number of training images).
     * @return number of words stopped now
     */
    virtual size_t stopWords(WordValue _MinLimit) {
        return m_pVocab -> stopWords(_MinLimit);
    }

    // Get methods

    /**
     * @breif  Whether the database is ready.
     * @author smallchimney
     * @return {@code true} after the vocabulary is ready
     */
    virtual bool ready() const noexcept {
        return m_pVocab && m_pVocab -> ready();
    }

    /**
     * @breif  Returns the number of entries in the database
     * @return number of entries
     */
    virtual size_t size() const noexcept {
        return m_ulNumEntries;
    }

    /**
     * @breif  Returns the size of vocabulary (if has)
     * @return Size of the vocabulary, both unready vocabulary and
     *         non-vocabulary will return {@code 0}
     */
    size_t getVocabSize() const noexcept {
        return m_pVocab ? m_pVocab -> size() : 0;
    }

    /**
     * @breif  Returns whether the database is empty.
     * @return {@code true} before any valid query.
     */
    virtual bool empty() const noexcept {
        return m_ulNumEntries == 0;
    }

    /**
     * Returns the a feature vector associated with a database entry
     * @param id entry id (must be < size())
     * @return const reference to map of nodes and their associated features in
     *   the given entry
     */
    const FeatureVector& retrieveFeatures(const EntryId _ID) const noexcept(false) {
        if(!m_bUseDI) {
            throw std::runtime_error(TDBOW_LOG("Direct index file not built."));
        }
        // Confirm the parameters, be careful when change the structure of database.
        if(_ID < 0 || _ID >= size()) {
            throw std::runtime_error(TDBOW_LOG("Required entry not existed."));
        }
        return m_aDFile[_ID];
    }

    /**
     * @brief  Checks if the direct index is being used
     * @return {@code true} if using direct index
     */
    bool usingDirectIndex() const noexcept {
        return m_bUseDI;
    }

    /**
     * @brief  Returns the DI-levels when using direct index
     * @return Direct Index upper search levels in the vocabulary tree
     */
    unsigned getDirectIndexLevels() const noexcept {
        return m_uiDILevel;
    }

    /**
     * @brief  Get the const reference of the vocabulary
     * @author smallchimney
     * @return Vocabulary reference.
     */
    const Vocabulary& getVocabulary() const noexcept(false) {
        if(!ready()) {
            throw std::runtime_error(TDBOW_LOG("vocabulary is not ready yet."));
        }
        return *m_pVocab;
    }

    // Set methods

    /**
     * @brief  Change the database parameter, only success when empty.
     * @author smallchimney
     * @param  _UseDI    Whether build DI after the query.
     * @param  _DILevel  Levels to go up the vocabulary tree to
     *                   select the node id to store in the direct
     *                   index when adding images.
     * @return           This instance
     */
    inline TemplatedDatabase& updateParam(bool _UseDI,
            unsigned _DILevel = 0) noexcept(false);

    /**
     * @breif Sets the vocabulary to use and clears the content of
     *        the database.
     * @param _Vocab  Vocabulary to move
     * @return        This instance
     */
    inline TemplatedDatabase& setVocabulary(VocabularyPtr&& _Vocab) noexcept;

    /**
     * @brief Sets the vocabulary to use and the direct index parameters,
     *        and clears the content of the database
     * @param _Vocab   New Vocabulary to move
     * @param _UseDI   A direct index is used to store feature
     *                 indices.
     * @param _DILevel Levels to go up the vocabulary tree to select
     *                 the node id to store in the direct index when
     *                 adding images.
     * @return         This instance
     */
    inline TemplatedDatabase& setVocabulary(VocabularyPtr&& _Vocab,
            bool _UseDI, unsigned _DILevel = 0) noexcept;

    // I/O methods
    /**
     * @breif  Saves the vocabulary to a file
     * @author smallchimney
     * @param  _Filename  The relative/absolute filename, will be
     *                    regarded as relative to the `vocab` directory.
     * @param  _Binary    Whether save in binary file format.
     */
    void save(const std::string& _Filename,
              bool _Binary = true) const noexcept(false);

    /**
     * @breif  Load the vocabulary from a file
     * @author smallchimney
     * @param  _Filename  The relative/absolute filename, will be
     *                    regarded as relative to the `vocab` directory.
     * @param  _Mode      The mode of operation when loading.
     *                    {@code AUTO} will decided by filename.
     *                    {@code BINARY} will load as binary file.
     *                    {@code YAML} will load as yaml file.
     */
    void load(const std::string& _Filename,
            LoadMode _Mode = LoadMode::AUTO) noexcept(false);

protected:
    /// Associated vocabulary
    VocabularyPtr m_pVocab;

    /// Flag to use direct index
    bool m_bUseDI;

    /// Levels to go up the vocabulary tree to select nodes to store
    /// in the direct index
    unsigned m_uiDILevel;

    /// Inverted file (must have size() == |words|)
    InvertedFile m_aIFile;

    /// Direct file (resized for allocation)
    DirectFile m_aDFile;

    /// Number of valid entries in direct file
    size_t m_ulNumEntries;
  
};

/* ********************************************************************************
 *                        CONSTRUCTION && INITIALIZATION                          *
 ******************************************************************************** */

template <class Vocabulary>
TemplatedDatabase<Vocabulary>::TemplatedDatabase(
        const bool _UseDirectIdx, const unsigned _DirectIdxLevel,
        VocabularyPtr&& _Vocab)
        : m_pVocab(nullptr), m_bUseDI(_UseDirectIdx),
        m_uiDILevel(_DirectIdxLevel), m_ulNumEntries(0) {
    setVocabulary(std::forward<VocabularyPtr>(_Vocab));
}

template <class Vocabulary>
TemplatedDatabase<Vocabulary>::TemplatedDatabase(
        const TemplatedDatabase<Vocabulary>&& _DB) noexcept
        : m_pVocab(std::move(_DB.m_pVocab)), m_bUseDI(_DB.m_bUseDI),
        m_uiDILevel(_DB.m_uiDILevel), m_aIFile(std::move(_DB.m_aIFile)),
        m_aDFile(std::move(_DB.m_aDFile)), m_ulNumEntries(_DB.m_ulNumEntries) {
}

template <class Vocabulary>
TemplatedDatabase<Vocabulary>::TemplatedDatabase(const std::string& _Filename,
        const bool _UseDirectIdx, const unsigned _DirectIdxLevel)
        : m_pVocab(nullptr), m_bUseDI(_UseDirectIdx),
        m_uiDILevel(_DirectIdxLevel), m_ulNumEntries(0) {
    setVocabulary(VocabularyPtr(new Vocabulary(_Filename)));
}

template <class Vocabulary>
TemplatedDatabase<Vocabulary>::TemplatedDatabase(const char* _Filename,
        const bool _UseDirectIdx, const unsigned _DirectIdxLevel)
        : m_pVocab(nullptr), m_bUseDI(_UseDirectIdx),
          m_uiDILevel(_DirectIdxLevel), m_ulNumEntries(0) {
    setVocabulary(VocabularyPtr(new Vocabulary(_Filename)));
}


/* ********************************************************************************
 *                              FUNCTIONAL METHODS                                *
 ******************************************************************************** */

template <class Vocabulary>
EntryId TemplatedDatabase<Vocabulary>::add(
        const DescriptorArray& _Features,
        const BowVectorPtr& _BowVec, const FeatureVectorPtr& _FeatVec) noexcept(false) {
    if(!ready()) {
        throw std::runtime_error(TDBOW_LOG("vocabulary is not ready yet."));
    }
    if(_Features.empty()) {
        throw std::runtime_error("Try to add empty image.");
    }
    // BoW vector result is required, because it will be add into the database.
    BowVector ignored;
    auto& bowRet = _BowVec ? *_BowVec : ignored;
    // Feature vector result is optional according to the {@code m_bUseDI} flag.
    // We will still return the value when required, but not added.
    auto featPlaceholder = std::make_shared<FeatureVector>();
    auto featRet = _FeatVec != nullptr ? _FeatVec : (m_bUseDI ? featPlaceholder : nullptr);
    m_pVocab -> transform(_Features, bowRet, featRet, m_uiDILevel);
    return add(bowRet, featRet);
}

template <class Vocabulary>
EntryId TemplatedDatabase<Vocabulary>::add(
        const Descriptors& _Features,
        const BowVectorPtr& _BowVec, const FeatureVectorPtr& _FeatVec) noexcept(false) {
    if(!ready()) {
        throw std::runtime_error(TDBOW_LOG("vocabulary is not ready yet."));
    }
    if(_Features.rows() == 0) {
        throw std::runtime_error("Try to add empty image.");
    }
    // BoW vector result is required, because it will be add into the database.
    BowVector bowPlaceholder;
    auto& bowRet = _BowVec ? *_BowVec : bowPlaceholder;
    // Feature vector result is optional according to the {@code m_bUseDI} flag.
    // We will still return the value when required, but not added.
    auto featPlaceholder = std::make_shared<FeatureVector>();
    auto featRet = _FeatVec != nullptr ? _FeatVec : (m_bUseDI ? featPlaceholder : nullptr);
    m_pVocab -> transform(_Features, bowRet, featRet, m_uiDILevel);
    return add(bowRet, featRet);
}

template <class Vocabulary>
EntryId TemplatedDatabase<Vocabulary>::add(
        const BowVector& _BowVec, const FeatureVectorConstPtr& _FeatVec) noexcept(false) {
    if(!ready()) {
        throw std::runtime_error(TDBOW_LOG("vocabulary is not ready yet."));
    }
    auto entryID = static_cast<EntryId>(m_ulNumEntries++);
    // Update direct index file
    if(m_bUseDI) {
        if(_FeatVec == nullptr) {
            throw std::runtime_error(TDBOW_LOG(
                    "Current setting is \"USE_DI_ON\", "
                    "so the feature vector is required."));
        }
        m_aDFile.emplace_back(*_FeatVec);
    }
    // Update inverted index file
    for(const auto& word : _BowVec) {
        // Only valid word will be added, so speed up the calculation
        // NOTE: Be careful when stop word in a lower level later.
        if(m_pVocab -> getWordWeight(word.first)) {
            auto& row = m_aIFile[word.first];
            row.emplace_back(entryID, word.second);
        }
    }
    return entryID;
}

template <class Vocabulary>
QueryResults TemplatedDatabase<Vocabulary>::query(
        const DescriptorArray& _Features,
        const BowVectorPtr& _BowVec, const unsigned _MaxResults,
        const unsigned _MinCommon, EntryId _MaxId) const noexcept(false) {
    if(!ready()) {
        throw std::runtime_error(TDBOW_LOG("vocabulary is not ready yet."));
    }
    // BoW vector result is required, because it will be used when query
    BowVector vec;
    auto& bowRet = _BowVec ? *_BowVec : vec;
    m_pVocab -> transform(_Features, bowRet);
    return query(bowRet, _MaxResults, _MinCommon, _MaxId);
}

template <class Vocabulary>
QueryResults TemplatedDatabase<Vocabulary>::query(
        const Descriptors& _Features,
        const BowVectorPtr& _BowVec, const unsigned _MaxResults,
        const unsigned _MinCommon, EntryId _MaxId) const noexcept(false) {
    if(!ready()) {
        throw std::runtime_error(TDBOW_LOG("vocabulary is not ready yet."));
    }
    // BoW vector result is required, because it will be used when query
    BowVector vec;
    auto& bowRet = _BowVec ? *_BowVec : vec;
    m_pVocab -> transform(_Features, bowRet);
    return query(bowRet, _MaxResults, _MinCommon, _MaxId);
}

template <class Vocabulary>
QueryResults TemplatedDatabase<Vocabulary>::query(
        const BowVector& _Vec, const unsigned _MaxResults,
        const unsigned _MinCommon, EntryId _MaxId) const noexcept(false) {
    if(!ready()) {
        throw std::runtime_error(TDBOW_LOG("vocabulary is not ready yet."));
    }
    return m_pVocab -> getScoringObj().score(
            _Vec, m_aIFile, _MaxResults, _MinCommon, _MaxId);
}

/* ********************************************************************************
 *                               GET/SET METHODS                                  *
 ******************************************************************************** */

template <class Vocabulary>
TemplatedDatabase<Vocabulary>&
TemplatedDatabase<Vocabulary>::updateParam(const bool _UseDI,
        const unsigned _DILevel) noexcept(false) {
    if(m_bUseDI != _UseDI || m_uiDILevel != _DILevel) {
        if(!empty()) {
            throw std::runtime_error(TDBOW_LOG(
                    "The database is not empty, cannot change "
                    "the parameters."));
        }
        m_bUseDI = _UseDI;
        m_uiDILevel = _DILevel;
    }
    return *this;
}

template <class Vocabulary>
TemplatedDatabase<Vocabulary>&
TemplatedDatabase<Vocabulary>::setVocabulary(VocabularyPtr&& _Vocab) noexcept {
    if(_Vocab != m_pVocab) {
        m_pVocab = std::forward<VocabularyPtr>(_Vocab);
    }
    clear();
    return *this;
}

template <class Vocabulary>
TemplatedDatabase<Vocabulary>&
TemplatedDatabase<Vocabulary>::setVocabulary(VocabularyPtr&& _Vocab,
        const bool _UseDI, const unsigned _DILevel) noexcept {
    clear();
    updateParam(_UseDI, _DILevel);
    return setVocabulary(std::forward<VocabularyPtr>(_Vocab));
}

/* ********************************************************************************
 *                             INPUT/OUTPUT METHODS                               *
 ******************************************************************************** */

template <class Vocabulary>
void TemplatedDatabase<Vocabulary>::save(
        const std::string& _Filename, bool _Binary) const noexcept(false) {
    if(!m_pVocab) {
        throw std::runtime_error(TDBOW_LOG("No active vocabulary."));
    }
    m_pVocab -> save(_Filename, _Binary);
}

template <class Vocabulary>
void TemplatedDatabase<Vocabulary>::load(
        const std::string& _Filename, LoadMode _Mode) noexcept(false) {
    clear();
    m_pVocab = VocabularyPtr(new Vocabulary(_Filename, _Mode));
}

template <class Vocabulary>
std::ostream& operator<<(std::ostream& _Out, const TemplatedDatabase<Vocabulary>& _DB) {
    _Out << "Database: Entries = " << _DB.size() << ", "
            "Using direct index = " << (_DB.usingDirectIndex() ? "yes" : "no");
    if(_DB.usingDirectIndex()) {
        _Out << ", Direct index levels = " << _DB.getDirectIndexLevels();
    }
    return _Out << ". " << _DB.getVocabulary();
}

} // namespace TDBoW


#endif  // __ROCKAUTO_TDBOW_TEMPLATED_DATABASE_HPP__
