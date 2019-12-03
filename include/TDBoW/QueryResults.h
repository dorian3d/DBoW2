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
 * File: QueryResults.h
 * Date: March, November 2011
 * Author: Dorian Galvez-Lopez
 * Description: structure to store results of database queries
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_QUERY_RESULTS__
#define __D_T_QUERY_RESULTS__

#include "BowVector.h"

namespace TDBoW {

/// Id of entries of the database
typedef unsigned int EntryId;

/// Single result of a query
class Result {
public:

    /// Entry id
    EntryId Id{};

    /// Score obtained
    double Score{};

    /**
     * Empty constructors
     */
    Result() = default;

    /**
     * Creates a result with the given data
     * @param _id entry id
     * @param _score score
     */
    Result(EntryId _id, WordValue _score): Id(_id), Score(_score) {}

    virtual ~Result() = default;

    /**
     * Compares the scores of two results
     * @return true iff this.score < r.score
     */
    bool operator<(const Result &r) const {
        return this->Score < r.Score;
    }

    /**
     * Compares the scores of two results
     * @return true iff this.score > r.score
     */
    bool operator>(const Result &r) const {
        return this->Score > r.Score;
    }

    /**
     * Compares the entry id of the result
     * @return true iff this.id == id
     */
    bool operator==(EntryId id) const {
        return this->Id == id;
    }

    /**
     * Compares the score of this entry with a given one
     * @param s score to compare with
     * @return true iff this score < s
     */
    bool operator<(double s) const {
        return this->Score < s;
    }

    /**
     * Compares the score of this entry with a given one
     * @param s score to compare with
     * @return true iff this score > s
     */
    bool operator>(double s) const {
        return this->Score > s;
    }

    /**
     * Compares the score of two results
     * @param a
     * @param b
     * @return true iff a.Score > b.Score
     */
    static bool gt(const Result &a, const Result &b)
    {
        return a.Score > b.Score;
    }

    /**
     * Compares the scores of two results
     * @return true iff a.Score > b.Score
     */
    static bool ge(const Result &a, const Result &b) {
        return a.Score > b.Score;
    }

    /**
     * Returns true iff a.Score >= b.Score
     * @param a
     * @param b
     * @return true iff a.Score >= b.Score
     */
    static bool geq(const Result &a, const Result &b) {
        return a.Score >= b.Score;
    }

    /**
     * Returns true iff a.Score >= s
     * @param a
     * @param s
     * @return true iff a.Score >= s
     */
    static bool geqv(const Result &a, double s) {
        return a.Score >= s;
    }


    /**
     * Returns true iff a.Id < b.Id
     * @param a
     * @param b
     * @return true iff a.Id < b.Id
     */
    static bool ltId(const Result &a, const Result &b) {
        return a.Id < b.Id;
    }

    /**
     * Prints a string version of the result
     * @param os ostream
     * @param ret Result to print
     */
    friend std::ostream & operator<<(std::ostream& os, const Result& ret );
};

/// Multiple results from a query
class QueryResults: public std::vector<Result> {
public:

    /**
     * Prints a string version of the results
     * @param os ostream
     * @param ret QueryResults to print
     */
    friend std::ostream & operator<<(std::ostream& os, const QueryResults& ret );

    /**
     * Saves a matlab file with the results
     * @param filename
     */
    void saveM(const std::string &filename) const;

};

// --------------------------------------------------------------------------

} // namespace TemplatedBoW
  
#endif

