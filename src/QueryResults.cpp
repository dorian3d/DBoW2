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
 * File: QueryResults.cpp
 * Date: March, November 2011
 * Author: Dorian Galvez-Lopez
 * Description: structure to store results of database queries
 * License: see the LICENSE.txt file
 *
 */

#include <fstream>
#include <QueryResults.h>

using std::ostream;
using std::endl;

namespace TDBoW {

// ---------------------------------------------------------------------------

ostream& operator<<(ostream& _Out, const Result& _Ret) {
    _Out << "<EntryId: " << _Ret.Id << ", Score: " << _Ret.Score << ">";
    return _Out;
}

// ---------------------------------------------------------------------------

ostream& operator<<(ostream& _Out, const QueryResults& _Ret) {
    _Out << _Ret.size() << " results:";
    if(_Ret.empty())return _Out << endl;
    for(const auto& ret : _Ret) {
        _Out << endl << ret;
    }
    return _Out;
}

// ---------------------------------------------------------------------------

void QueryResults::saveM(const std::string& filename) const {
    std::fstream f(filename.c_str(), std::ios::out);
    for(const auto& ret : *this) {
        f << ret.Id << " " << ret.Score << endl;
    }
    f.close();
}

// ---------------------------------------------------------------------------

} // namespace DBoW2

