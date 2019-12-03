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
   * File Name     : SpinLock.cpp
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-12-04 13:43:21
   * Last Modified : smallchimney
   * Modified Time : 2019-12-04 14:01:06
************************************************************************* */
#include <SpinLock.h>

namespace TDBoW {

SpinLock::SpinLock(std::atomic_bool& _Flag): m_bFlag(_Flag) {
    while(true) {
        bool excepted = false;
        if(m_bFlag.compare_exchange_strong(excepted, true)) {
            break;
        }
    }
}
SpinLock::~SpinLock() {
    m_bFlag = false;
}

}   // namespace TDBoW
