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
   * File Name     : SpinLock.h
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-12-04 13:41:56
   * Last Modified : smallchimney
   * Modified Time : 2019-12-04 13:42:53
************************************************************************* */
#ifndef __ROCKAUTO_SPIN_LOCK_H__
#define __ROCKAUTO_SPIN_LOCK_H__

#include <atomic>

namespace TDBoW {

/**
 * @brief  Try to lock the resource, actually behave as a spin lock
 * @author smallchimney
 */
class SpinLock {
public:
    SpinLock() = delete;
    explicit SpinLock(std::atomic_bool& _Flag);
    virtual ~SpinLock();

private:
    std::atomic_bool& m_bFlag;
};

}

#endif //__ROCKAUTO_SPIN_LOCK_H__
