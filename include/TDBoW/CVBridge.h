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
/*! \mainpage TDBoW Library
 *
 * TDBoW library for C++:
 * 3D bag-of-word image database for pointcloud retrieval.
 *
 * Written by Yichong Xu,
 * Lanzhou University, China
 *
 * Forked from Dorian Galvez-Lopez,
 * University of Zaragoza
 *
 * \section requirements Requirements
 * This library requires the Eigen, yaml-cpp and Boost-filesystem libraries.
 *
 * \section citation Citation
 * If you use this software in academic works, please cite:
 <pre>
   @@ARTICLE{
   todo: Still work-in-progress
   }
  }
 </pre>
 *
 */
/* *************************************************************************
   * File Name     : CVBridge.h
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-12-01 21:22:57
   * Last Modified : smallchimney
   * Modified Time : 2019-12-13 17:39:28
************************************************************************* */
#ifndef __ROCKAUTO_TDBOW_CV_BRIDGE_H__
#define __ROCKAUTO_TDBOW_CV_BRIDGE_H__

#include "TDBoW.h"
#include <opencv2/core/eigen.hpp>   // this must be included after <eigen3/Eigen/Core>

namespace TDBoW {
    /** @brief ORB Vocabulary */
    typedef TemplatedVocabulary<uint8_t, 256 / 8> Orb256Vocabulary;

    /** @brief ORB Database */
    typedef TemplatedDatabase<Orb256Vocabulary> Orb256Database;

    /** @brief BRIEF Vocabulary */
    typedef TemplatedVocabulary<uint8_t, 256 / 8> Brief256Vocabulary;

    /** @brief BRIEF Database */
    typedef TemplatedDatabase<Brief256Vocabulary> Brief256Database;
}

#endif //__ROCKAUTO_TDBOW_CV_BRIDGE_H__
