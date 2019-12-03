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
   * File Name     : traits.h
   * Author        : smallchimney
   * Author Email  : smallchimney@foxmail.com
   * Created Time  : 2019-11-22 22:22:49
   * Last Modified : smallchimney
   * Modified Time : 2019-12-02 17:13:40
************************************************************************* */
#ifndef __ROCKAUTO_TRAITS_H__
#define __ROCKAUTO_TRAITS_H__

#include <string>
#include <memory>

namespace TDBoW {
namespace traits {

/* ********************************************************************************
 *                                BASIC TRAITS                                    *
 ******************************************************************************** */

template <typename T>
class basic_traits {
public:
    typedef std::shared_ptr<T> Ptr;
    typedef std::shared_ptr<T const> ConstPtr;
};

/* ********************************************************************************
 *                                 TYPE TRAITS                                    *
 ******************************************************************************** */

namespace type_trait {
    typedef int type_id;
}

/**
 * @brief  Since we cannot get standard type_info for TScalar on
 *         different runtime context, we can force use custom standard.
 * @author smallchimney
 * @tparam TScalar  the data scalar defined
 */
template <typename TScalar>
class type_traits: public basic_traits<TScalar> {
public:
    typedef type_trait::type_id type_id;
    static std::string name() { return "unknown"; }
    static constexpr type_id id() { return -1; }
};

#define __TYPE_TRAITS(TYPE, BRIEF, VALUE)\
template <>\
class type_traits<TYPE>: public basic_traits<TYPE> {\
public:\
    typedef type_trait::type_id type_id;\
    static std::string name() { return BRIEF; }\
    static constexpr type_id id() { return VALUE; }\
}

__TYPE_TRAITS(uint8_t , "unsigned char(8 bits)" , 0);
__TYPE_TRAITS(uint16_t, "unsigned char(16 bits)", 1);
__TYPE_TRAITS(uint32_t, "unsigned char(32 bits)", 2);
__TYPE_TRAITS(uint64_t, "unsigned char(64 bits)", 3);
__TYPE_TRAITS(int8_t  , "signed char(8 bits)"   , 4);
__TYPE_TRAITS(int16_t , "signed char(16 bits)"  , 5);
__TYPE_TRAITS(int32_t , "signed char(16 bits)"  , 6);
__TYPE_TRAITS(int64_t , "signed char(16 bits)"  , 7);
// {@code long long int} will be discarded, because in some platform
// it might conflict with {@code int64_t}
__TYPE_TRAITS(bool    , "boolean"               , 8);
__TYPE_TRAITS(float   , "float"                 , 9);
__TYPE_TRAITS(double  , "double"                , 10);
// only fixed size type is support,
// so avoid use type like std::string or std::vector

#undef __TYPE_TRAITS

}}  // namespace TDBow::traits

#endif //__ROCKAUTO_TRAITS_H__
