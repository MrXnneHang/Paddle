// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <math.h>
#include "paddle/phi/common/complex.h"
namespace phi {
namespace funcs {

#define COMPARE_FUNCTOR(func_name, op)                           \
  template <typename InT, typename OutT = bool>                  \
  struct func_name {                                             \
    HOSTDEVICE OutT operator()(const InT a, const InT b) const { \
      return static_cast<OutT>(a op b);                          \
    }                                                            \
  };

COMPARE_FUNCTOR(LessThanFunctor, <)
COMPARE_FUNCTOR(LessEqualFunctor, <=)
COMPARE_FUNCTOR(GreaterThanFunctor, >)
COMPARE_FUNCTOR(GreaterEqualFunctor, >=)
#undef COMPARE_FUNCTOR

template <typename InT, typename OutT = bool>
struct EqualFunctor {
  HOSTDEVICE OutT operator()(const InT a, const InT b) const {
    printf("Equal泛化模板。\n");
    if (std::is_floating_point<InT>::value) {
      if (isinf(static_cast<float>(a)) || isinf(static_cast<float>(b)))
        return static_cast<OutT>(a != b);
      if (isnan(static_cast<float>(a)) || isnan(static_cast<float>(b)))
        return static_cast<OutT>(true);
      return static_cast<OutT>(fabs(static_cast<double>(a - b)) < 1e-8);
    } else {
      return static_cast<OutT>(a != b);
    }
  }
};

template <typename T>
struct EqualFunctor<phi::dtype::complex<T>> {
  HOSTDEVICE bool operator()(const phi::dtype::complex<T> a,
                             const phi::dtype::complex<T> b) const {
    printf("Equal偏特化模板\n");
    if (isnan(static_cast<T>(a.real)) || isnan(static_cast<T>(a.imag)) ||
        isnan(static_cast<T>(b.real)) || isnan(static_cast<T>(b.imag))) {
      printf("Equal偏特化:存在NAN\n");
      return static_cast<bool>(false);
    }
    if (isinf(static_cast<T>(a.real)) || isinf(static_cast<T>(a.imag)) ||
        isinf(static_cast<T>(b.real)) || isinf(static_cast<T>(b.imag))) {
      printf("Equal偏特化:存在INF\n");
      return static_cast<bool>(a.real == b.real && a.imag == b.imag);
    }
    printf("Equal偏特化:正常数.\n");
    return static_cast<bool>(fabs(static_cast<double>(a.real - b.real)) <
                                 1e-8 &&
                             fabs(static_cast<double>(a.imag - b.imag)) < 1e-8);
  }
};

template <typename InT, typename OutT = bool>
struct NotEqualFunctor {
  HOSTDEVICE bool operator()(const InT a, const InT b) const {
    printf("NotEqual泛化模板。\n");
    return !EqualFunctor<InT, OutT>()(a, b);
  }
};

template <typename T>
struct NotEqualFunctor<phi::dtype::complex<T>> {
  HOSTDEVICE bool operator()(const phi::dtype::complex<T> a,
                             const phi::dtype::complex<T> b) const {
    printf("NotEqual偏特化模板。\n");
    return !EqualFunctor<phi::dtype::complex<T>>()(a, b);
  }
};

}  // namespace funcs
}  // namespace phi
