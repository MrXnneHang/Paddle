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
  template <typename InT, typename bool = bool>                  \
  struct func_name {                                             \
    HOSTDEVICE bool operator()(const InT a, const InT b) const { \
      return static_cast<bool>(a op b);                          \
    }                                                            \
  };

COMPARE_FUNCTOR(LessThanFunctor, <)
COMPARE_FUNCTOR(LessEqualFunctor, <=)
COMPARE_FUNCTOR(GreaterThanFunctor, >)
COMPARE_FUNCTOR(GreaterEqualFunctor, >=)
#undef COMPARE_FUNCTOR

template <typename InT>
struct EqualFunctor {
  HOSTDEVICE bool operator()(const InT a, const InT b) const {
    if (std::is_floating_point<InT>::value) {
      if (isinf(static_cast<float>(a)) || isinf(static_cast<float>(b)))
        return static_cast<bool>(a == b);
      if (isnan(static_cast<float>(a)) || isnan(static_cast<float>(b)))
        return static_cast<bool>(false);
      return static_cast<bool>(fabs(static_cast<double>(a - b)) < 1e-8);
    } else {
      return static_cast<bool>(a == b);
    }
  }
};

template <typename InT>
struct EqualFunctor<phi::dtype::complex<InT>> {
  HOSTDEVICE bool operator()(const phi::dtype::complex<InT> a,
                             const phi::dtype::complex<InT> b) const {
    if (isnan(static_cast<InT>(a.real)) || isnan(static_cast<InT>(a.imag)) ||
        isnan(static_cast<InT>(b.real)) || isnan(static_cast<InT>(b.imag))) {
      return static_cast<bool>(false);
    }
    if (isinf(static_cast<InT>(a.real)) || isinf(static_cast<InT>(a.imag)) ||
        isinf(static_cast<InT>(b.real)) || isinf(static_cast<InT>(b.imag))) {
      return static_cast<bool>(a.real == b.real && a.imag == b.imag);
    }
    return static_cast<bool>(fabs(static_cast<double>(a.real - b.real)) <
                                 1e-8 &&
                             fabs(static_cast<double>(a.imag - b.imag)) < 1e-8);
  }
};

template <typename InT>
struct NotEqualFunctor {
  HOSTDEVICE bool operator()(const InT a, const InT b) const {
    return !EqualFunctor<InT, bool>()(a, b);
  }
};

}  // namespace funcs
}  // namespace phi
