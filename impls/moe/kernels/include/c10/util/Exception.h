// SPDX-License-Identifier: Apache-2.0
// Stand-in for PyTorch's <c10/util/Exception.h>, which the pinned FlashInfer
// MoE headers include only for the host-side TORCH_CHECK / TORCH_WARN macros.
// The kernel translation units never execute host code; the compile-time
// probe aborts on a failed check exactly where upstream would throw.
#pragma once
#include <cstdio>
#include <cstdlib>

#define TORCH_CHECK(condition, ...)                                              \
  do {                                                                           \
    if (!(condition)) {                                                          \
      std::fprintf(stderr, "TORCH_CHECK failed: %s (%s:%d)\n", #condition,      \
                   __FILE__, __LINE__);                                          \
      std::abort();                                                              \
    }                                                                            \
  } while (0)
#define TORCH_WARN(...)                                                          \
  do {                                                                           \
    std::fprintf(stderr, "TORCH_WARN (%s:%d)\n", __FILE__, __LINE__);            \
    std::abort();                                                                \
  } while (0)
