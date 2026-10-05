// SPDX-License-Identifier: Apache-2.0
// Stand-in for FlashInfer's include/flashinfer/trtllm/common.h (PyTorch/ATen
// stream helpers), included by csrc/trtllm_batched_gemm_runner.cu. Only the
// compile-time host probe includes it; the runner uses none of its torch
// declarations.
#pragma once
#include <cuda.h>
#include <cuda_fp8.h>
#include <limits.h>
#include <stdint.h>

#include <cassert>

#include <c10/util/Exception.h>
