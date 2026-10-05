// SPDX-License-Identifier: Apache-2.0
// Single-kernel translation unit: the previous package's pack_scales glue
// kernel (impls/templates/nvfp4.cu), verbatim. It copies logical K-major
// E4M3 block scales [L][rows][cols] into CUTLASS's Sm1xxBlkScaledConfig
// layout: per batch, ceil(rows/128) x ceil(cols/4) atoms of 512 bytes, each
// a 128x4 tile stored as [row % 32][row / 32 % 4][col % 4]; padding is zero.
#include <cstdint>

__global__ void pack_scales(const uint8_t* input, uint8_t* output, int rows, int cols, int batches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int padded_rows = (rows + 127) / 128 * 128, padded_cols = (cols + 3) / 4 * 4;
  int size = padded_rows * padded_cols;
  if (i >= batches * size) return;
  int l = i / size, offset = i % size;
  int tile = offset / 512, inside = offset % 512;
  int row = (tile / (padded_cols / 4)) * 128 + (inside / 16) + ((inside % 16) / 4) * 32;
  int col = (tile % (padded_cols / 4)) * 4 + inside % 4;
  output[i] = row < rows && col < cols ? input[(l * rows + row) * cols + col] : 0;
}
