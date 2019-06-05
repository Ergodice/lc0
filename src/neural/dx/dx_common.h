/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#pragma once

#include <d3d12.h>
#include <stdint.h>
#include "d3dx12.h"

namespace lczero {

uint16_t FP32toFP16(float f32);
float FP16toFP32(uint16_t f16);

namespace dx_backend {

void DxError(HRESULT status, const char* file, const int& line);
#define ReportDxErrors(status) DxError(status, __FILE__, __LINE__)

struct DXAlloc {
  ID3D12Resource* pResource;
  uint32_t offset;
  uint64_t gpuVA;
  D3D12_GPU_DESCRIPTOR_HANDLE descHandle;
};

typedef unsigned short dx_half;
typedef DXAlloc* dx_alloc_handle;
typedef ID3D12GraphicsCommandList5* dx_command_stream;

inline int DivUp(int a, int b) { return (a + b - 1) / b; }

}  // namespace dx_backend
}  // namespace lczero