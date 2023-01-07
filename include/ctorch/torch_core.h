// Copyright (c) 2023 Lynn <lynnplus90@gmail.com>
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

#ifndef CTORCH_TORCH_CORE_H
#define CTORCH_TORCH_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>


// Interface visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef BUILDING_DLL
        #ifdef __GNUC__
            #define CTORCH_PUBLIC __attribute__((dllexport))
        #else  // __GNUC__
            #define CTORCH_PUBLIC __declspec(dllexport)
        #endif // __GNUC__
    #else // BUILDING_DLL
        #ifdef __GNUC__
            #define CTORCH_PUBLIC __attribute__((dllimport))
        #else
            #define CTORCH_PUBLIC __declspec(dllimport)
        #endif // __GNUC__
    #endif // BUILDING_DLL
#else // _WIN32 || __CYGWIN__
    #if __GNUC__ >= 4
        #define CTORCH_PUBLIC __attribute__((visibility("default")))
    #else
        #define CTORCH_PUBLIC
    #endif
#endif


typedef enum {
    TorchDeviceType_CPU = 0,
    TorchDeviceType_CUDA = 1, // CUDA.
    TorchDeviceType_MKLDNN = 2, // Reserved for explicit MKLDNN
    TorchDeviceType_OPENGL = 3, // OpenGL
    TorchDeviceType_OPENCL = 4, // OpenCL
    TorchDeviceType_IDEEP = 5, // IDEEP.
    TorchDeviceType_HIP = 6, // AMD HIP
    TorchDeviceType_FPGA = 7, // FPGA
    TorchDeviceType_ORT = 8, // ONNX Runtime / Microsoft
    TorchDeviceType_XLA = 9, // XLA / TPU
    TorchDeviceType_Vulkan = 10, // Vulkan
    TorchDeviceType_Metal = 11, // Metal
    TorchDeviceType_XPU = 12, // XPU
    TorchDeviceType_MPS = 13, // MPS
    TorchDeviceType_Meta = 14, // Meta (tensors with no data)
    TorchDeviceType_HPU = 15, // HPU / HABANA
    TorchDeviceType_VE = 16, // SX-Aurora / NEC
    TorchDeviceType_Lazy = 17, // Lazy Tensors
    TorchDeviceType_IPU = 18, // Graphcore IPU
} TorchDeviceType;

typedef enum {
    TorchScalarType_Byte = 0,
    TorchScalarType_Char,
    TorchScalarType_Short,
    TorchScalarType_Int,
    TorchScalarType_Long,
    TorchScalarType_Half,
    TorchScalarType_Float,
    TorchScalarType_Double = 7,
    TorchScalarType_Bool = 11
} TorchScalarType;

typedef struct {
    TorchDeviceType deviceType;
    int deviceIndex;
} TorchDevice;

typedef struct {
    void *data;
    int batchSize;
    int channels;
    int height;
    int width;
} TorchBlob;

typedef struct {
    int code;
    char *msg;//use free
    void *ctx;
} TorchStatus;

CTORCH_PUBLIC void torch_status_clear(TorchStatus *status);

typedef void *TorchModule;
typedef void *TorchIValue;

typedef void *TorchTensor;
typedef void *TorchTuple;


#ifdef __cplusplus
}
#endif

#endif //CTORCH_TORCH_CORE_H
