//-------------------------------------------------------------------------------
// Copyright (c) 2022 by Robert Bosch GmbH. All rights reserved.
// This file is property of Robert Bosch GmbH. Any unauthorized copy, use or
// distribution is an offensive act against international law and may be
// prosecuted under federal law. Its content is company confidential.
//-------------------------------------------------------------------------------

#ifndef XPER_COMMON_XINFER_SRC_NVIDIA_NV_UTILS_
#define XPER_COMMON_XINFER_SRC_NVIDIA_NV_UTILS_


#include <NvInfer.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>

namespace InferEngine
{
inline int64_t volume(const nvinfer1::Dims& d) { 
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<>()); }

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    default:
        throw std::runtime_error("Invalid DataType.");
    }
}

// std::string dataTypeToString_(nvinfer1::DataType type) {
//     switch (type) {
//         case nvinfer1::DataType::kFLOAT: return "kFLOAT";
//         case nvinfer1::DataType::kHALF: return "kHALF";
//         case nvinfer1::DataType::kINT8: return "kINT8";
//         case nvinfer1::DataType::kINT32: return "kINT32";
//         case nvinfer1::DataType::kBOOL: return "kBOOL";
//         default: return "Unknown DataType";
//     }
// }



#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                                                            \
    {                                                                                                                  \
        cudaError_t error_code = callstr;                                                                              \
        if (error_code != cudaSuccess)                                                                                 \
        {                                                                                                              \
            exit(0);                                                                                                   \
        }                                                                                                              \
    }
#endif

inline void* safeCudaMalloc(size_t memSize, bool useManaged = false)
{
    void* device_mem;

    if (useManaged)
    {
        CUDA_CHECK(cudaMallocManaged(&device_mem, memSize, cudaMemAttachHost));
        //CUDA_CHECK(cudaMallocManaged(&device_mem, memSize));
    }
    else
    {
        CUDA_CHECK(cudaMallocManaged(&device_mem, memSize));
        //CUDA_CHECK(cudaMalloc(&device_mem, memSize));
    }
    if (device_mem == nullptr)
    {
        // XPER_ERROR("Out of memory");
        exit(1);
    }
    return device_mem;
}

inline void* safeCudaMallocDevMem(size_t memSize)
{
    void* device_mem;

    CUDA_CHECK(cudaMalloc(&device_mem, memSize));

    if (device_mem == nullptr)
    {
        std::cout << "out of memory" << std::endl; 
        // XPER_ERROR("Out of memory");
        exit(1);
    }
    std::cout << "have malloced memory" << std::endl; 
    return device_mem;
}

inline void* safeCudaMallocHostMem(size_t memSize)
{
    void* device_mem;

    CUDA_CHECK(cudaMallocHost(&device_mem, memSize));

    if (device_mem == nullptr)
    {
        // XPER_ERROR("Out of memory");
        exit(1);
    }
    return device_mem;
}

inline void safeCudaFree(void* deviceMem)
{
    if (deviceMem != nullptr)
    {
        CUDA_CHECK(cudaFree(deviceMem));
        deviceMem = nullptr;
    }
}

inline void safeCudaFreeHost(void* deviceMem)
{
    if (deviceMem != nullptr)
    {
        CUDA_CHECK(cudaFree(deviceMem));
        deviceMem = nullptr;
    }
}

} // namespace InferEngine
#endif /* XPER_COMMON_XINFER_SRC_NVIDIA_NV_UTILS_ */
