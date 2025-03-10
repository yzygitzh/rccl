/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include "nccl.h"
#include "nccl_common.h"
#include "device.h"

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above
#define ALLTOALL_PIVOT_SLICESTEPS 2
#define ALLTOALL_PIVOT_CHUNKSTEPS 4

const char* ncclFuncToString(ncclFunc_t op);
const char* ncclDevRedOpToString(ncclDevRedOp_t op);
const char* ncclDatatypeToString(ncclDataType_t type);
const char* ncclAlgoToString(int algo);
const char* ncclProtoToString(int proto);

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
#if defined(RCCL_FLOAT8)
  case ncclFp8E4M3:
  case ncclFp8E5M2:
#endif
    return 1;
  case ncclFloat16:
#if defined(RCCL_BFLOAT16)
  case ncclBfloat16:
#endif
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

#include <sys/types.h>

#define NCCL_MODE_NORMAL 0
#define NCCL_MODE_OFFSET 1
#define NCCL_MODE_PTR    2
struct ncclConnFifo {
  int mode;
  int offset;
  ssize_t size;
  void* ptr;
};
#endif
