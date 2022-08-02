/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#if 1
#include "msccl_interpreter.h"

template<int ALGO, typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, ALGO, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS>;
    runInterpreter<T, RedOp, Proto>(args, 1);
  }
};

template<int ALGO, typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, ALGO, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runInterpreter<T, RedOp, ProtoLL128>(args, 1);
  }
};

template<int ALGO, typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, ALGO, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runInterpreter<T, RedOp, ProtoLL>(args, 1);
  }
};
#endif
