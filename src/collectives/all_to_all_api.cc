/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "graph/topo.h"
#include "devcomm.h"

ncclResult_t msccl2DAllToAll(const void *sendbuff, void *recvbuff, size_t sendcount,
                          ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream){
  int nGpus = comm->localRanks, nNodes = comm->nNodes;
  if (nGpus == 1 || nNodes == 1){
    WARN("number of local GPUs (%d) or number of nodes (%d) is 1.", nGpus, nNodes);
    return ncclInvalidUsage;
  }
  // 2D Hierarchical AlltoAll algorithm
  // phase 0. per-gpu (nGpus) stride copy
  CUDACHECK(strideMemcpyAsync(recvbuff, sendbuff, sendcount * ncclTypeSize(datatype), nGpus, nNodes, stream));
  // phase 1. intra-node alltoall
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGpus; g++)
  {
    NCCLCHECK(ncclSend(((char *)recvbuff) + g * nNodes * sendcount * ncclTypeSize(datatype), nNodes * sendcount, datatype, g + comm->node * nGpus, comm, stream));
    NCCLCHECK(ncclRecv(((char *)sendbuff) + g * nNodes * sendcount * ncclTypeSize(datatype), nNodes * sendcount, datatype, g + comm->node * nGpus, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  // phase 2. per-gpu (nNodes) stride copy
  CUDACHECK(strideMemcpyAsync(recvbuff, sendbuff, sendcount * ncclTypeSize(datatype), nNodes, nGpus, stream));
  // phase 3. inter-node alltoall
  NCCLCHECK(ncclGroupStart());
  for (int n = 0; n < nNodes; n++)
  {
    NCCLCHECK(ncclSend(((char *)recvbuff) + n * nGpus * sendcount * ncclTypeSize(datatype), nGpus * sendcount, datatype, n * nGpus + comm->hipDev, comm, stream));
    NCCLCHECK(ncclRecv(((char *)sendbuff) + n * nGpus * sendcount * ncclTypeSize(datatype), nGpus * sendcount, datatype, n * nGpus + comm->hipDev, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(hipMemcpyAsync(recvbuff, sendbuff, comm->nRanks * sendcount * ncclTypeSize(datatype), hipMemcpyDeviceToDevice, stream));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream);
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclComm_t comm, hipStream_t stream) {
  if (count == 0) return ncclSuccess;
  if (sendbuff == recvbuff){
    WARN("In-place alltoall is not possible currently.");
    return ncclInvalidUsage;
  }

  size_t allcount = count * comm->nRanks;
  size_t nbytes = allcount * ncclTypeSize(datatype);

  struct mscclHostCommInfo* mscclInfo = &comm->mscclHostComm;
  if (mscclInfo->nMscclRegistrations > 0)
  {
    for (int i = 0; i < mscclInfo->nMscclRegistrations; ++i)
    {
      struct mscclRegistration *reg = &mscclInfo->mscclRegistrations[i];
      if (reg->minBytes <= nbytes && (nbytes < reg->maxBytes || reg->maxBytes == -1))
      {
        struct mscclAlgorithm *mscclAlgo = &mscclInfo->mscclDevComm.mscclAlgos[reg->algoIndex];
        if ((mscclAlgo->isValid) && (mscclAlgo->collectiveType == ncclFuncAllToAll) && (comm->nRanks == mscclAlgo->ngpus) && ((allcount % mscclAlgo->nchunksPerLoop) == 0))
        {
          // if it was the 2D algorithm, select it first.
          if (!strcmp(mscclAlgo->name, "2D")) {
            return msccl2DAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
          } else {
            NVTX3_FUNC_RANGE_IN(nccl_domain);
            struct ncclInfo info = {ncclFuncAllToAll, "AllToAll",
                                    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
                                    MSCCL_CHUNKSTEPS, MSCCL_SLICESTEPS};
            info.algorithm = NCCL_ALGO_MSCCL;
            info.mscclInfo.mscclAlgoIndex = reg->algoIndex;
            info.protocol = reg->protocol;
            return ncclEnqueueCheck(&info);
          }
        }
      }
    }
  }
  else
  {
    for (int mscclAlgoIndex = 0; mscclAlgoIndex < mscclInfo->numberOfMSCCLAlgorithms; mscclAlgoIndex++)
    {
      struct mscclAlgorithm *mscclAlgo = &mscclInfo->mscclDevComm.mscclAlgos[mscclAlgoIndex];
      if ((mscclAlgo->isValid) && (mscclAlgo->collectiveType == ncclFuncAllToAll) && (comm->nRanks == mscclAlgo->ngpus)
        && ((allcount % mscclAlgo->nchunksPerLoop) == 0) && (nbytes >= mscclAlgo->minBytes) && (nbytes < mscclAlgo->maxBytes))
      {
        // if it was the 2D algorithm, select it first.
        if (!strcmp(mscclAlgo->name, "2D")) {
          return msccl2DAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
        } else {
          NVTX3_FUNC_RANGE_IN(nccl_domain);
          struct ncclInfo info = {ncclFuncAllToAll, "AllToAll",
                                  sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
                                  MSCCL_CHUNKSTEPS, MSCCL_SLICESTEPS};
          info.algorithm = NCCL_ALGO_MSCCL;
          info.mscclInfo.mscclAlgoIndex = mscclAlgoIndex;
          info.protocol = mscclAlgo->protocol;
          return ncclEnqueueCheck(&info);
        }
      }
    }
  }

  // Determine Pivot A2A support now that we know number of channels
  comm->topo->pivotA2AEnabled = comm->topo->pivotA2AEnabled && comm->nChannels >= comm->topo->pivotA2ANumBiRings * 2;
  if (comm->topo->pivotA2AEnabled) {
    struct ncclInfo info = { ncclFuncAllToAllPivot, "AllToAllPivot",
      sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
      ALLTOALL_PIVOT_CHUNKSTEPS, ALLTOALL_PIVOT_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    int nRanks;
    NCCLCHECK(ncclCommCount(comm, &nRanks));
    size_t rankOffset = count * ncclTypeSize(datatype);
    if (count == 0) return ncclSuccess;
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<nRanks; r++) {
      NCCLCHECK(ncclSend(((char*)sendbuff)+r*rankOffset, count, datatype, r, comm, stream));
      NCCLCHECK(ncclRecv(((char*)recvbuff)+r*rankOffset, count, datatype, r, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
  }
}
