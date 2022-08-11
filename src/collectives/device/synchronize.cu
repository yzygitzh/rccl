#include "synchronize.h"
#include <stdio.h>

#define MSCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
  MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * MSCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)


__global__ void mscclSynchronize(int workIndex, struct ncclDevComm* comm) {
    int tid = threadIdx.x;
    volatile struct mscclFlag* mscclFlags = ((ncclDevCommAndChannels*)comm)->mscclInfo->flags;
    uint64_t curFlag = COMPUTE_FLAG(workIndex, 0, 0);
    mscclFlags[tid].flag = curFlag;
    uint64_t goalFlag = COMPUTE_FLAG(workIndex, 0, 100);
    while ((mscclFlags + tid)->flag < goalFlag){};
}
