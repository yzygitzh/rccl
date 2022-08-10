#include "devcomm.h"

__global__ void mscclSynchronize(int workIndex, struct ncclDevComm* comm);
