/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "device.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <sched.h>

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm);
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks);
int ncclTopoPathAllNVLink(struct ncclTopoSystem* system);
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm);

// Query topology
ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank);
ncclResult_t ncclTopoCheckP2p(struct ncclTopoSystem* system, int64_t id1, int64_t id2, int* p2p, int *read, int* intermediateRank);
ncclResult_t ncclTopoCheckMNNVL(struct ncclTopoSystem* system, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* ret);
ncclResult_t ncclTopoCheckGdr(struct ncclTopoSystem* topo, int64_t busId, int64_t netId, int read, int* useGdr);
#define MAX_XGMI_INTER_GPUS 4
ncclResult_t ncclTopoGetIntraNetDev(struct ncclTopoSystem* system, int rank, struct ncclTopoGraph* graph, int channelId, int type, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLinkType(struct ncclTopoSystem* system, int cudaDev1, int cudaDev2, bool* isXGMI, int maxInter=MAX_XGMI_INTER_GPUS, int nInter=0, int *inter=nullptr);
ncclResult_t ncclTopoNeedFlush(struct ncclTopoSystem* system, int64_t busId, int* flush);
ncclResult_t ncclTopoCheckNet(struct ncclTopoSystem* system, int64_t id1, int64_t id2, int* net);
int ncclPxnDisable(struct ncclComm* comm);
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks);

// Find CPU affinity
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity);

#define NCCL_TOPO_CPU_ARCH_X86 1
#define NCCL_TOPO_CPU_ARCH_POWER 2
#define NCCL_TOPO_CPU_ARCH_ARM 3
#define NCCL_TOPO_CPU_ARCH_MIXED 4
#define NCCL_TOPO_CPU_VENDOR_INTEL 1
#define NCCL_TOPO_CPU_VENDOR_AMD 2
#define NCCL_TOPO_CPU_VENDOR_ZHAOXIN 3
#define NCCL_TOPO_CPU_VENDOR_MIXED 4
#define NCCL_TOPO_CPU_TYPE_BDW 1
#define NCCL_TOPO_CPU_TYPE_SKL 2
#define NCCL_TOPO_CPU_TYPE_ZEN 3
#define NCCL_TOPO_CPU_TYPE_ROME 4
#define NCCL_TOPO_CPU_TYPE_YONGFENG 1
ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model);
ncclResult_t ncclTopoGetGpuCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetNvsCount(struct ncclTopoSystem* system, int* count);
ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev);
ncclResult_t ncclTopoGetLocalGpu(struct ncclTopoSystem* system, int64_t netId, int* gpuIndex);
ncclResult_t getLocalNetCountByBw(struct ncclTopoSystem* system, int gpu, int *count);

#define NCCL_TOPO_MAX_NODES 64

// Init search. Needs to be done before calling ncclTopoCompute
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct
struct ncclTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS*2];
  int nIntraChannels;
  int intraNets[MAXCHANNELS*NCCL_TOPO_MAX_NODES*2];
  char treeBase[NCCL_TOPO_MAX_NODES][NCCL_TOPO_MAX_NODES*4];
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs);

struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeToParent[MAXCHANNELS];
  int treeToChild0[MAXCHANNELS];
  int treeToChild1[MAXCHANNELS];
  int nvlsHeads[MAXCHANNELS];
  int nvlsHeadNum;
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns,
    struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs, struct ncclComm* parent, int nc);
ncclResult_t ncclTreeBasePostset(struct ncclComm* comm, struct ncclTopoGraph* treeGraph);

ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs);
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time, bool* backup=nullptr);

#endif
