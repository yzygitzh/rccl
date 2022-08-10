#!/bin/bash
#
# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

dir=$1

datatypes="i8 u8 i32 u32 i64 u64 f16 f32 f64"
if [ "$CUDA_MAJOR" -ge 11 ]
then
    datatypes+=" bf16"
fi

targets="GENOBJS := \\\\\n"

echo "${dir}/synchronize.o : synchronize.cu ${dir}/synchronize.dep"
echo "	@printf \"Compiling  %-35s > %s\\\\n\" synchronize.cu ${dir}/synchronize.o"
echo "	mkdir -p ${dir}"
echo "	\${NVCC} \${NVCUFLAGS} -dc synchronize.cu -o ${dir}/synchronize.o"
echo ""
targets="$targets\t${dir}/synchronize.o \\\\\n"

for base in sendrecv all_reduce all_gather broadcast reduce reduce_scatter all_to_all custom_collective; do
  opn=0
  for op in sum prod min max premulsum sumpostdiv; do
    dtn=0
    # Order must match that of the ncclDataType_t enum
    for dt in ${datatypes}; do
      echo "${dir}/${base}_${op}_${dt}.o : ${base}.cu ${dir}/${base}.dep"
      echo "	@printf \"Compiling  %-35s > %s\\\\n\" ${base}.cu ${dir}/${base}_${op}_${dt}.o"
      echo "	mkdir -p ${dir}"
      echo "	\${NVCC} -DNCCL_OP=${opn} -DNCCL_TYPE=${dtn} \${NVCUFLAGS} -dc ${base}.cu -o ${dir}/${base}_${op}_${dt}.o"
      echo ""
      targets="$targets\t${dir}/${base}_${op}_${dt}.o \\\\\n"
      dtn=$(($dtn + 1))
    done
    opn=$(($opn + 1))
  done
done
echo -e "$targets"
