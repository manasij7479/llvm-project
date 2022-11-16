//===- KVCanonicalization.td - KV dialect patterns -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KV_PATTERNS
#define KV_PATTERNS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/KV/IR/KVOps.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"

// def KVTestPattern :
//   Pat<(SetOp $ctx, $k, $v),
//       (GetOp $ctx, $k)>;

// set(k, C + get(k)) => incrby(k, C)
def KVRCUPattern1 :
  Pat<(SetOp $ctx, $k,
    (LLVM_AddOp $amount, (GetOp $ctx, $k))),
  (IncrByOp $ctx, $k, $amount)>;

// hset(k, C + hget(k)) => hincrby(k, C)
def KVRCUPattern2 :
  Pat<(HSetOp $ctx, $t, $k,
    (LLVM_AddOp $amount, (HGetOp $ctx, $t, $k))),
  (HIncrByOp $ctx, $t, $k, $amount)>;

#endif // KV_PATTERNS
