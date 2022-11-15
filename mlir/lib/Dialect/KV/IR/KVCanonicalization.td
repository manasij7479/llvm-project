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

def KVTestPattern :
    Pat<(SetOp $ctx, $k, $v),
        (GetOp $ctx, $k)>;

#endif // KV_PATTERNS
