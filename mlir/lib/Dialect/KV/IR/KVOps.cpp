//===- KVOps.cpp - MLIR Arithmetic dialect ops implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <utility>

#include "mlir/Dialect/KV/IR/KV.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::kv;


//===----------------------------------------------------------------------===//
// TableGen'd canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {
#include "KVCanonicalization.inc"
} // namespace


void kv::SetOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<KVTestPattern,
               KVRCUPattern>(
      context);
}


//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/KV/IR/KVOps.cpp.inc"



//===----------------------------------------------------------------------===//
// TableGen'd enum attribute definitions
//===----------------------------------------------------------------------===//

// #include "mlir/Dialect/Arithmetic/IR/ArithmeticOpsEnums.cpp.inc"
