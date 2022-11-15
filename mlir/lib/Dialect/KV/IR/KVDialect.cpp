//===- KVDialect.cpp - MLIR Arithmetic dialect implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/KV/IR/KV.h"
#include "mlir/IR/Builders.h"
// #include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::kv;

#include "mlir/Dialect/KV/IR/KVOpsDialect.cpp.inc"

void kv::KVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/KV/IR/KVOps.cpp.inc"
      >();
}

