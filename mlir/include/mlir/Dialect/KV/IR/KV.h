//===- KV.h - KV dialect --------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_KV_IR_KV_H_
#define MLIR_DIALECT_KV_IR_KV_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"

//===----------------------------------------------------------------------===//
// KVDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/KV/IR/KVOpsDialect.h.inc"

// //===----------------------------------------------------------------------===//
// // KV Dialect Enum Attributes
// //===----------------------------------------------------------------------===//

// #include "mlir/Dialect/KV/IR/KVOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// KV Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/KV/IR/KVOps.h.inc"


#endif // MLIR_DIALECT_KV_IR_KV_H_
