//===- KVToLLVMIRTranslation.cpp - Translate LLVM dialect to LLVM IR ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/KV/KVToLLVMIRTranslation.h"
#include "mlir/Dialect/KV/IR/KV.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/MatrixBuilder.h"
#include "llvm/IR/Operator.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::getLLVMConstant;


static LogicalResult
convertOperationImpl(Operation &opInst, llvm::IRBuilderBase &builder,
                     LLVM::ModuleTranslation &moduleTranslation) {

  // I could not get this automation to work 
  // #include "mlir/Dialect/KV/KVConversions.inc"
  // Okay to just write C++ for this I think

  // if (auto op = dyn_cast<kv::GetOp>(opInst)) {
  //   moduleTranslation.mapValue(op.getRes()) = builder.CreateRetVoid();
  //   return success();
  // }

  // if (auto op = dyn_cast<kv::SetOp>(opInst)) {
  //   moduleTranslation.mapValue(op.getRes()) = builder.CreateRetVoid();
  //   return success();
  // }

  return failure();
}

namespace {

class KVDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    return convertOperationImpl(*op, builder, moduleTranslation);
  }
};
} // namespace

void mlir::registerKVDialectTranslation(DialectRegistry &registry) {
  registry.insert<kv::KVDialect>();
  registry.addExtension(+[](MLIRContext *ctx, kv::KVDialect *dialect) {
    dialect->addInterfaces<KVDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerKVDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerKVDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
