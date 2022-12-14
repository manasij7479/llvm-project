//===- DXILTranslateMetadata.cpp - Pass to emit DXIL metadata ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
//===----------------------------------------------------------------------===//

#include "DirectX.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

static uint32_t ConstMDToUint32(const MDOperand &MDO) {
  ConstantInt *pConst = mdconst::extract<ConstantInt>(MDO);
  return (uint32_t)pConst->getZExtValue();
}

static ConstantAsMetadata *Uint32ToConstMD(unsigned v, LLVMContext &Ctx) {
  return ConstantAsMetadata::get(
      Constant::getIntegerValue(IntegerType::get(Ctx, 32), APInt(32, v)));
}

constexpr StringLiteral ValVerKey = "dx.valver";
constexpr unsigned DXILVersionNumFields = 2;

static void emitDXILValidatorVersion(Module &M, VersionTuple &ValidatorVer) {
  NamedMDNode *DXILValidatorVersionMD = M.getNamedMetadata(ValVerKey);

  // Allow re-writing the validator version, since this can be changed at
  // later points.
  if (DXILValidatorVersionMD)
    M.eraseNamedMetadata(DXILValidatorVersionMD);

  DXILValidatorVersionMD = M.getOrInsertNamedMetadata(ValVerKey);

  auto &Ctx = M.getContext();
  Metadata *MDVals[DXILVersionNumFields];
  MDVals[0] = Uint32ToConstMD(ValidatorVer.getMajor(), Ctx);
  MDVals[1] = Uint32ToConstMD(ValidatorVer.getMinor().value_or(0), Ctx);

  DXILValidatorVersionMD->addOperand(MDNode::get(Ctx, MDVals));
}

static VersionTuple loadDXILValidatorVersion(MDNode *ValVerMD) {
  if (ValVerMD->getNumOperands() != DXILVersionNumFields)
    return VersionTuple();

  unsigned Major = ConstMDToUint32(ValVerMD->getOperand(0));
  unsigned Minor = ConstMDToUint32(ValVerMD->getOperand(1));
  return VersionTuple(Major, Minor);
}

namespace {
class DXILTranslateMetadata : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  explicit DXILTranslateMetadata() : ModulePass(ID), ValidatorVer(1, 0) {}

  StringRef getPassName() const override { return "DXIL Metadata Emit"; }

  bool runOnModule(Module &M) override;

private:
  VersionTuple ValidatorVer;
};

} // namespace

bool DXILTranslateMetadata::runOnModule(Module &M) {
  if (NamedMDNode *ValVerMD = M.getNamedMetadata(ValVerKey)) {
    VersionTuple ValVer = loadDXILValidatorVersion(ValVerMD->getOperand(0));
    if (!ValVer.empty())
      ValidatorVer = ValVer;
  }
  emitDXILValidatorVersion(M, ValidatorVer);
  return false;
}

char DXILTranslateMetadata::ID = 0;

ModulePass *llvm::createDXILTranslateMetadataPass() {
  return new DXILTranslateMetadata();
}

INITIALIZE_PASS(DXILTranslateMetadata, "dxil-metadata-emit",
                "DXIL Metadata Emit", false, false)
