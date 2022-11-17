#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/KV/IR/KV.h"
#include "mlir/Dialect/KV/IR/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"
#include <functional>

namespace mlir {

#define MLIR_MAGIC_INCANTATIONS(x, cmd, desc)         \
MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(x)       \
  StringRef getArgument() const override final {      \
    return cmd;                                       \
  }                                                   \
  StringRef getDescription() const override final {   \
    return desc;                                      \
  }                                                   \
  void runOnOperation() override {                    \
    LLVMFunctionPass<x>::run(getOperation());         \
  }                                                   \

template <typename CustomPass>
struct LLVMFunctionPass
  : public PassWrapper<CustomPass,
                       OperationPass<>> {

  virtual StringRef getArgument() const override {
    llvm_unreachable("Abstract LLVMFunctionPass getArgument");
  }
  virtual StringRef getDescription() const override {
    llvm_unreachable("Abstract LLVMFunctionPass getDescription");
  }

  virtual void runOnFunction(LLVM::LLVMFuncOp &F) {};
  virtual void runOnBasicBlock(mlir::Block &R) {};
  virtual void runOnInstruction(mlir::Operation &Op) {};

  // TODO: Do we need any other hooks?
  void run(mlir::Operation *Op) {
    for (auto &&R : Op->getRegions()) {
      for (auto &&B : R.getBlocks()) {
        for (auto &&Global : B.getOperations()) {
          if (auto &&Function = llvm::dyn_cast<LLVM::LLVMFuncOp>(&Global)) {
            runOnFunction(Function);
            for (auto &&Block : Function.getRegion()) {
              runOnBasicBlock(Block);
              for (auto &&Instruction : Block.getOperations()) {
                runOnInstruction(Instruction);
              }
              for (auto &&Peep : Peeps) {
                Peep(Block);
              }
            }
          }
        }
      }
    }
  }

  std::vector<std::function<void(mlir::Block &)>> Peeps;
};


struct KVOptimizerPass : LLVMFunctionPass<KVOptimizerPass> {
MLIR_MAGIC_INCANTATIONS(KVOptimizerPass, "kv-opt", "KV Optimization")

  KVOptimizerPass() {
    // Peephole optimiations here

    Peeps.push_back([](mlir::Block &B) {
    //   llvm::errs() << "BLOCKPOST!\n";
    });
  }

  void runOnFunction(LLVM::LLVMFuncOp &F) override {
    // llvm::errs() << F.getName() << "<-FUNCNAME\n";
  }

  void runOnBasicBlock(mlir::Block &R) override {
    // Pre processing for each block here
    // llvm::errs() << "BLOCK\n";

  }

  void runOnInstruction(mlir::Operation &Op) override {
//    llvm::errs() << Op.getName() << "\n";
  }
};

struct LLVMToKVPass : LLVMFunctionPass<LLVMToKVPass> {
MLIR_MAGIC_INCANTATIONS(LLVMToKVPass, "llvm-to-kv", "LLVM to KV")
};

struct KVToLLVMPass : LLVMFunctionPass<KVToLLVMPass> {
MLIR_MAGIC_INCANTATIONS(KVToLLVMPass, "kv-to-llvm", "KV to LLVM")
};

#undef MLIR_MAGIC_INCANTATIONS

namespace kv {
void registerKVPasses() {
  PassRegistration<LLVMToKVPass>();
  PassRegistration<KVOptimizerPass>();
  PassRegistration<KVToLLVMPass>();
}
}
}