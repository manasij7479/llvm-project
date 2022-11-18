#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/KV/IR/KV.h"
#include "mlir/Dialect/KV/IR/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <unordered_set>

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
  void getDependentDialects(                          \
    DialectRegistry &registry) const override {       \
    registry.insert<mlir::kv::KVDialect>();           \
  }

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
  virtual void runOnBasicBlock(mlir::Block &B,
                               SymbolTableCollection *SymTab) {};
  virtual void runOnInstruction(mlir::Operation &Op,
                                SymbolTableCollection *SymTab) {};

  // TODO: Do we need any other hooks?
  void run(mlir::Operation *Op) {
    SymbolTableCollection STC;
    SymbolUserMap symbolUserMap(STC, Op);

    for (auto &&R : Op->getRegions()) {
      for (auto &&B : R.getBlocks()) {
        for (auto &&Global : B.getOperations()) {

          if (auto &&Function = llvm::dyn_cast<LLVM::LLVMFuncOp>(&Global)) {
            runOnFunction(Function);
            for (auto &&Block : Function.getRegion()) {
              runOnBasicBlock(Block, &STC);
              for (auto &&Instruction : Block.getOperations()) {
                runOnInstruction(Instruction, &STC);
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
    // Peephole optimizations here
    Peeps.push_back([](mlir::Block &B) {
    });
  }
};

struct LLVMToKVPass : LLVMFunctionPass<LLVMToKVPass> {
MLIR_MAGIC_INCANTATIONS(LLVMToKVPass, "llvm-to-kv", "LLVM to KV")

  void runOnFunction(LLVM::LLVMFuncOp &F) override {
    // llvm::errs() << F.getName() << "<-FUNCNAME\n";
  }

  // TODO: Figure out if there is a more idiomatic way to do this matching
  std::optional<std::string> getKVPrefix(mlir::Operation &Op,
                                             SymbolTableCollection *SymTab) {
    if (!isa_and_nonnull<LLVM::CallOp>(Op) || Op.getNumOperands() <= 1) {
      return {};
    }
    LLVM::CallOp &&Call = dyn_cast<LLVM::CallOp>(Op);
    if(Call.getCallee().has_value() && Call.getCallee().value() != "redisCommand") {
      return {};
    }
    if (!isa_and_nonnull<LLVM::GEPOp>(Call.getOperand(1).getDefiningOp())) {
      return {};
    }
    auto GEP = dyn_cast<LLVM::GEPOp>(Call.getOperand(1).getDefiningOp());
    if (!isa_and_nonnull<LLVM::AddressOfOp>(GEP.getBase().getDefiningOp())) {
      return {};
    }
    auto AddrOf = dyn_cast<LLVM::AddressOfOp>(GEP.getBase().getDefiningOp());
    LLVM::GlobalOp Global = AddrOf.getGlobal(*SymTab);

    if (!Global.getValue().has_value()) {
      return {};
    }

    // TODO: We might need a cleverer analysis of the format string.
    // This assumes redisCommand("CMD %fmts...", args...);
    // We could also potentially have redisCommand("CMD HARDCODED_ARGS")
    // Or an arbitrary mix of the above options.
    std::string FullStr;
    llvm::raw_string_ostream s(FullStr);
    s << Global.getValue().value();
    return FullStr.substr(1, FullStr.find(' ') - 1);
  }

  bool replaceInst(mlir::Operation &Op,
                       SymbolTableCollection *SymTab){
    // Op.dump();
    auto Prefix_p = getKVPrefix(Op, SymTab);
    if (!Prefix_p.has_value()) {
      return false; // Likely not a KV operation
    }
    auto Prefix = Prefix_p.value();


    std::unordered_set<std::string> Supported{"GET", "SET", "DEL"};

    if (Supported.find(Prefix) == Supported.end()) {
      llvm::errs() << "Unsupported KV OP: " << Prefix << "\n";
      return false;
    }

    MLIRContext *context = &getContext();
    mlir::OpBuilder B(context);

    Value Val;
    // Op.dump();

    B.setInsertionPoint(&Op);
    if (Prefix == "GET") {
      Val = B.createOrFold<kv::GetOp>(Op.getLoc(), Op.getResultTypes(),
                                        Op.getOperand(0), Op.getOperand(2));
    } else if (Prefix == "SET") {
      B.createOrFold<kv::SetOp>(Op.getLoc(),
                                Op.getOperand(0), Op.getOperand(2),
                                Op.getOperand(3));
    } else if (Prefix == "DEL") {
      B.createOrFold<kv::DelOp>(Op.getLoc(),
                                Op.getOperand(0), Op.getOperand(2));
    } else {
      llvm_unreachable("Unknown prefix.");
    }

    for (auto U : Op.getUsers()) {
      //U->dump();
      if (isa<LLVM::CallOp>(U)) {
        U->erase();
      }
      if (isa<LLVM::GEPOp>(U)) {
        if (Prefix == "GET") {
          // This path has to include all operations which return values.
          U->replaceAllUsesWith(Val.getDefiningOp());
          U->erase();
        } else {
          llvm_unreachable("Unhandled situation, printing out empty redis result.");
        }
      }
    }
    return true;
  }

  void runOnBasicBlock(mlir::Block &B, SymbolTableCollection *STC) override {
    std::vector<Operation *> ToRemove;
    for (auto &&Instruction : B.getOperations()) {
      if (replaceInst(Instruction, STC)) {
        ToRemove.push_back(&Instruction);
      }
    }
    for (auto Op : ToRemove) {
      B.getOperations().remove(Op);
      Op->destroy();
    }
  }
};

struct KVToLLVMPass : LLVMFunctionPass<KVToLLVMPass> {
MLIR_MAGIC_INCANTATIONS(KVToLLVMPass, "kv-to-llvm", "KV to LLVM")

  void runOnBasicBlock(mlir::Block &B, SymbolTableCollection *STC) override {
    // TODO: Lower KV Operations to LLVM.
    // Find redis function calls from symbol table
  }
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