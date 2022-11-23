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

  virtual void runOnFunction(LLVM::LLVMFuncOp &F,
                             SymbolTableCollection *SymTab) {};
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
            runOnFunction(Function, &STC);
            for (auto &&Block : Function.getRegion()) {
              runOnBasicBlock(Block, &STC);
              for (auto &&Instruction : Block.getOperations()) {
                runOnInstruction(Instruction, &STC);
              }
            }
          }
        }
      }
    }
  }

};

struct KVOptimizerPass : LLVMFunctionPass<KVOptimizerPass> {
MLIR_MAGIC_INCANTATIONS(KVOptimizerPass, "kv-opt", "KV Optimization")
  std::set<Operation *> ToRemove;

  // Optimization DSL definition begin

  // Removes both, replaces First with new operation
  template<typename A, typename B, typename Func, typename... K>
  void ReplaceFirstWith(Func F, Operation *First, Operation *Second, K ...Keys) {
    if (auto X = dyn_cast<A>(First)) {
      if (auto Y = dyn_cast<B>(Second)) {
        if (((First->getOperand(Keys) == Second->getOperand(Keys)) && ...)) {
          ToRemove.insert(First);
          ToRemove.insert(Second);
          if (First->getNumResults() != 0) {
            First->replaceAllUsesWith(F(First, Second));
          }
        }
      }
    }
  }

  // Removes First
  template<typename A, typename B, typename... K>
  void RedundantFirst(Operation *First, Operation *Second, K ...Keys) {
    if (auto X = dyn_cast<A>(First)) {
      if (auto Y = dyn_cast<B>(Second)) {
        if (((First->getOperand(Keys) == Second->getOperand(Keys)) && ...)) {
          if (First->getNumResults() != 0) {
            First->replaceAllUsesWith(Second);
          }
          ToRemove.insert(First);
        }
      }
    }
  }

  template<typename T, typename ...O>
  std::function<Operation *(Operation *, Operation *)>
  Create(OpBuilder &Builder, bool First, O ...OpIds) {
    // The following line requires C++20 support.
    // Replace with std::tuple and std::apply if we need to compile with an older compiler
    if (First) {
      return [&, ... OpIds = std::forward<O>(OpIds)](auto A, auto B) {
        Builder.setInsertionPoint(A);
        return Builder.create<T>(A->getLoc(), A->getResultTypes(), A->getOperand(OpIds)  ...);
      };
    } else {
      return [&, ... OpIds = std::forward<O>(OpIds)](auto A, auto B) {
        Builder.setInsertionPoint(A);
        return Builder.create<T>(A->getLoc(), A->getResultTypes(), B->getOperand(OpIds)  ...);
      };
    }
  }

    // Optimization DSL end

  void runOnBasicBlock(mlir::Block &B, SymbolTableCollection *STC) override {
    std::vector<Operation *> KVOps;

    for (auto &&Instruction : B.getOperations()) {
      if (Instruction.getDialect() && Instruction.getDialect()->getNamespace() == "kv") {
        KVOps.push_back(&Instruction);
      }
    }

    OpBuilder Builder(B.getParentOp()->getContext());

    if (KVOps.size() == 1) {
      // Folds here
    }
    // A & B indicates:
    // Op B comes after Op A in the same basic block
    // They have the same key
    // There is no intervening write
    // TODO: Formulate a theory of safe peephole optimizations for this domain

    if (KVOps.size() >= 2) {
      for (size_t i = 0; i < KVOps.size() - 1 ; ++i) {
        // The following peepholes are written in the ad-hoc DSL defined by the
        // methods of this class.

        // get & del => getdel
        ReplaceFirstWith<kv::GetOp, kv::DelOp>(
          Create<kv::GetDelOp>(Builder, true , 0, 1), // <- the numbers are operand numbers
            KVOps[i], KVOps[i + 1], /*Keys=*/ 1);

        // get & set => getset
        ReplaceFirstWith<kv::GetOp, kv::SetOp>(
          Create<kv::GetSetOp>(Builder, false, 0, 1, 2),
            KVOps[i], KVOps[i + 1], /*Keys=*/ 1);

        // set_1 & set_2 => set_2
        RedundantFirst<kv::SetOp, kv::SetOp>(KVOps[i], KVOps[i + 1], 1);

        // get_1 & get_2 => get_2
        RedundantFirst<kv::GetOp, kv::GetOp>(KVOps[i], KVOps[i + 1], 1);
      }

    }

    for (auto Op : ToRemove) {
      B.getOperations().remove(Op);
      Op->destroy();
    }
    ToRemove.clear();
  }
};

struct LLVMToKVPass : LLVMFunctionPass<LLVMToKVPass> {
MLIR_MAGIC_INCANTATIONS(LLVMToKVPass, "llvm-to-kv", "LLVM to KV")

  void runOnFunction(LLVM::LLVMFuncOp &F,
                     SymbolTableCollection *SymTab) override {
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

  mlir::Value getGlobalString(mlir::Operation *Root, StringRef Str) {
    mlir::OpBuilder B(Root->getContext());
    static int StrNum = 0;
    auto Name = "s" + std::to_string(StrNum++);
    SmallString<16> NullTerminatedStr(Str);
    NullTerminatedStr.push_back('\0');
    B.setInsertionPoint(Root);
    return LLVM::createGlobalString(Root->getLoc(), B, Name, NullTerminatedStr, LLVM::linkage::Linkage::Internal);
  }

  template <typename ...Args>
  Value Call(mlir::OpBuilder B, mlir::Operation *Root, LLVM::LLVMFuncOp F, Args... args) {
    ValueRange CallArgs{args...};
    return B.create<LLVM::CallOp>(Root->getLoc(), F, CallArgs).getResult();
  }

  template <typename ...Args>
  LLVM::GEPOp GEP(mlir::OpBuilder B, mlir::Operation *Root, Value Ptr, Args... idx) {
    return B.create<LLVM::GEPOp>(Root->getLoc(), Root->getResult(0).getType(),
                                 Ptr, ArrayRef<LLVM::GEPArg>{idx...});
  }

  void ReplaceUses(mlir::OpBuilder &B, mlir::Operation *Root, mlir::Operation *New) {
    B.setInsertionPointAfter(New);
    // TODO: Is it possible to get the last dependent use without traversing the use tree?
    std::vector<mlir::Operation *> Stack{Root};
    std::set<mlir::Operation *> Visited;
    while (!Stack.empty()) {
      auto Cur = Stack.back();
      Stack.pop_back();
      if (Visited.find(Cur) != Visited.end()) {
        continue;
      }
      Visited.insert(Cur);
      B.setInsertionPointAfter(Cur);
      for (auto &&User : Cur->getUsers()) {
        Stack.push_back(User);
      }
    }

    Root->replaceAllUsesWith(New);
  }

  std::string FMT(mlir::Value V) {
    if (V.getType().isInteger(32)) {
      return "%i ";
    } else {
      return "%s ";
    }
  }

  std::string KWD(mlir::Operation *Op) {
    // TODO switch possible?
    if (isa<kv::GetOp>(Op)) {
      return "GET ";
    } else if (isa<kv::GetDelOp>(Op)) {
      return "GETDEL ";
    } else if (isa<kv::DelOp>(Op)) {
      return "DEL ";
    } else if (isa<kv::SetOp>(Op) || isa<kv::GetSetOp>(Op)) {
      return "SET ";
    } else {
      llvm_unreachable("Unimplemented translator");
    }
  }

  bool replaceInst(mlir::Operation *Op, LLVM::LLVMFuncOp RedisF,
                   LLVM::LLVMFuncOp FreeF, mlir::MLIRContext *Ctx) {
    mlir::OpBuilder B(Ctx);
    B.setInsertionPoint(Op);

    // TODO Create abstractions for the following logic

    if (isa<kv::GetOp>(Op) || isa<kv::GetDelOp>(Op)) {
      mlir::Value Str;
      Str = getGlobalString(Op, KWD(Op) + FMT(Op->getOperand(1)));
      Value BasePtr = Call(B, Op, RedisF, Op->getOperand(0), Str, Op->getOperand(1));

      auto Ptr = GEP(B, Op, BasePtr, 32);
      // TODO: The index 32 is for the string result, observe the value for int and branch accordingly.
      ReplaceUses(B, Op, Ptr);
      Call(B, Op, FreeF, BasePtr);
      return true;

    } else if (isa<kv::SetOp>(Op)) {
      auto Str = getGlobalString(Op, KWD(Op) + FMT(Op->getOperand(1)) + FMT(Op->getOperand(2)));
      auto BasePtr = Call(B, Op, RedisF, Op->getOperand(0), Str, Op->getOperand(1), Op->getOperand(2));
      Call(B, Op, FreeF, BasePtr);
      return true;

    } else if (isa<kv::DelOp>(Op)) {
      auto Str = getGlobalString(Op, KWD(Op) + FMT(Op->getOperand(1)));
      auto BasePtr = Call(B, Op, RedisF, Op->getOperand(0), Str, Op->getOperand(1));
      Call(B, Op, FreeF, BasePtr);
      return true;

    } else if (isa<kv::GetSetOp>(Op)) {
      mlir::Value Str;
      Str = getGlobalString(Op, KWD(Op) + FMT(Op->getOperand(1)) + FMT(Op->getOperand(2)) + " GET " );
      Value BasePtr = Call(B, Op, RedisF, Op->getOperand(0), Str, Op->getOperand(1), Op->getOperand(2));

      auto Ptr = GEP(B, Op, BasePtr, 32);
      // TODO: The index 32 is for the string result, observe the value for int and branch accordingly.
      ReplaceUses(B, Op, Ptr);
      Call(B, Op, FreeF, BasePtr);
      return true;

    }

    return false;
  }

  void runOnFunction(LLVM::LLVMFuncOp &F, SymbolTableCollection *STC) override {
    LLVM::LLVMFuncOp RedisF = STC->lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(F,
                              SymbolRefAttr::get(&getContext(), "redisCommand"));
    LLVM::LLVMFuncOp FreeF = STC->lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(F,
                              SymbolRefAttr::get(&getContext(), "freeReplyObject"));

    if (!RedisF || !FreeF) {
      llvm::errs() << "Symbol table lookup failed.\n";
      return;
    }

    for (auto &&B : F.getRegion()) {
      std::vector<Operation *> ToRemove;
      for (auto &&Instruction : B.getOperations()) {
        if (replaceInst(&Instruction, RedisF, FreeF, &getContext())) {
          ToRemove.push_back(&Instruction);
        }
      }
      for (auto Op : ToRemove) {
        B.getOperations().remove(Op);
        Op->destroy();
      }
    }
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