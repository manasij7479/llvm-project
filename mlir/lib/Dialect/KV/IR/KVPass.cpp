#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/KV/IR/KV.h"
#include "mlir/Dialect/KV/IR/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <unordered_set>
#include <deque>

namespace mlir {

struct KVOptions {
  KVOptions() {
    SuggestMode = false;
  }
  llvm::cl::opt<bool> SuggestMode {
    "suggest",
    llvm::cl::desc("Suggest optimizations instead of"
                   "automatic code generation")};
};

static llvm::ManagedStatic<KVOptions> Options;

void registerKVPassOptions() {
  *Options;
}

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

mlir::Value getGlobalString(mlir::Operation *Root, StringRef Str) {
  mlir::OpBuilder B(Root->getContext());
  static int StrNum = 0;
  auto Name = "s" + std::to_string(StrNum++);
  SmallString<16> NullTerminatedStr(Str);
  NullTerminatedStr.push_back('\0');
  B.setInsertionPoint(Root);
  return LLVM::createGlobalString(Root->getLoc(), B, Name, NullTerminatedStr, LLVM::linkage::Linkage::Internal);
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
  // Optimization DSL helpers
  std::set<Operation *> ToRemove;
  void Remove(Operation *Op) {
    if (Options->SuggestMode) {
      Op->emitRemark() << "can be removed.\n";
    } else {
      ToRemove.insert(Op);
    }
  }

  void Replace(Operation *From, Operation *To) {
    if (Options->SuggestMode) {
      std::string data;
      llvm::raw_string_ostream str(data);
      To->print(str);
      From->emitRemark() << "can be replaced with " << data << "\n";
      ToRemove.insert(To);
    } else {
      From->replaceAllUsesWith(To);
    }
  }

  bool AllOperandsDominate(Operation *NewOp, Operation *Target) {
    DominanceInfo D(NewOp);
    for (auto Op : NewOp->getOperands()) {
      if (!D.dominates(Op, Target)) {
        return false;
      }
    }
    return true;
  }

  // Optimization DSL definition begin

  // Removes both, replaces First with new operation
  template<typename A, typename B, typename Func, typename... K>
  void ReplaceFirstWith(Func F, Operation *First, Operation *Second, K ...Keys) {
    if (auto X = dyn_cast<A>(First)) {
      if (auto Y = dyn_cast<B>(Second)) {
        if (((First->getOperand(Keys) == Second->getOperand(Keys)) && ...)) {
          if (First->getNumResults() != 0) {
            auto NewOp = F(First, Second);
            if (AllOperandsDominate(NewOp, First)) {
              Remove(First);
              Remove(Second);
              Replace(First, NewOp);
            } else {
              ToRemove.insert(NewOp);
            }
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
            Replace(First, Second);
          }
          Remove(First);
        }
      }
    }
  }

  void CombineGets(OpBuilder &B, Operation* firstGet, Operation* secondGet) {
    if(isa<kv::GetOp>(firstGet)&&isa<kv::GetOp>(secondGet)) {
      B.setInsertionPoint(firstGet);
      std::vector<Type> res{firstGet->getResultTypes().front(),secondGet->getResultTypes().front()};
      std::vector<Value> opers{firstGet->getOperand(0),firstGet->getOperand(1),secondGet->getOperand(1)};
      auto mgetOp=B.create<kv::MGetOp>(secondGet->getLoc(),res,opers);
      std::vector<Value> tmp_res{mgetOp.getResult(0)};
      firstGet->replaceAllUsesWith(tmp_res);
      tmp_res.clear();
      tmp_res.push_back(mgetOp.getResult(1));
      secondGet->replaceAllUsesWith(tmp_res);
      Remove(firstGet);
      Remove(secondGet);
    }
  }

  void CombineSets(OpBuilder &B, Operation* firstSet, Operation* secondSet) {
    if(isa<kv::SetOp>(firstSet)&&isa<kv::SetOp>(secondSet)) {
      B.setInsertionPoint(secondSet);
      std::vector<Type> res{firstSet->getResultTypes().front(),secondSet->getResultTypes().front()};
      std::vector<Value> opers{firstSet->getOperand(0),firstSet->getOperand(1),secondSet->getOperand(1)};
      auto msetOp=B.create<kv::MGetOp>(secondSet->getLoc(),res,opers);
      std::vector<Value> tmp_res{msetOp.getResult(0)};
      firstSet->replaceAllUsesWith(tmp_res);
      tmp_res.clear();
      tmp_res.push_back(msetOp.getResult(1));
      secondSet->replaceAllUsesWith(tmp_res);
      Remove(firstSet);
      Remove(secondSet);
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

  // Find and fix edge cases
  // For example: do we want dependencies from other basic blocks at all?
  // Do we want dependencies which do not dominate Root?
  // Is that situation even possible?
  void DFS(Operation *Root, std::vector<Operation*> &Results, size_t MaxDepth) {
    std::vector<std::pair<Operation *, size_t>> Stack{{Root, 0}};
    std::set<Operation *> Visited;
    while (!Stack.empty()) {
      auto Cur = Stack.back();
      Stack.pop_back();
      if (Cur.second >= MaxDepth) {
        continue;
      }

      // Cur.first->dump();
      // llvm::errs() << "\n";
      // if (isa<LLVM::CallOp>(Cur.first)) {
      //   continue;
      // }
      if (Visited.find(Cur.first) != Visited.end()){
        continue;
      } else {
        Visited.insert(Cur.first);
      }

      if (Cur.first != Root && Cur.first->getDialect()
          && Cur.first->getDialect()->getNamespace() == "kv"
          && Cur.first->hasOneUse()) {
        Results.push_back(Cur.first);
      }
      for (auto Op : Cur.first->getOperands()) {
        if (Op.getDefiningOp()) {
          Stack.push_back({Op.getDefiningOp(), Cur.second + 1});
        }
      }
    }
  }

  // Switch to a more efficient approach if this is a bottleneck
  // Maybe topological sort and then transitive dependencies.
  // i.e. if A is a dependency of B and B is a dependency of C, A is a
  // dependency of C. Just have to compute B before C.
  std::map<Operation *, std::vector<Operation*>>
  findDependencies(const std::vector<Operation *> &Ops) {
    const size_t MaxDepth = 10; // Search depth
    std::map<Operation *, std::vector<Operation*>> Results;
    for (const auto &Op : Ops) {
      std::vector<Operation *> Deps;
      DFS(Op, Deps, MaxDepth);
      Results[Op] = Deps;
    }
    return Results;
  }

  // TODO: Also match types
  Value Follow(Operation *Root, std::deque<size_t> Path) {
    if (Path.empty()) {
      return Root->getResult(0); // This might not work for mget and family
    }
    if (Root) {
      auto Cur = Path.front();
      Path.pop_front();
      if (Cur < Root->getNumOperands()) {
        auto Val = Root->getOperand(Cur);
        if (Path.empty()) {
          return Val;
        } else {
          return Follow(Val.getDefiningOp(), Path);
        }
      } else {
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }
  void Remove(Operation *Root, std::deque<size_t> Path) {
    if (Path.empty()) {
      return Remove(Root);
    }
    if (Root) {
      auto Cur = Path.front();
      Path.pop_front();
      if (Cur < Root->getNumOperands()) {
        auto Val = Root->getOperand(Cur);
        if (auto O = Val.getDefiningOp()) {
          Remove(O);
          return Remove(O, Path);
        }
      }
    }
  }

  Value Search(Operation *Root, std::deque<std::pair<std::string, size_t>> Path) {
    if (Path.empty()) {
      return Root->getResult(0); // This might not work for mget and family
    }
    if (Root) {
      auto Cur = Path.front();
      Path.pop_front();
      if (Cur.second < Root->getNumOperands()) {
        auto Val = Root->getOperand(Cur.second);
        if (Cur.first != "") {
          if (Val.getDefiningOp()) {
            if (Val.getDefiningOp()->getName().getStringRef() != Cur.first) {
              return nullptr;
            }
          } else {
            return nullptr;
          }
        }
        if (Path.empty()) {
          return Val;
        } else {
          return Search(Val.getDefiningOp(), Path);
        }
      } else {
        return nullptr;
      }
    } else {
      return nullptr;
    }
  }


  // Pattern matching DSL? Using existing one?
  Operation* TryDAGRewrites(Operation *Op, OpBuilder& Builder) {
    Builder.setInsertionPoint(Op);
    // set(k, get(k) + N) => incrby(n, N)
    if (isa<kv::SetOp>(*Op)) {
      auto Target = Search(Op, {{"llvm.add", 2}, {"llvm.call", 0},
                                {"llvm.load", 0}, {"llvm.bitcast", 0},
                                {"kv.get", 0}});
      if (Target) {
        auto IncrVal = Follow(Op, {2, 1});
        if (IncrVal) {
          Remove(Op);
          Remove(Op, {2, 0, 0, 0, 0});
          return Builder.create<kv::IncrByOp>(Op->getLoc(), Op->getResultTypes(),
            Op->getOperand(0), Op->getOperand(1), IncrVal);
        }
      }
    }
    return nullptr;
  }

  std::vector<Operation *> CommitAndGetKVOps(mlir::Block &B) {
    int Tries = 5;
    std::set<Operation *> Removed;

    while (Tries--) {
      for (auto Op : ToRemove) {
        if (Removed.find(Op) == Removed.end()) {
          if (Op->getUses().empty()) {
            B.getOperations().remove(Op);
            Op->destroy();
            Removed.insert(Op);
          }
        }
      }
    }

    ToRemove.clear();

    std::vector<Operation *> KVOps;
    for (auto &&Instruction : B.getOperations()) {
      if (Instruction.getDialect() && Instruction.getDialect()->getNamespace() == "kv") {
        KVOps.push_back(&Instruction);
      }
    }
    return KVOps;
  }

  void runOnBasicBlock(mlir::Block &B, SymbolTableCollection *STC) override {
    std::vector<Operation *> KVOps = CommitAndGetKVOps(B);
    OpBuilder Builder(B.getParentOp()->getContext());
    for (auto &&Op : KVOps) {
      if (auto Replacement = TryDAGRewrites(Op, Builder)) {
        Replace(Op, Replacement);
      }
    }

    KVOps = CommitAndGetKVOps(B);

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
          Create<kv::GetDelOp>(Builder, true , /*Operand indices=*/ 0, 1),
            KVOps[i], KVOps[i + 1], /*Keys=*/ 1);

        // get & set => getset
        ReplaceFirstWith<kv::GetOp, kv::SetOp>(
          Create<kv::GetSetOp>(Builder, false, 0, 1, 2),
            KVOps[i], KVOps[i + 1], /*Keys=*/ 1);


        // get1 & get2 => mget
        // get1 and get2 have to be independent
        CombineGets(Builder,KVOps[i], KVOps[i+1]);

        // set_1 & set_2 => set_2
        RedundantFirst<kv::SetOp, kv::SetOp>(KVOps[i], KVOps[i + 1], 1);

        // // get_1 & get_2 => get_1
        // RedundantSecond<kv::GetOp, kv::GetOp>(KVOps[i], KVOps[i + 1], 1);
      }

    }
    CommitAndGetKVOps(B);
  }
};

struct LLVMToKVPass : LLVMFunctionPass<LLVMToKVPass> {
MLIR_MAGIC_INCANTATIONS(LLVMToKVPass, "llvm-to-kv", "LLVM to KV")

  void runOnFunction(LLVM::LLVMFuncOp &F,
                     SymbolTableCollection *SymTab) override {
    // llvm::errs() << F.getName() << "<-FUNCNAME\n";
  }

  std::vector<std::string> split(std::string str) {
    std::vector<std::string> Results;
    std::istringstream in(str);
    std::string word;
    while (in >> word) {
      Results.push_back(word);
    }
    return Results;
  }
  // TODO: Figure out if there is a more idiomatic way to do this matching
  std::vector<std::string> getKVPrefix(mlir::Operation &Op,
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

    std::vector<std::string> Results;
    for (auto &&Word : split(FullStr.substr(1, FullStr.length() - 2))) {
      Results.push_back(Word);
    }

    return Results;
  }

  bool replaceInst(mlir::Operation &Op,
                       SymbolTableCollection *SymTab){
    // Op.dump();
    auto Args = getKVPrefix(Op, SymTab);
    if (Args.empty()) {
      return false; // Likely not a KV operation
    }

    auto Prefix = Args[0];

    std::vector<Value> Operands;
    Operands.push_back(Op.getOperand(0));
    size_t NextOpnIdx = 0;
    for (size_t i = 1; i < Args.size(); ++i) {
      if (Args[i][0] == '%') {
        Operands.push_back(Op.getOperand(2 + NextOpnIdx++));
      } else {
        Operands.push_back(getGlobalString(&Op, Args[i]));
      }
    }

    auto OPN = [&](size_t i) { return Operands[i];};

    std::unordered_set<std::string> Supported{"GET", "SET", "DEL", "HGET", "HSET"};

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
                                        OPN(0), OPN(1));
    } else if (Prefix == "SET") {
      B.createOrFold<kv::SetOp>(Op.getLoc(), OPN(0), OPN(1), OPN(2));
    } else if (Prefix == "HSET") {
      B.createOrFold<kv::HSetOp>(Op.getLoc(), OPN(0), OPN(1), OPN(2), OPN(3));
    } else if (Prefix == "DEL") {
      B.createOrFold<kv::DelOp>(Op.getLoc(), OPN(0), OPN(1));
    } else if (Prefix == "HGET") {
      Val = B.createOrFold<kv::HGetOp>(Op.getLoc(), Op.getResultTypes(),
                                       OPN(0), OPN(1), OPN(2));
    } else {
      llvm_unreachable("Unknown prefix.");
    }

    for (auto U : Op.getUsers()) {
      //U->dump();
      if (isa<LLVM::CallOp>(U)) {
        U->erase();
      }
      if (isa<LLVM::GEPOp>(U)) {
        if (Prefix == "GET" || Prefix == "HGET") {
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

  template<typename ValuesT>
  void ReplaceUses(mlir::OpBuilder &B, mlir::Operation *Root, ValuesT&& New,mlir::Operation* insertPoint) {
    B.setInsertionPointAfter(insertPoint);
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
    } else if (isa<kv::HGetOp>(Op)) {
      return "HGET ";
    } else if (isa<kv::GetDelOp>(Op)) {
      return "GETDEL ";
    } else if (isa<kv::DelOp>(Op)) {
      return "DEL ";
    } else if (isa<kv::SetOp>(Op) || isa<kv::GetSetOp>(Op)) {
      return "SET ";
    } else if(isa<kv::MGetOp>(Op)){
        return "MGET ";
    }else if (isa<kv::HSetOp>(Op)) {
      return "HSET ";
    } else if (isa<kv::IncrByOp>(Op)) {
      return "INCRBY ";
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
      ReplaceUses(B, Op, Ptr,Ptr.getOperation());
      Call(B, Op, FreeF, BasePtr);
      return true;

    }else if(isa<kv::MGetOp>(Op)){
      auto Str= getGlobalString(Op,KWD(Op) + FMT(Op->getOperand(1))+FMT(Op->getOperand(2)));
      auto BasePtr=Call(B,Op,RedisF,Op->getOperand(0),Str,Op->getOperand(1),Op->getOperand(2));
      static threeStarsIntPtr=LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(B.getI8Type())));
      auto CastedPtr=LLVM::BitcastOp::build(B,BasePtr.getLoc(),threeStarsIntPtr,BasePtr);
      std::vector<LLVM::GEPOp> elePtrs;
      //getBitcastType(Op->getBlock());
      for(int i=0;i<2;++i) {
        //need more think about the offsets.
        auto resStr=GEP(B, Op, CastedPtr, 56,0,i);
        elePtrs.push_back(resStr);

      }
      mlir::Operation* latest=elePtrs[0].getOperation();
      for(int i=1;i<2;++i){
        if(latest->isBeforeInBlock(elePtrs[i].getOperation())){
          latest=elePtrs[i].getOperation();
        }
      }
      ReplaceUses(B,Op,elePtrs,latest);
      Call(B,Op,FreeF,BasePtr);
      return true;
    }
    else if (isa<kv::SetOp>(Op)) {
      auto Str = getGlobalString(Op, KWD(Op) + FMT(Op->getOperand(1)) + FMT(Op->getOperand(2)));
      auto BasePtr = Call(B, Op, RedisF, Op->getOperand(0), Str, Op->getOperand(1), Op->getOperand(2));
      Call(B, Op, FreeF, BasePtr);
      return true;

    } else if (isa<kv::IncrByOp>(Op)) {
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
      ReplaceUses(B, Op, Ptr, Ptr.getOperation());
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
  registerKVPassOptions();
  PassRegistration<LLVMToKVPass>();
  PassRegistration<KVOptimizerPass>();
  PassRegistration<KVToLLVMPass>();
}
}
}
