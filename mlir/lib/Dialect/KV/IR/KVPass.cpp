#include "mlir/Dialect/KV/IR/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {

struct KVOptimizerPass
    : public PassWrapper<KVOptimizerPass,
                         OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      KVOptimizerPass)

  StringRef getArgument() const final {
    return "kv-opt";
  }
  StringRef getDescription() const final {
    return "KV Optimization";
  }
  void runOnOperation() override {
    // MLIRContext *context = &getContext();
    getOperation()->dump();
  }
};

namespace kv {
void registerKVPasses() {
    PassRegistration<KVOptimizerPass>();

}
}
}