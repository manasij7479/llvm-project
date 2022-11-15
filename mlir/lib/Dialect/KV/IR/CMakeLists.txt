# set(LLVM_TARGET_DEFINITIONS ArithmeticCanonicalization.td)
# mlir_tablegen(ArithmeticCanonicalization.inc -gen-rewriters)
# add_public_tablegen_target(MLIRArithmeticCanonicalizationIncGen)

add_mlir_dialect_library(MLIRKVDialect
  KVOps.cpp
  KVDialect.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/KV

#   DEPENDS
#   MLIRArithmeticOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
)
