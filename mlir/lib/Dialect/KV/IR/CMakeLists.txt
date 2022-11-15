set(LLVM_TARGET_DEFINITIONS KVCanonicalization.td)
mlir_tablegen(KVCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRKVCanonicalizationIncGen)

add_mlir_dialect_library(MLIRKVDialect
  KVOps.cpp
  KVDialect.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/KV

  DEPENDS
  MLIRKVOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
)
