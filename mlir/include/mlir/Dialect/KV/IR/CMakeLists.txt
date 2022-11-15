set(LLVM_TARGET_DEFINITIONS KVOps.td)
add_mlir_dialect(KVOps kv)
add_mlir_doc(KVOps KVOps Dialects/ -gen-dialect-doc)
