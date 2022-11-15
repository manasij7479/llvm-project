set(LLVM_TARGET_DEFINITIONS KVOps.td)
add_mlir_dialect(KVOps kv)
add_mlir_doc(KVOps KVOps Dialects/ -gen-dialect-doc)

# set(LLVM_TARGET_DEFINITIONS KVOps.td)
# mlir_tablegen(KVConversions.inc -gen-llvmir-conversions)
# # mlir_tablegen(LLVMConversionEnumsToLLVM.inc -gen-enum-to-llvmir-conversions)
# # mlir_tablegen(LLVMConversionEnumsFromLLVM.inc -gen-enum-from-llvmir-conversions)
# add_public_tablegen_target(MLIRKVConversionsIncGen)