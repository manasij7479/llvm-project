add_mlir_translation_library(MLIRKVToLLVMIRTranslation
  KVToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRKVDialect 
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
