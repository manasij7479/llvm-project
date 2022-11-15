//===- KVOps.td - Base defs for arith dialect ------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef KV_BASE
#define KV_BASE

// include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"


def KV_Dialect : Dialect {
  let name = "kv";
  let cppNamespace = "::mlir::kv";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let description = [{TODO Describe}];

  // let hasConstantMaterializer = 1;
}

class KV_Op<string mnemonic, list<Trait> traits = []> :
    Op<KV_Dialect, mnemonic, traits>;

def NoOp : KV_Op<"nop", [NoSideEffect]> {
  let summary = "nop";

  let arguments = (ins LLVM_AnyPointer:$ptr);
  let results = (outs LLVM_AnyPointer);

  let assemblyFormat = [{
    `(` $ptr `:` type($ptr) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a TransposeOp with from the input operand.
  // let builders = [
  //   OpBuilder<(ins "Value":$input)>
  // ];

  // Indicate that additional verification for this operation is necessary.
  // let hasVerifier = 1;
}


// Request Ops, each returns a pointer to a reply object

def GetOp : KV_Op<"get", [NoSideEffect]> {
  let summary = "get";
  let arguments = (ins LLVM_AnyPointer:$ctx, I32:$key);
  let results = (outs LLVM_AnyPointer);
}

def SetOp : KV_Op<"set", []> {
  let summary = "set";
  let arguments = (ins LLVM_AnyPointer:$ctx, I32:$key, I32:$value);
  let results = (outs LLVM_AnyPointer);
}

def IncrOp : KV_Op<"incr", []> {
  let summary = "incr";
  let arguments = (ins LLVM_AnyPointer:$ctx, I32:$key, I32:$value);
  let results = (outs LLVM_AnyPointer);
}

// Utility Ops

def IvOp : KV_Op<"ival", [NoSideEffect]> {
  let summary = "Integer Value from reply";
  let arguments = (ins LLVM_AnyPointer:$reply);
  let results = (outs I32);
}

def SvOp : KV_Op<"sval", [NoSideEffect]> {
  let summary = "String Value from reply";
  let arguments = (ins LLVM_AnyPointer:$reply);
  let results = (outs LLVM_AnyPointer);
}

#endif // KV_BASE
