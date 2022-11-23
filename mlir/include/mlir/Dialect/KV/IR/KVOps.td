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


// Use NoOp as a testbed for experimenting with MLIR
def NoOp : KV_Op<"nop", [NoSideEffect]> {
  let summary = "nop";
  let arguments = (ins LLVM_AnyPointer:$ptr);
  let results = (outs LLVM_AnyPointer);
}


def GetOp : KV_Op<"get", [NoSideEffect]> {
  let summary = "get";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key);
  let results = (outs AnyTypeOf<[I32, LLVM_AnyPointer]>:$res);
}

def HGetOp : KV_Op<"hget", [NoSideEffect]> {
  let summary = "hget";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$tab,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key);
  let results = (outs AnyTypeOf<[I32, LLVM_AnyPointer]>:$res);
}

def MGetOp : KV_Op<"mget", [NoSideEffect]> {
  let summary = "mget";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$keys);
  let results = (outs Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$results);
}

def HMGetOp : KV_Op<"hmget", [NoSideEffect]> {
  let summary = "hmget";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$tab,
                   Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$keys);
  let results = (outs Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$results);
}

def SetOp : KV_Op<"set", []> {
  let summary = "set";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$value);
  let hasCanonicalizer = 1;
}

def HSetOp : KV_Op<"hset", []> {
  let summary = "hset";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$tab,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$value);
  let hasCanonicalizer = 1;
}

def MSetOp : KV_Op<"mset", [SameVariadicOperandSize]> {
  let summary = "mset";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$keys,
                   Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$values);
}

def HMSetOp : KV_Op<"hmset", [SameVariadicOperandSize]> {
  let summary = "hmset";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$tab,
                   Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$keys,
                   Variadic<AnyTypeOf<[I32, LLVM_AnyPointer]>>:$values);
}

def GetDelOp : KV_Op<"getdel", []> {
  let summary = "getdel";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key);
  let results = (outs AnyTypeOf<[I32, LLVM_AnyPointer]>:$res);
}

def GetSetOp : KV_Op<"getset", []> {
  let summary = "getset";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$value);
  let results = (outs AnyTypeOf<[I32, LLVM_AnyPointer]>:$res);
}

def DelOp : KV_Op<"del", []> {
  let summary = "del";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key);
}

def IncrByOp : KV_Op<"incrby", []> {
  let summary = "incrby";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$value);
}

def HIncrByOp : KV_Op<"hincrby", []> {
  let summary = "hincrby";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$tab,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$key,
                   AnyTypeOf<[I32, LLVM_AnyPointer]>:$value);
}

def ExistsOp : KV_Op<"exists", []> {
  let summary = "exists";
  let arguments = (ins LLVM_AnyPointer:$ctx,
                       AnyTypeOf<[I32, LLVM_AnyPointer]>:$key);
  let results = (outs I32);
}

// TODO Add operations which require loop matching

// TODO add list operations

// TODO Custom function with string attribute

#endif // KV_BASE
