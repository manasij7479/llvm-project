// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// expected-error @+1 {{cannot be converted to LLVM IR}}
func.func @foo() {
  llvm.return
}

// -----

// expected-error @+1 {{llvm.noalias attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_noalias(%arg0 : f32 {llvm.noalias}) -> f32 {
  llvm.return %arg0 : f32
}

// -----

// expected-error @+1 {{llvm.sret attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_sret(%arg0 : f32 {llvm.sret = f32}) -> f32 {
  llvm.return %arg0 : f32
}

// -----

// expected-error @+1 {{llvm.sret attribute attached to LLVM pointer argument of a different type}}
llvm.func @invalid_sret(%arg0 : !llvm.ptr<f32> {llvm.sret = i32}) -> !llvm.ptr<f32> {
  llvm.return %arg0 : !llvm.ptr<f32>
}

// -----

// expected-error @+1 {{llvm.nest attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_nest(%arg0 : f32 {llvm.nest}) -> f32 {
  llvm.return %arg0 : f32
}
// -----

// expected-error @+1 {{llvm.byval attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_byval(%arg0 : f32 {llvm.byval = f32}) -> f32 {
  llvm.return %arg0 : f32
}

// -----

// expected-error @+1 {{llvm.byval attribute attached to LLVM pointer argument of a different type}}
llvm.func @invalid_sret(%arg0 : !llvm.ptr<f32> {llvm.byval = i32}) -> !llvm.ptr<f32> {
  llvm.return %arg0 : !llvm.ptr<f32>
}

// -----

// expected-error @+1 {{llvm.byref attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_byval(%arg0 : f32 {llvm.byref = f32}) -> f32 {
  llvm.return %arg0 : f32
}

// -----

// expected-error @+1 {{llvm.byref attribute attached to LLVM pointer argument of a different type}}
llvm.func @invalid_sret(%arg0 : !llvm.ptr<f32> {llvm.byref = i32}) -> !llvm.ptr<f32> {
  llvm.return %arg0 : !llvm.ptr<f32>
}

// -----

// expected-error @+1 {{llvm.inalloca attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_byval(%arg0 : f32 {llvm.inalloca = f32}) -> f32 {
  llvm.return %arg0 : f32
}

// -----

// expected-error @+1 {{llvm.inalloca attribute attached to LLVM pointer argument of a different type}}
llvm.func @invalid_sret(%arg0 : !llvm.ptr<f32> {llvm.inalloca = i32}) -> !llvm.ptr<f32> {
  llvm.return %arg0 : !llvm.ptr<f32>
}

// -----

// expected-error @+1 {{llvm.align attribute attached to LLVM non-pointer argument}}
llvm.func @invalid_align(%arg0 : f32 {llvm.align = 4}) -> f32 {
  llvm.return %arg0 : f32
}

// -----

llvm.func @no_non_complex_struct() -> !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>> {
  // expected-error @+1 {{expected struct type to be a complex number}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>>
  llvm.return %0 : !llvm.array<2 x array<2 x array<2 x struct<(i32)>>>>
}

// -----

llvm.func @no_non_complex_struct() -> !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>> {
  // expected-error @+1 {{expected struct type to be a complex number}}
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>>
  llvm.return %0 : !llvm.array<2 x array<2 x array<2 x struct<(i32, i32, i32)>>>>
}

// -----

llvm.func @struct_wrong_attribute_element_type() -> !llvm.struct<(f64, f64)> {
  // expected-error @+1 {{FloatAttr does not match expected type of the constant}}
  %0 = llvm.mlir.constant([1.0 : f32, 1.0 : f32]) : !llvm.struct<(f64, f64)>
  llvm.return %0 : !llvm.struct<(f64, f64)>
}

// -----

// expected-error @+1 {{unsupported constant value}}
llvm.mlir.global internal constant @test([2.5, 7.4]) : !llvm.array<2 x f64>

// -----

// expected-error @+1 {{LLVM attribute 'noinline' does not expect a value}}
llvm.func @passthrough_unexpected_value() attributes {passthrough = [["noinline", "42"]]}

// -----

// expected-error @+1 {{LLVM attribute 'alignstack' expects a value}}
llvm.func @passthrough_expected_value() attributes {passthrough = ["alignstack"]}

// -----

// expected-error @+1 {{expected 'passthrough' to contain string or array attributes}}
llvm.func @passthrough_wrong_type() attributes {passthrough = [42]}

// -----

// expected-error @+1 {{expected arrays within 'passthrough' to contain two strings}}
llvm.func @passthrough_wrong_type() attributes {
  passthrough = [[ 42, 42 ]]
}
