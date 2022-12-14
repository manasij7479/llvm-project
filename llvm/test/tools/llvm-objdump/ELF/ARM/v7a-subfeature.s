@ RUN: llvm-mc < %s -triple armv7a -mattr=+vfp3,+neon,+fp16,+hwdiv-arm -filetype=obj | llvm-objdump -d - | FileCheck %s

.eabi_attribute Tag_FP_arch, 3 // VFP3
.eabi_attribute Tag_Advanced_SIMD_arch, 2 // SIMDv1 with fp16
.eabi_attribute Tag_DIV_use, 2 // permitted

vfp2:
  vmla.f32 s0, s1, s2

@CHECK-LABEL: vfp2
@CHECK: ee000a81    vmla.f32 s0, s1, s2

vfp3:
  vmov.f32 s0, #0.5

@CHECK-LABEL: vfp3
@CHECK: eeb60a00    vmov.f32 s0, #5.000000e-01

neon:
  vmla.f32 d0, d1, d2

@CHECK-LABEL: neon
@CHECK: f2010d12    vmla.f32 d0, d1, d2

fp16:
  vcvt.f32.f16 q0, d2

@CHECK-LABEL: fp16
@CHECK: f3b60702     vcvt.f32.f16 q0, d2

div:
  udiv r0, r1, r2

@CHECK-LABEL: div
@CHECK: e730f211    udiv r0, r1, r2

