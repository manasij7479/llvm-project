; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   -mtriple=powerpc64le-linux-gnu < %s | FileCheck \
; RUN:   -check-prefix=CHECK-LE %s
; RUN: llc -ppc-asm-full-reg-names -verify-machineinstrs \
; RUN:   -mtriple=powerpc64-linux-gnu < %s | FileCheck \
; RUN:   -check-prefix=CHECK-BE %s

define dso_local i24 @_Z1f1c(i24 %g.coerce) local_unnamed_addr #0 {
; CHECK-LE-LABEL: _Z1f1c:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    clrlwi r3, r3, 24
; CHECK-LE-NEXT:    xxlxor f1, f1, f1
; CHECK-LE-NEXT:    mtfprwz f0, r3
; CHECK-LE-NEXT:    xscvuxddp f0, f0
; CHECK-LE-NEXT:    xsmuldp f0, f0, f1
; CHECK-LE-NEXT:    xscvdpsxws f0, f0
; CHECK-LE-NEXT:    mffprwz r3, f0
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: _Z1f1c:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    clrldi r3, r3, 56
; CHECK-BE-NEXT:    std r3, -16(r1)
; CHECK-BE-NEXT:    addis r3, r2, .LCPI0_0@toc@ha
; CHECK-BE-NEXT:    lfd f0, -16(r1)
; CHECK-BE-NEXT:    lfs f1, .LCPI0_0@toc@l(r3)
; CHECK-BE-NEXT:    fcfid f0, f0
; CHECK-BE-NEXT:    fmul f0, f0, f1
; CHECK-BE-NEXT:    fctiwz f0, f0
; CHECK-BE-NEXT:    stfd f0, -8(r1)
; CHECK-BE-NEXT:    lwz r3, -4(r1)
; CHECK-BE-NEXT:    blr
entry:
  %0 = and i24 %g.coerce, 255
  %conv1 = uitofp i24 %0 to double
  %mul = fmul double 0.000000e+00, %conv1
  %conv2 = fptoui double %mul to i8
  %retval.sroa.0.0.insert.ext = zext i8 %conv2 to i24
  ret i24 %retval.sroa.0.0.insert.ext
}

attributes #0 = { "use-soft-float"="false" }
