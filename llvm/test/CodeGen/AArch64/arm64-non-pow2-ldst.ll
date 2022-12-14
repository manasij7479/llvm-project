; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=arm64-eabi -verify-machineinstrs | FileCheck %s

define i24 @ldi24(ptr %p) nounwind {
; CHECK-LABEL: ldi24:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldrb w8, [x0, #2]
; CHECK-NEXT:    ldrh w0, [x0]
; CHECK-NEXT:    bfi w0, w8, #16, #16
; CHECK-NEXT:    ret
    %r = load i24, i24* %p
    ret i24 %r
}

define i56 @ldi56(ptr %p) nounwind {
; CHECK-LABEL: ldi56:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldrb w8, [x0, #6]
; CHECK-NEXT:    ldrh w9, [x0, #4]
; CHECK-NEXT:    ldr w0, [x0]
; CHECK-NEXT:    bfi w9, w8, #16, #16
; CHECK-NEXT:    bfi x0, x9, #32, #32
; CHECK-NEXT:    ret
    %r = load i56, i56* %p
    ret i56 %r
}

define i80 @ldi80(ptr %p) nounwind {
; CHECK-LABEL: ldi80:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldr x8, [x0]
; CHECK-NEXT:    ldrh w1, [x0, #8]
; CHECK-NEXT:    mov x0, x8
; CHECK-NEXT:    ret
    %r = load i80, i80* %p
    ret i80 %r
}

define i120 @ldi120(ptr %p) nounwind {
; CHECK-LABEL: ldi120:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldrb w8, [x0, #14]
; CHECK-NEXT:    ldrh w9, [x0, #12]
; CHECK-NEXT:    ldr w1, [x0, #8]
; CHECK-NEXT:    ldr x0, [x0]
; CHECK-NEXT:    bfi w9, w8, #16, #16
; CHECK-NEXT:    bfi x1, x9, #32, #32
; CHECK-NEXT:    ret
    %r = load i120, i120* %p
    ret i120 %r
}

define i280 @ldi280(ptr %p) nounwind {
; CHECK-LABEL: ldi280:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ldp x8, x1, [x0]
; CHECK-NEXT:    ldrb w9, [x0, #34]
; CHECK-NEXT:    ldrh w4, [x0, #32]
; CHECK-NEXT:    ldp x2, x3, [x0, #16]
; CHECK-NEXT:    mov x0, x8
; CHECK-NEXT:    bfi x4, x9, #16, #8
; CHECK-NEXT:    ret
    %r = load i280, i280* %p
    ret i280 %r
}

define void @sti24(ptr %p, i24 %a) nounwind {
; CHECK-LABEL: sti24:
; CHECK:       // %bb.0:
; CHECK-NEXT:    lsr w8, w1, #16
; CHECK-NEXT:    strh w1, [x0]
; CHECK-NEXT:    strb w8, [x0, #2]
; CHECK-NEXT:    ret
    store i24 %a, i24* %p
    ret void
}

define void @sti56(ptr %p, i56 %a) nounwind {
; CHECK-LABEL: sti56:
; CHECK:       // %bb.0:
; CHECK-NEXT:    lsr x8, x1, #48
; CHECK-NEXT:    lsr x9, x1, #32
; CHECK-NEXT:    str w1, [x0]
; CHECK-NEXT:    strb w8, [x0, #6]
; CHECK-NEXT:    strh w9, [x0, #4]
; CHECK-NEXT:    ret
    store i56 %a, i56* %p
    ret void
}

define void @sti80(ptr %p, i80 %a) nounwind {
; CHECK-LABEL: sti80:
; CHECK:       // %bb.0:
; CHECK-NEXT:    str x2, [x0]
; CHECK-NEXT:    strh w3, [x0, #8]
; CHECK-NEXT:    ret
    store i80 %a, i80* %p
    ret void
}

define void @sti120(ptr %p, i120 %a) nounwind {
; CHECK-LABEL: sti120:
; CHECK:       // %bb.0:
; CHECK-NEXT:    lsr x8, x3, #48
; CHECK-NEXT:    lsr x9, x3, #32
; CHECK-NEXT:    str x2, [x0]
; CHECK-NEXT:    str w3, [x0, #8]
; CHECK-NEXT:    strb w8, [x0, #14]
; CHECK-NEXT:    strh w9, [x0, #12]
; CHECK-NEXT:    ret
    store i120 %a, i120* %p
    ret void
}

define void @sti280(ptr %p, i280 %a) nounwind {
; CHECK-LABEL: sti280:
; CHECK:       // %bb.0:
; CHECK-NEXT:    lsr x8, x6, #16
; CHECK-NEXT:    stp x4, x5, [x0, #16]
; CHECK-NEXT:    stp x2, x3, [x0]
; CHECK-NEXT:    strh w6, [x0, #32]
; CHECK-NEXT:    strb w8, [x0, #34]
; CHECK-NEXT:    ret
    store i280 %a, i280* %p
    ret void
}
