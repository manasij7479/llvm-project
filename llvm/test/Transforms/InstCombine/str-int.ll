; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

@.str = private unnamed_addr constant [3 x i8] c"12\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"0\00", align 1
@.str.2 = private unnamed_addr constant [11 x i8] c"4294967296\00", align 1
@.str.3 = private unnamed_addr constant [24 x i8] c"10000000000000000000000\00", align 1
@.str.4 = private unnamed_addr constant [20 x i8] c"9923372036854775807\00", align 1
@.str.5 = private unnamed_addr constant [11 x i8] c"4994967295\00", align 1
@.str.6 = private unnamed_addr constant [10 x i8] c"499496729\00", align 1
@.str.7 = private unnamed_addr constant [11 x i8] c"4994967295\00", align 1

declare i32 @strtol(i8*, i8**, i32)
declare i32 @atoi(i8*)
declare i32 @atol(i8*)
declare i64 @atoll(i8*)
declare i64 @strtoll(i8*, i8**, i32)

define i32 @strtol_dec() #0 {
; CHECK-LABEL: @strtol_dec(
; CHECK-NEXT:    ret i32 12
;
  %call = call i32 @strtol(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8** null, i32 10) #2
  ret i32 %call
}

define i32 @strtol_base_zero() #0 {
; CHECK-LABEL: @strtol_base_zero(
; CHECK-NEXT:    ret i32 12
;
  %call = call i32 @strtol(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8** null, i32 0) #2
  ret i32 %call
}

define i32 @strtol_hex() #0 {
; CHECK-LABEL: @strtol_hex(
; CHECK-NEXT:    ret i32 18
;
  %call = call i32 @strtol(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8** null, i32 16) #2
  ret i32 %call
}

; Fold a call to strtol with an endptr known to be nonnull (the result
; of pointer increment).

define i32 @strtol_endptr_not_null(i8** %pend) {
; CHECK-LABEL: @strtol_endptr_not_null(
; CHECK-NEXT:    [[ENDP1:%.*]] = getelementptr inbounds i8*, i8** [[PEND:%.*]], i64 1
; CHECK-NEXT:    store i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 2), i8** [[ENDP1]], align 8
; CHECK-NEXT:    ret i32 12
;
  %endp1 = getelementptr inbounds i8*, i8** %pend, i32 1
  %call = call i32 @strtol(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8** %endp1, i32 10)
  ret i32 %call
}

; Don't fold a call to strtol with an endptr that's not known to be nonnull.

define i32 @strtol_endptr_maybe_null(i8** %end) {
; CHECK-LABEL: @strtol_endptr_maybe_null(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @strtol(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0), i8** [[END:%.*]], i32 10)
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @strtol(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i32 0, i32 0), i8** %end, i32 10)
  ret i32 %call
}

define i32 @atoi_test() #0 {
; CHECK-LABEL: @atoi_test(
; CHECK-NEXT:    ret i32 12
;
  %call = call i32 @atoi(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0)) #4
  ret i32 %call
}

define i32 @strtol_not_const_str(i8* %s) #0 {
; CHECK-LABEL: @strtol_not_const_str(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @strtol(i8* nocapture [[S:%.*]], i8** null, i32 10)
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @strtol(i8* %s, i8** null, i32 10) #3
  ret i32 %call
}

define i32 @atoi_not_const_str(i8* %s) #0 {
; CHECK-LABEL: @atoi_not_const_str(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @atoi(i8* nocapture [[S:%.*]])
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @atoi(i8* %s) #4
  ret i32 %call
}

define i32 @strtol_not_const_base(i32 %b) #0 {
; CHECK-LABEL: @strtol_not_const_base(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @strtol(i8* nocapture getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i8** null, i32 [[B:%.*]])
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @strtol(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8** null, i32 %b) #2
  ret i32 %call
}

define i32 @strtol_long_int() #0 {
; CHECK-LABEL: @strtol_long_int(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @strtol(i8* nocapture getelementptr inbounds ([11 x i8], [11 x i8]* @.str.2, i64 0, i64 0), i8** null, i32 10)
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @strtol(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.2, i32 0, i32 0), i8** null, i32 10) #3
  ret i32 %call
}


define i32 @strtol_big_overflow() #0 {
; CHECK-LABEL: @strtol_big_overflow(
; CHECK-NEXT:    [[CALL:%.*]] = call i32 @strtol(i8* nocapture getelementptr inbounds ([24 x i8], [24 x i8]* @.str.3, i64 0, i64 0), i8** null, i32 10)
; CHECK-NEXT:    ret i32 [[CALL]]
;
  %call = call i32 @strtol(i8* nocapture getelementptr inbounds ([24 x i8], [24 x i8]* @.str.3, i32 0, i32 0), i8** null, i32 10) #2
  ret i32 %call
}

define i32 @atol_test() #0 {
; CHECK-LABEL: @atol_test(
; CHECK-NEXT:    ret i32 499496729
;
; CHECK-NEXT
  %call = call i32 @atol(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.6, i32 0, i32 0)) #4
  ret i32 %call
}

define i64 @atoll_test() #0 {
; CHECK-LABEL: @atoll_test(
; CHECK-NEXT:    ret i64 4994967295
;
  %call = call i64 @atoll(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.5, i32 0, i32 0)) #3
  ret i64 %call
}

define i64 @strtoll_test() #0 {
; CHECK-LABEL: @strtoll_test(
; CHECK-NEXT:    ret i64 4994967295
;
; CHECK-NEXT
  %call = call i64 @strtoll(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.7, i32 0, i32 0), i8** null, i32 10) #5
  ret i64 %call
}
