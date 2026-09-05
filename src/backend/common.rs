//! Common Backend Implementation

use crate::traits::{SimdAdd, SimdBitAnd, SimdBitNot, SimdBitOr, SimdBitXor, SimdDiv, SimdDot, SimdHSum, SimdMatMul, SimdMatVec, SimdHMax, SimdMul, SimdShiftLeft, SimdShiftRight, SimdSub, SimdVMat, SimdHMin, SimdHOr, SimdTranspose, SimdMask, SimdScalarMul, SimdOuterProduct};

pub trait Backend: ArithmeticBackend +
                   ScalarMulBackend +
                   BitOperationsBackend +
                   BitShiftBackend +
                   ProductBackend +
                   HorizontalHSumBackend +
                   HorizontalMaxBackend +
                   HorizontalMinBackend +
                   HorizontalOrBackend +
                   TransposeBackend +
                   MaskBackend {
    fn new() -> Self;
}
pub trait MaskBackend: SimdMask<f32> +
                       SimdMask<f64> +
                       SimdMask<i8> +
                       SimdMask<i16> +
                       SimdMask<i32> +
                       SimdMask<i64> +
                       SimdMask<i16> +
                       SimdMask<i8> {
}
pub trait TransposeBackend: SimdTranspose<f32> +
                            SimdTranspose<f64> +
                            SimdTranspose<i32> +
                            SimdTranspose<i64> +
                            SimdTranspose<i16> +
                            SimdTranspose<i8> {
}
pub trait HorizontalHSumBackend: SimdHSum<f32> +
                                SimdHSum<f64> +
                                SimdHSum<i32> +
                                SimdHSum<i64> +
                                SimdHSum<i16> +
                                SimdHSum<i8> {
}
pub trait HorizontalMaxBackend: SimdHMax<f32> +
                                SimdHMax<f64> +
                                SimdHMax<i32> +
                                SimdHMax<i64> +
                                SimdHMax<i16> +
                                SimdHMax<i8> {
}
pub trait HorizontalMinBackend: SimdHMin<f32> +
                                SimdHMin<f64> +
                                SimdHMin<i32> +
                                SimdHMin<i64> +
                                SimdHMin<i16> +
                                SimdHMin<i8> {
}
pub trait HorizontalOrBackend: SimdHOr<f32> +
                               SimdHOr<f64> +
                               SimdHOr<i32> +
                               SimdHOr<i64> +
                               SimdHOr<i16> +
                               SimdHOr<i8> {
}
pub trait ProductBackend: SimdDot<f32,f32,f32> +
                          SimdDot<f64,f64,f64> +
                          SimdDot<i8,i8,i32> +
                          SimdDot<i8,i16,i32> +
                          SimdDot<i16,i16,i32> +
                          SimdVMat<f32,f32,f32> +
                          SimdVMat<f64,f64,f64> +
                          SimdVMat<i8,i8,i32> +
                          SimdVMat<i8,i16,i32> +
                          SimdVMat<i16,i16,i32> +
                          SimdMatVec<f32,f32,f32> +
                          SimdMatVec<f64,f64,f64> +
                          SimdMatVec<i8,i8,i32> +
                          SimdMatVec<i8,i16,i32> +
                          SimdMatVec<i16,i16,i32> +
                          SimdMatMul<f32,f32,f32> +
                          SimdMatMul<f64,f64,f64> +
                          SimdMatMul<i8,i8,i32> +
                          SimdMatMul<i8,i16,i32> +
                          SimdMatMul<i16,i16,i32> +
                          SimdOuterProduct<f32,f32,f32> +
                          SimdOuterProduct<f64,f64,f64> +
                          SimdOuterProduct<i8,i8,i32> +
                          SimdOuterProduct<i8,i16,i32> +
                          SimdOuterProduct<i16,i16,i32> {
}
pub trait BitShiftBackend: SimdShiftLeft<i8> +
                        SimdShiftLeft<i16> +
                        SimdShiftLeft<i32> +
                        SimdShiftLeft<i64> +
                        SimdShiftLeft<f32> +
                        SimdShiftLeft<f64> +
                        SimdShiftRight<i8> +
                        SimdShiftRight<i16> +
                        SimdShiftRight<i32> +
                        SimdShiftRight<i64> +
                        SimdShiftRight<f32> +
                        SimdShiftRight<f64> {
}
pub trait BitOperationsBackend: SimdBitOr<i8> +
                                SimdBitOr<i16> +
                                SimdBitOr<i32> +
                                SimdBitOr<i64> +
                                SimdBitOr<f32> +
                                SimdBitOr<f64> +
                                SimdBitAnd<i8> +
                                SimdBitAnd<i16> +
                                SimdBitAnd<i32> +
                                SimdBitAnd<i64> +
                                SimdBitAnd<f32> +
                                SimdBitAnd<f64> +
                                SimdBitXor<i8> +
                                SimdBitXor<i16> +
                                SimdBitXor<i32> +
                                SimdBitXor<i64> +
                                SimdBitXor<f32> +
                                SimdBitXor<f64> +
                                SimdBitNot<i8> +
                                SimdBitNot<i16> +
                                SimdBitNot<i32> +
                                SimdBitNot<i64> +
                                SimdBitNot<f32> +
                                SimdBitNot<f64> {

}
pub trait ScalarMulBackend: SimdScalarMul<i8,i8,i32> +
                            SimdScalarMul<i8,i16,i32> +
                            SimdScalarMul<i16,i16,i32> +
                            SimdScalarMul<i32,i32,i32> +
                            SimdScalarMul<i64,i64,i64> +
                            SimdScalarMul<f32,f32,f32> +
                            SimdScalarMul<f64,f64,f64> {
}
pub trait ArithmeticBackend: SimdAdd<f32,f32,f32> +
                             SimdAdd<f64,f64,f64> +
                             SimdAdd<i32,i32,i32> +
                             SimdAdd<i64,i64,i64> +
                             SimdAdd<i16,i16,i16> +
                             SimdAdd<i8,i8,i8> +
                             SimdSub<f32,f32,f32> +
                             SimdSub<f64,f64,f64> +
                             SimdSub<i32,i32,i32> +
                             SimdSub<i64,i64,i64> +
                             SimdSub<i16,i16,i16> +
                             SimdSub<i8,i8,i8> +
                             SimdDiv<f32,f32,f32> +
                             SimdDiv<f64,f64,f64> +
                             SimdDiv<i32,i32,i32> +
                             SimdDiv<i64,i64,i64> +
                             SimdDiv<i16,i16,i16> +
                             SimdDiv<i8,i8,i8> +
                             SimdMul<f32,f32,f32> +
                             SimdMul<f64,f64,f64> +
                             SimdMul<i32,i32,i32> +
                             SimdMul<i64,i64,i64> +
                             SimdMul<i8,i8,i32> +
                             SimdMul<i8,i16,i32> +
                             SimdMul<i16,i16,i32> {
}