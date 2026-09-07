//! Common Backend Implementation

use crate::traits::{SimdAdd, SimdBitAnd, SimdBitNot, SimdBitOr, SimdBitXor, SimdDot, SimdHSum, SimdMatMul, SimdMatVec, SimdHMax, SimdMul, SimdShiftLeft, SimdShiftRight, SimdSub, SimdVMat, SimdHMin, SimdMask, SimdScalarMul, SimdOuterProduct, SimdLoad, SimdStore, SimdReg, SimdMulAdd, SimdZero};

pub trait Backend/* : ArithmeticBackend +
                   ScalarMulBackend +
                   MulAddBackend +
                   BitOperationsBackend +
                   BitShiftBackend +
                   ProductBackend +
                   HorizontalHSumBackend +
                   HorizontalMaxBackend +
                   HorizontalMinBackend +
                   MaskBackend +
                   StoreBackend +
                   LoadBackend +
                   RegBackend */{
    fn new() -> Self;
}
pub trait RegBackend: SimdReg<f32> +
                      SimdReg<f64> +
                      SimdReg<i8> +
                      SimdReg<i16> +
                      SimdReg<i32> +
                      SimdReg<i64> {
}
pub trait LoadBackend: SimdLoad<f32> +
                       SimdLoad<f64> +
                       SimdLoad<i8> +
                       SimdLoad<i16> +
                       SimdLoad<i32> +
                       SimdLoad<i64> {
}
pub trait StoreBackend: SimdStore<f32> +
                        SimdStore<f64> +
                        SimdStore<i8> +
                        SimdStore<i16> +
                        SimdStore<i32> {
}
pub trait MaskBackend: SimdMask<f32> +
                       SimdMask<f64> +
                       SimdMask<i8> +
                       SimdMask<i16> +
                       SimdMask<i32> {
}
pub trait HorizontalHSumBackend: SimdHSum<f32> +
                                 SimdHSum<f64> +
                                 SimdHSum<i32> {
}
pub trait HorizontalMaxBackend: SimdHMax<f32> +
                                SimdHMax<f64> +
                                SimdHMax<i32> +
                                SimdHMax<i16> +
                                SimdHMax<i8> {
}
pub trait HorizontalMinBackend: SimdHMin<f32> +
                                SimdHMin<f64> +
                                SimdHMin<i32> +
                                SimdHMin<i16> +
                                SimdHMin<i8> {
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
                           SimdShiftLeft<f32> +
                           SimdShiftLeft<f64> +
                           SimdShiftRight<i8> +
                           SimdShiftRight<i16> +
                           SimdShiftRight<i32> +
                           SimdShiftRight<f32> +
                           SimdShiftRight<f64> {
}
pub trait BitOperationsBackend: SimdBitOr<i8> +
                                SimdBitOr<i16> +
                                SimdBitOr<i32> +
                                SimdBitOr<f32> +
                                SimdBitOr<f64> +
                                SimdBitAnd<i8> +
                                SimdBitAnd<i16> +
                                SimdBitAnd<i32> +
                                SimdBitAnd<f32> +
                                SimdBitAnd<f64> +
                                SimdBitXor<i8> +
                                SimdBitXor<i16> +
                                SimdBitXor<i32> +
                                SimdBitXor<f32> +
                                SimdBitXor<f64> +
                                SimdBitNot<i8> +
                                SimdBitNot<i16> +
                                SimdBitNot<i32> +
                                SimdBitNot<f32> +
                                SimdBitNot<f64> {

}
pub trait ZeroBackend: SimdZero<i8> +
                       SimdZero<i16> +
                       SimdZero<i32> +
                       SimdZero<f32> +
                       SimdZero<f64> {}
pub trait MulAddBackend: SimdMulAdd<f32,f32,f32> +
                          SimdMulAdd<f64,f64,f64> +
                          SimdMulAdd<i32,i32,i32> +
                          SimdMulAdd<i8,i8,i32> +
                          SimdMulAdd<i8,i16,i32> +
                          SimdMulAdd<i16,i16,i32> {
}
pub trait ScalarMulBackend: SimdScalarMul<i8,i8,i32> +
                            SimdScalarMul<i8,i16,i32> +
                            SimdScalarMul<i16,i16,i32> +
                            SimdScalarMul<i32,i32,i32> +
                            SimdScalarMul<f32,f32,f32> +
                            SimdScalarMul<f64,f64,f64> {
}
pub trait ArithmeticBackend: SimdAdd<f32,f32,f32> +
                             SimdAdd<f64,f64,f64> +
                             SimdAdd<i32,i32,i32> +
                             SimdAdd<i16,i16,i16> +
                             SimdAdd<i8,i8,i8> +
                             SimdSub<f32,f32,f32> +
                             SimdSub<f64,f64,f64> +
                             SimdSub<i32,i32,i32> +
                             SimdSub<i16,i16,i16> +
                             SimdSub<i8,i8,i8> +
                             SimdMul<f32,f32,f32> +
                             SimdMul<f64,f64,f64> +
                             SimdMul<i32,i32,i32> +
                             SimdMul<i8,i8,i32> +
                             SimdMul<i8,i16,i32> +
                             SimdMul<i16,i16,i32> {
}