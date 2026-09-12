//! Common Backend Implementation

use crate::traits::{SimdAddVector, SimdBitAndVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdDot, SimdHSum, SimdMatMul, SimdMatVec, SimdHMax, SimdMulVector, SimdShlVector, SimdShrVector, SimdSubVector, SimdVMat, SimdHMin, SimdMask, SimdScalarMulVector, SimdOuterProduct, SimdLoad, SimdStore, SimdReg, SimdMulAdd, SimdZero};

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
pub trait BitShiftBackend: SimdShlVector<i8> +
                           SimdShlVector<i16> +
                           SimdShlVector<i32> +
                           SimdShlVector<f32> +
                           SimdShlVector<f64> +
                           SimdShrVector<i8> +
                           SimdShrVector<i16> +
                           SimdShrVector<i32> +
                           SimdShrVector<f32> +
                           SimdShrVector<f64> {
}
pub trait BitOperationsBackend: SimdBitOrVector<i8> +
                                SimdBitOrVector<i16> +
                                SimdBitOrVector<i32> +
                                SimdBitOrVector<f32> +
                                SimdBitOrVector<f64> +
                                SimdBitAndVector<i8> +
                                SimdBitAndVector<i16> +
                                SimdBitAndVector<i32> +
                                SimdBitAndVector<f32> +
                                SimdBitAndVector<f64> +
                                SimdBitXorVector<i8> +
                                SimdBitXorVector<i16> +
                                SimdBitXorVector<i32> +
                                SimdBitXorVector<f32> +
                                SimdBitXorVector<f64> +
                                SimdBitNotVector<i8> +
                                SimdBitNotVector<i16> +
                                SimdBitNotVector<i32> +
                                SimdBitNotVector<f32> +
                                SimdBitNotVector<f64> {

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
pub trait ScalarMulBackend: SimdScalarMulVector<i8,i8,i32> +
                            SimdScalarMulVector<i8,i16,i32> +
                            SimdScalarMulVector<i16,i16,i32> +
                            SimdScalarMulVector<i32,i32,i32> +
                            SimdScalarMulVector<f32,f32,f32> +
                            SimdScalarMulVector<f64,f64,f64> {
}
pub trait ArithmeticBackend: SimdAddVector<f32,f32,f32> +
                             SimdAddVector<f64,f64,f64> +
                             SimdAddVector<i32,i32,i32> +
                             SimdAddVector<i16,i16,i16> +
                             SimdAddVector<i8,i8,i8> +
                             SimdSubVector<f32,f32,f32> +
                             SimdSubVector<f64,f64,f64> +
                             SimdSubVector<i32,i32,i32> +
                             SimdSubVector<i16,i16,i16> +
                             SimdSubVector<i8,i8,i8> +
                             SimdMulVector<f32,f32,f32> +
                             SimdMulVector<f64,f64,f64> +
                             SimdMulVector<i32,i32,i32> +
                             SimdMulVector<i8,i8,i32> +
                             SimdMulVector<i8,i16,i32> +
                             SimdMulVector<i16,i16,i32> {
}