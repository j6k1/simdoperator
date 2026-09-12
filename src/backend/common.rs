//! Common Backend Implementation

use crate::traits::{SimdAddVector, SimdBitAndVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdDot, SimdHSum, SimdMatMul, SimdMatVec, SimdHMax, SimdMulVector, SimdShlVector, SimdShrVector, SimdSubVector, SimdVMat, SimdHMin, SimdMask, SimdScalarMulVector, SimdOuterProduct, SimdLoad, SimdStore, SimdReg, SimdMulAdd, SimdZero};

pub trait Backend {
    fn new() -> Self;
}
