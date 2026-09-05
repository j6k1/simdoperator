//! Implementation of SIMD Operations Using AVX2

use std::arch::x86_64::{__m256, __m256d, __m256i, _mm256_and_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_loadu_si256};
use crate::backend::common::Backend;
use crate::traits::{SimdLanes, SimdMask, SimdReg};
use crate::{Vector};

pub struct Avx2 {

}
impl Backend for Avx2 {
    fn new() -> Self {
        Avx2 {}
    }
}
impl SimdLanes<f32> for Avx2 {
    const LANES: usize = 8;
}
impl SimdLanes<f64> for Avx2 {
    const LANES: usize = 4;
}
impl SimdLanes<i8> for Avx2 {
    const LANES: usize = 32;
}
impl SimdLanes<i16> for Avx2 {
    const LANES: usize = 16;
}
impl SimdLanes<i32> for Avx2 {
    const LANES: usize = 8;
}
impl SimdLanes<i64> for Avx2 {
    const LANES: usize = 4;
}
impl SimdReg<i8> for Avx2 {
    type Reg = __m256i;
}
impl SimdReg<i16> for Avx2 {
    type Reg = __m256i;
}
impl SimdReg<i32> for Avx2 {
    type Reg = __m256i;
}
impl SimdReg<f32> for Avx2 {
    type Reg = __m256;
}
impl SimdReg<f64> for Avx2 {
    type Reg = __m256d;
}
impl SimdMask<i8> for Avx2 where Self: SimdReg<i8> {
    type Mask = __m256i;

    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpgt_epi8(a,b) }
    }

    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi8(a,b) }
    }

    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_si256(a, mask) }
    }

    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_epi8(mask,a,b) }
    }

    fn tail_mask(&self, index: usize, total: usize) -> Self::Mask {
        let mut arr = [0u32; 8];

        for i in 0..8 {
            if index + i < total {
                arr[i] = u32::MAX;
            }
        }

        unsafe { _mm256_loadu_si256(arr.as_ptr() as *const __m256i) }
    }
}