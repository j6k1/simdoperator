//! Implementation of SIMD Operations Using AVX2

use std::arch::x86_64::{__m256, __m256d, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi8, _mm256_add_pd, _mm256_add_ps, _mm256_and_pd, _mm256_and_ps, _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_blendv_pd, _mm256_blendv_ps, _mm256_castpd256_pd128, _mm256_castpd_si256, _mm256_castps256_ps128, _mm256_castps_si256, _mm256_castsi256_pd, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmp_pd, _mm256_cmp_ps, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpeq_epi8, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cmpgt_epi8, _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi16, _mm256_extractf128_pd, _mm256_extractf128_ps, _mm256_extracti128_si256, _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_mul_pd, _mm256_mul_ps, _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_set1_pd, _mm256_set1_ps, _mm256_set_m128i, _mm256_setzero_pd, _mm256_setzero_ps, _mm256_setzero_si256, _mm256_sllv_epi32, _mm256_sllv_epi64, _mm256_srlv_epi32, _mm256_srlv_epi64, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_sub_epi8, _mm256_sub_pd, _mm256_sub_ps, _mm256_unpackhi_epi32, _mm256_unpackhi_ps, _mm256_unpacklo_epi32, _mm256_unpacklo_ps, _mm256_xor_si256, _mm_add_epi32, _mm_add_pd, _mm_add_ps, _mm_add_sd, _mm_add_ss, _mm_cvtsd_f64, _mm_cvtsi128_si32, _mm_cvtss_f32, _mm_movehl_ps, _mm_packs_epi16, _mm_packs_epi32, _mm_setzero_si128, _mm_shuffle_ps, _mm_srli_si128, _mm_unpackhi_pd, _CMP_EQ_OQ, _CMP_GT_OQ};
use std::convert::Into;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul};
use crate::backend::common::{Backend, PartialDotAcc, Regs};
use crate::traits::{SimdCols, SimdDot, SimdHSum, SimdLanes, SimdLoad, SimdMask, SimdMatMul, SimdMatVec, SimdMulAdd, SimdPartialDot, SimdReg, SimdRows, SimdStore, SimdTranspose, SimdVMat, SimdZero, SimdAdd, SimdSub, SimdMul, SimdPromote, SimdScalarMul, SimdSplat, SimdBitOr, SimdBitAnd, SimdReinterpret, SimdBitXor, SimdBitNot, BitsBitAnd, BitsBitOr, BitsBitXor, SimdShl, SimdShr, SimdDemote, SimdConvert, FoldRegs, SimdShiftWidth, NativeBackend, RegsInto};
use crate::{derive_matmul, matmul_tile, ColumnMajorMatrix, MatrixMut, MatrixView, OwnedVector, VectorView};
use crate::error::InstantiationError;

/// Avx2 Backend
pub struct Avx2 {

}
impl Backend for Avx2 {
    #[inline(always)]
    fn new() -> Result<Self,InstantiationError> {
        Ok(Avx2 {})
    }
}
impl NativeBackend for Avx2 {}
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
impl SimdRows<i8> for Avx2 {
    const ROWS: usize = 1;
}
impl SimdRows<i16> for Avx2 {
    const ROWS: usize = 1;
}
impl SimdRows<i32> for Avx2 {
    const ROWS: usize = 2;
}
impl SimdRows<i64> for Avx2 {
    const ROWS: usize = 1;
}
impl SimdRows<f32> for Avx2 {
    const ROWS: usize = 2;
}
impl SimdRows<f64> for Avx2 {
    const ROWS: usize = 1;
}
impl SimdCols<i8> for Avx2 {
    const COLS: usize = 8;
}
impl SimdCols<i16> for Avx2 {
    const COLS: usize = 8;
}
impl SimdCols<i32> for Avx2 {
    const COLS: usize = 8;
}
impl SimdCols<f32> for Avx2 {
    const COLS: usize = 4;
}
impl SimdCols<f64> for Avx2 {
    const COLS: usize = 2;
}
impl SimdReg<i8> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = i8;
    type ShiftWidth = __m256i;
}
impl SimdReg<i16> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = i16;
    type ShiftWidth = __m256i;
}
impl SimdReg<i32> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = i32;
    type ShiftWidth = __m256i;
}
impl SimdReg<i64> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = i64;
    type ShiftWidth = __m256i;
}
impl SimdReg<u32> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = u32;
    type ShiftWidth = __m256i;
}
impl SimdReg<u64> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = u64;
    type ShiftWidth = __m256i;
}
impl SimdReg<f32> for Avx2 {
    type Reg = __m256;
    type Mask = __m256i;
    type Bits = i32;
    type ShiftWidth = __m256i;
}
impl SimdReg<f64> for Avx2 {
    type Reg = __m256d;
    type Mask = __m256i;
    type Bits = i64;
    type ShiftWidth = __m256i;
}
impl SimdLoad<i8> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const i8) -> Self::Reg {
        unsafe {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoad<i16> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const i16) -> Self::Reg {
        unsafe {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoad<i32> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const i32) -> Self::Reg {
        unsafe {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoad<i64> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const i64) -> Self::Reg {
        unsafe {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoad<u32> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const u32) -> Self::Reg {
        unsafe {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoad<u64> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const u64) -> Self::Reg {
        unsafe {
            _mm256_loadu_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoad<f32> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const f32) -> Self::Reg {
        unsafe {
            _mm256_loadu_ps(ptr)
        }
    }
}
impl SimdLoad<f64> for Avx2 {
    #[inline(always)]
    unsafe fn load(&self, ptr: *const f64) -> Self::Reg {
        unsafe {
            _mm256_loadu_pd(ptr)
        }
    }
}
impl SimdStore<i8> for Avx2 {
    #[inline(always)]
    unsafe fn store(&self, ptr: *mut i8, a: Self::Reg) {
        unsafe {
            _mm256_storeu_si256(ptr as *mut __m256i, a);
        }
    }
}
impl SimdStore<i16> for Avx2 {
    #[inline(always)]
    unsafe fn store(&self, ptr: *mut i16, a: Self::Reg) {
        unsafe {
            _mm256_storeu_si256(ptr as *mut __m256i, a);
        }
    }
}
impl SimdStore<i32> for Avx2 {
    #[inline(always)]
    unsafe fn store(&self, ptr: *mut i32, a: Self::Reg) {
        unsafe {
            _mm256_storeu_si256(ptr as *mut __m256i, a);
        }
    }
}
impl SimdStore<f32> for Avx2 {
    #[inline(always)]
    unsafe fn store(&self, ptr: *mut f32, a: Self::Reg) {
        unsafe {
            _mm256_storeu_ps(ptr, a);
        }
    }
}
impl SimdStore<f64> for Avx2 {
    #[inline(always)]
    unsafe fn store(&self, ptr: *mut f64, a: Self::Reg) {
        unsafe {
            _mm256_storeu_pd(ptr, a);
        }
    }
}
impl SimdMask<i8> for Avx2 where Self: SimdReg<i8> {
    #[inline(always)]
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpgt_epi8(a,b) }
    }

    #[inline(always)]
    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi8(a,b) }
    }

    #[inline(always)]
    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_epi8(b,a,mask) }
    }

    #[inline(always)]
    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_si256(a, mask) }
    }

    #[inline(always)]
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
impl SimdMask<i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpgt_epi16(a,b) }
    }

    #[inline(always)]
    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi16(a,b) }
    }

    #[inline(always)]
    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_epi8(b,a,mask) }
    }

    #[inline(always)]
    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_si256(a, mask) }
    }

    #[inline(always)]
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
impl SimdMask<i32> for Avx2 where Self: SimdReg<i32> {
    #[inline(always)]
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpgt_epi32(a,b) }
    }

    #[inline(always)]
    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi32(a,b) }
    }

    #[inline(always)]
    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_epi8(b,a,mask) }
    }

    #[inline(always)]
    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_si256(a, mask) }
    }

    #[inline(always)]
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
impl SimdMask<f32> for Avx2 where Self: SimdReg<f32> {
    #[inline(always)]
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castps_si256(_mm256_cmp_ps(a,b, _CMP_GT_OQ)) }
    }

    #[inline(always)]
    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castps_si256(_mm256_cmp_ps(a,b, _CMP_EQ_OQ)) }
    }

    #[inline(always)]
    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_ps(b, a, _mm256_castsi256_ps(mask)) }
    }

    #[inline(always)]
    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_ps(a,_mm256_castsi256_ps(mask)) }
    }

    #[inline(always)]
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
impl SimdMask<f64> for Avx2 where Self: SimdReg<f64> {
    #[inline(always)]
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castpd_si256(_mm256_cmp_pd(a,b, _CMP_GT_OQ)) }
    }

    #[inline(always)]
    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castpd_si256(_mm256_cmp_pd(a,b, _CMP_EQ_OQ)) }
    }

    #[inline(always)]
    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_pd(b, a, _mm256_castsi256_pd(mask)) }
    }

    #[inline(always)]
    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_pd(a,_mm256_castsi256_pd(mask)) }
    }

    #[inline(always)]
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
impl SimdPromote<i8,i16> for Avx2 where Self: SimdReg<i8> + SimdReg<i16> {
    type Output = [<Self as SimdReg<i16>>::Reg;2];

    #[inline(always)]
    fn promotion(&self, reg: <Self as SimdReg<i8>>::Reg) -> Self::Output {
        unsafe {
            let lo = _mm256_castsi256_si128(reg);
            let hi = _mm256_extracti128_si256(reg,1);

           [_mm256_cvtepi8_epi16(lo),_mm256_cvtepi8_epi16(hi)]
        }
    }
}
impl SimdPromote<i16,i32> for Avx2 where Self: SimdReg<i16> + SimdReg<i32> {
    type Output = [<Self as SimdReg<i32>>::Reg;2];

    #[inline(always)]
    fn promotion(&self, reg: <Self as SimdReg<i16>>::Reg) -> Self::Output {
        unsafe {
            let lo = _mm256_castsi256_si128(reg);
            let hi = _mm256_extracti128_si256(reg,1);

            let lo32 = _mm256_cvtepi16_epi32(lo);
            let hi32 = _mm256_cvtepi16_epi32(hi);

            [lo32,hi32]
        }
    }
}
impl SimdDemote<i32,i16> for Avx2 where Self: SimdReg<i32> + SimdReg<i16> {
    type Input = [<Self as SimdReg<i32>>::Reg;2];

    fn demotion(&self, reg: [<Self as SimdReg<i32>>::Reg;2]) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            let [lo32,hi32] = reg;

            let lo_lo = _mm256_castsi256_si128(lo32);
            let lo_hi = _mm256_extracti128_si256(lo32, 1);
            let hi_lo = _mm256_castsi256_si128(hi32);
            let hi_hi = _mm256_extracti128_si256(hi32, 1);

            let lo16 = _mm_packs_epi32(lo_lo, lo_hi);
            let hi16 = _mm_packs_epi32(hi_lo, hi_hi);

            _mm256_set_m128i(hi16, lo16)
        }
    }
}
impl SimdDemote<i16,i8> for Avx2 where Self: SimdReg<i16> + SimdReg<i8> {
    type Input = [<Self as SimdReg<i16>>::Reg;1];
    fn demotion(&self, reg: [<Self as SimdReg<i16>>::Reg;1]) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            let [reg16] = reg;

            let lo16 = _mm256_castsi256_si128(reg16);
            let hi16 = _mm256_extracti128_si256(reg16, 1);

            let lo8 = _mm_packs_epi16(lo16, hi16);

            _mm256_set_m128i(_mm_setzero_si128(), lo8)
        }
    }
}
impl SimdConvert<i16,f32> for Avx2 where Self: SimdReg<i16> + SimdReg<f32> {
    type Output = [<Self as SimdReg<f32>>::Reg;2];

    fn convert(&self, reg: <Self as SimdReg<i16>>::Reg) -> Self::Output {
        unsafe {
            let lo_i16 = _mm256_castsi256_si128(reg);
            let hi_i16 = _mm256_extracti128_si256(reg, 1);

            let lo_i32 = _mm256_cvtepi16_epi32(lo_i16);
            let hi_i32 = _mm256_cvtepi16_epi32(hi_i16);

            let lo_f32 = _mm256_cvtepi32_ps(lo_i32);
            let hi_f32 = _mm256_cvtepi32_ps(hi_i32);

            [lo_f32,hi_f32]
        }
    }
}
impl SimdConvert<i32,f32> for Avx2 where Self: SimdReg<i32> + SimdReg<f32> {
    type Output = [<Self as SimdReg<f32>>::Reg;1];

    fn convert(&self, reg: <Self as SimdReg<i32>>::Reg) -> Self::Output {
        unsafe {
            [_mm256_cvtepi32_ps(reg)]
        }
    }
}
impl SimdAdd<i8,i8,i8> for Avx2 where Self: SimdReg<i8> {

    #[inline(always)]
    fn add(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<i8>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            _mm256_add_epi8(l, r)
        }
    }
}
impl SimdAdd<i16,i16,i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn add(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<i16>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            _mm256_add_epi16(l, r)
        }
    }
}
impl SimdAdd<i32,i32,i32> for Avx2 where Self: SimdReg<i32> {

    #[inline(always)]
    fn add(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_add_epi32(l, r)
        }
    }
}
impl SimdAdd<f32,f32,f32> for Avx2 where Self: SimdReg<f32> {

    #[inline(always)]
    fn add(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<f32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            _mm256_add_ps(l, r)
        }
    }
}
impl SimdAdd<f64,f64,f64> for Avx2 where Self: SimdReg<f64> {

    #[inline(always)]
    fn add(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<f64>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            _mm256_add_pd(l, r)
        }
    }
}
impl SimdSub<i8,i8,i8> for Avx2 where Self: SimdReg<i8> {

    #[inline(always)]
    fn sub(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<i8>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            _mm256_sub_epi8(l, r)
        }
    }
}
impl SimdSub<i16,i16,i16> for Avx2 where Self: SimdReg<i16> {

    #[inline(always)]
    fn sub(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<i16>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            _mm256_sub_epi16(l, r)
        }
    }
}
impl SimdSub<i32,i32,i32> for Avx2 where Self: SimdReg<i32> {

    #[inline(always)]
    fn sub(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_sub_epi32(l, r)
        }
    }
}
impl SimdSub<f32,f32,f32> for Avx2 where Self: SimdReg<f32> {

    #[inline(always)]
    fn sub(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<f32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            _mm256_sub_ps(l, r)
        }
    }
}
impl SimdSub<f64,f64,f64> for Avx2 where Self: SimdReg<f64> {

    #[inline(always)]
    fn sub(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<f64>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            _mm256_sub_pd(l, r)
        }
    }
}
impl SimdMul<i8,i8,i32> for Avx2 where Self: SimdReg<i8> + SimdReg<i32> {
    type Output = [<Self as SimdReg<i32>>::Reg;4];
    type Regs = Regs<<Self as SimdReg<i32>>::Reg,4>;
    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<i8>>::Reg)
        -> [<Self as SimdReg<i32>>::Reg;4] {
        let [lo,hi] = <Self as SimdPromote<i8,i16>>::promotion(self,l);
        let [lo_r,hi_r] = <Self as SimdPromote<i8,i16>>::promotion(self,r);

        let [lo32_lo,lo32_hi] = <Self as SimdMul<i16,i16,i32>>::mul(self,lo,lo_r);
        let [hi32_lo,hi32_hi] = <Self as SimdMul<i16,i16,i32>>::mul(self,hi,hi_r);

        [lo32_lo,lo32_hi,hi32_lo,hi32_hi]
    }
}
impl SimdMul<i16,i16,i32> for Avx2 where Self: SimdReg<i16> + SimdReg<i32> {
    type Output = [<Self as SimdReg<i32>>::Reg;2];
    type Regs = Regs<<Self as SimdReg<i32>>::Reg,2>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<i16>>::Reg)
        -> [<Self as SimdReg<i32>>::Reg;2] {
        unsafe {
            let prod16 = _mm256_mullo_epi16(l, r);

            <Self as SimdPromote<i16,i32>>::promotion(self,prod16)
        }
    }
}
impl SimdMul<i32,i32,i32> for Avx2 where Self: SimdReg<i32> {
    type Output = [<Self as SimdReg<i32>>::Reg;1];
    type Regs = Regs<<Self as SimdReg<i32>>::Reg,1>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<i32>>::Reg) -> [<Self as SimdReg<i32>>::Reg;1] {
        unsafe {
            [_mm256_mullo_epi32(l, r)]
        }
    }
}
impl SimdMul<f32,f32,f32> for Avx2 where Self: SimdReg<f32> {
    type Output = [<Self as SimdReg<f32>>::Reg;1];
    type Regs = Regs<<Self as SimdReg<f32>>::Reg,1>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<f32>>::Reg) -> [<Self as SimdReg<f32>>::Reg;1] {
        unsafe {
            [_mm256_mul_ps(l, r)]
        }
    }
}
impl SimdMul<f64,f64,f64> for Avx2 where Self: SimdReg<f64> {
    type Output = [<Self as SimdReg<f64>>::Reg;1];
    type Regs = Regs<<Self as SimdReg<f64>>::Reg,1>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<f64>>::Reg) -> [<Self as SimdReg<f64>>::Reg;1] {
        unsafe {
            [_mm256_mul_pd(l, r)]
        }
    }
}
impl SimdSplat<i8> for Avx2 where Self: SimdReg<i8> {

    fn splat(&self, v: i8) -> Self::Reg {
        unsafe {
            _mm256_set1_epi8(v)
        }
    }
}
impl SimdSplat<i16> for Avx2 where Self: SimdReg<i16> {

    fn splat(&self, v: i16) -> Self::Reg {
        unsafe {
            _mm256_set1_epi16(v)
        }
    }
}
impl SimdSplat<i32> for Avx2 where Self: SimdReg<i32> {

    fn splat(&self, v: i32) -> Self::Reg {
        unsafe {
            _mm256_set1_epi32(v)
        }
    }
}
impl SimdSplat<f32> for Avx2 where Self: SimdReg<f32> {

    fn splat(&self, v: f32) -> Self::Reg {
        unsafe {
            _mm256_set1_ps(v)
        }
    }
}
impl SimdSplat<f64> for Avx2 where Self: SimdReg<f64> {

    fn splat(&self, v: f64) -> Self::Reg {
        unsafe {
            _mm256_set1_pd(v)
        }
    }
}
impl<SL,SR,SO> SimdScalarMul<SL,SR,SO> for Avx2
    where Self: SimdReg<SL> +
                SimdReg<SR> +
                SimdReg<SO> +
                SimdSplat<SL> +
                SimdMul<SL,SR,SO> {

    #[inline(always)]
    fn scalarmul(&self, l: SL, r: <Self as SimdReg<SR>>::Reg) -> <Self as SimdMul<SL,SR,SO>>::Output {
        let splat = <Self as SimdSplat<SL>>::splat(self,l);
        <Self as SimdMul<SL,SR,SO>>::mul(self,splat,r)
    }
}
impl SimdReinterpret<f32,i32> for Avx2 where Self: SimdReg<i32> {

    fn reinterpret(&self, reg: <Self as SimdReg<f32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_castps_si256(reg)
        }
    }
}
impl SimdReinterpret<f64,i64> for Avx2 where Self: SimdReg<i64> {

    fn reinterpret(&self, reg: <Self as SimdReg<f64>>::Reg) -> <Self as SimdReg<i64>>::Reg {
        unsafe {
            _mm256_castpd_si256(reg)
        }
    }
}
impl SimdReinterpret<i32,f32> for Avx2 where Self: SimdReg<f32> {

    fn reinterpret(&self, reg: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            _mm256_castsi256_ps(reg)
        }
    }
}
impl SimdReinterpret<i64,f64> for Avx2 where Self: SimdReg<f64> {

    fn reinterpret(&self, reg: <Self as SimdReg<i64>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            _mm256_castsi256_pd(reg)
        }
    }
}
impl SimdBitAnd<i8> for Avx2 where Self: SimdReg<i8> {
    #[inline(always)]
    fn bitand(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<<i8 as BitsBitAnd>::Bits>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            _mm256_and_si256(l, r)
        }
    }
}
impl SimdBitAnd<i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn bitand(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<<i16 as BitsBitAnd>::Bits>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            _mm256_and_si256(l, r)
        }
    }
}
impl SimdBitAnd<i32> for Avx2 where Self: SimdReg<i32> {
    #[inline(always)]
    fn bitand(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<<i32 as BitsBitAnd>::Bits>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_and_si256(l, r)
        }
    }
}
impl SimdBitAnd<f32> for Avx2 where Self: SimdReg<f32> {
    #[inline(always)]
    fn bitand(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<<f32 as BitsBitAnd>::Bits>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            let lr = <Self as SimdReinterpret<f32,i32>>::reinterpret(self,l);
            let lcr = _mm256_and_si256(lr, r);

            <Self as SimdReinterpret<i32,f32>>::reinterpret(self,lcr)
        }
    }
}
impl SimdBitAnd<f64> for Avx2 where Self: SimdReg<f64> {
    #[inline(always)]
    fn bitand(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<<f64 as BitsBitAnd>::Bits>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            let lr = <Self as SimdReinterpret<f64,i64>>::reinterpret(self,l);
            let lcr = _mm256_and_si256(lr, r);

            <Self as SimdReinterpret<i64,f64>>::reinterpret(self,lcr)
        }
    }
}
impl SimdBitOr<i8> for Avx2 where Self: SimdReg<i8> {
    #[inline(always)]
    fn bitor(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<<i8 as BitsBitOr>::Bits>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            _mm256_or_si256(l, r)
        }
    }
}
impl SimdBitOr<i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn bitor(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<<i16 as BitsBitOr>::Bits>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            _mm256_or_si256(l, r)
        }
    }
}
impl SimdBitOr<i32> for Avx2 where Self: SimdReg<i32> {
    #[inline(always)]
    fn bitor(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<<i32 as BitsBitOr>::Bits>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_or_si256(l, r)
        }
    }
}
impl SimdBitOr<f32> for Avx2 where Self: SimdReg<f32> {
    #[inline(always)]
    fn bitor(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<<f32 as BitsBitOr>::Bits>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            let lr = <Self as SimdReinterpret<f32,i32>>::reinterpret(self,l);
            let lcr = _mm256_or_si256(lr, r);

            <Self as SimdReinterpret<i32,f32>>::reinterpret(self,lcr)
        }
    }
}
impl SimdBitOr<f64> for Avx2 where Self: SimdReg<f64> {
    #[inline(always)]
    fn bitor(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<<f64 as BitsBitOr>::Bits>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            let lr = <Self as SimdReinterpret<f64,i64>>::reinterpret(self,l);
            let lcr = _mm256_or_si256(lr, r);

            <Self as SimdReinterpret<i64,f64>>::reinterpret(self,lcr)
        }
    }
}
impl SimdBitXor<i8> for Avx2 where Self: SimdReg<i8> {
    #[inline(always)]
    fn bitxor(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<<i8 as BitsBitXor>::Bits>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            _mm256_xor_si256(l, r)
        }
    }
}
impl SimdBitXor<i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn bitxor(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<<i16 as BitsBitXor>::Bits>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            _mm256_xor_si256(l, r)
        }
    }
}
impl SimdBitXor<i32> for Avx2 where Self: SimdReg<i32> {
    #[inline(always)]
    fn bitxor(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<<i32 as BitsBitXor>::Bits>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_xor_si256(l, r)
        }
    }
}
impl SimdBitXor<f32> for Avx2 where Self: SimdReg<f32> {
    #[inline(always)]
    fn bitxor(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<<f32 as BitsBitXor>::Bits>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            let lr = <Self as SimdReinterpret<f32,i32>>::reinterpret(self,l);
            let lcr = _mm256_xor_si256(lr, r);

            <Self as SimdReinterpret<i32,f32>>::reinterpret(self,lcr)
        }
    }
}
impl SimdBitXor<f64> for Avx2 where Self: SimdReg<f64> {
    #[inline(always)]
    fn bitxor(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<<f64 as BitsBitXor>::Bits>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            let lr = <Self as SimdReinterpret<f64,i64>>::reinterpret(self,l);
            let lcr = _mm256_xor_si256(lr, r);

            <Self as SimdReinterpret<i64,f64>>::reinterpret(self,lcr)
        }
    }
}
impl SimdBitNot<i8> for Avx2 where Self: SimdReg<i8> {
    #[inline(always)]
    fn bitnot(&self, v: <Self as SimdReg<i8>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            let mask = _mm256_set1_epi32(-1);
            _mm256_andnot_si256(v, mask)
        }
    }
}
impl SimdBitNot<i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn bitnot(&self, v: <Self as SimdReg<i16>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            let mask = _mm256_set1_epi32(-1);
            _mm256_andnot_si256(v, mask)
        }
    }
}
impl SimdBitNot<i32> for Avx2 where Self: SimdReg<i32> {
    #[inline(always)]
    fn bitnot(&self, v: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            let mask = _mm256_set1_epi32(-1);
            _mm256_andnot_si256(v, mask)
        }
    }
}
impl SimdBitNot<f32> for Avx2 where Self: SimdReg<f32> {
    #[inline(always)]
    fn bitnot(&self, v: <Self as SimdReg<f32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            let vr = <Self as SimdReinterpret<f32,i32>>::reinterpret(self,v);
            let mask = _mm256_set1_epi32(-1);
            let vcr = _mm256_andnot_si256(vr, mask);

            <Self as SimdReinterpret<i32,f32>>::reinterpret(self,vcr)
        }
    }
}
impl SimdBitNot<f64> for Avx2 where Self: SimdReg<f64> {
    #[inline(always)]
    fn bitnot(&self, v: <Self as SimdReg<f64>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            let vr = <Self as SimdReinterpret<f64,i64>>::reinterpret(self,v);
            let mask = _mm256_set1_epi32(-1);
            let vcr = _mm256_andnot_si256(vr, mask);

            <Self as SimdReinterpret<i64,f64>>::reinterpret(self,vcr)
        }
    }
}
impl SimdShiftWidth<i8> for Avx2 where Self: SimdReg<i8> {
    #[inline(always)]
    fn shift_width(&self, w: usize) -> <Self as SimdReg<i8>>::ShiftWidth {
        unsafe {
            _mm256_set1_epi32(w as i32)
        }
    }
}
impl SimdShiftWidth<i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn shift_width(&self, w: usize) -> <Self as SimdReg<i16>>::ShiftWidth {
        unsafe {
            _mm256_set1_epi32(w as i32)
        }
    }
}
impl SimdShiftWidth<i32> for Avx2 where Self: SimdReg<i32> {
    #[inline(always)]
    fn shift_width(&self, w: usize) -> <Self as SimdReg<i32>>::ShiftWidth {
        unsafe {
            _mm256_set1_epi32(w as i32)
        }
    }
}
impl SimdShiftWidth<f32> for Avx2 where Self: SimdReg<f32> {
    #[inline(always)]
    fn shift_width(&self, w: usize) -> <Self as SimdReg<f32>>::ShiftWidth {
        unsafe {
            _mm256_set1_epi32(w as i32)
        }
    }
}
impl SimdShiftWidth<f64> for Avx2 where Self: SimdReg<f64> {
    #[inline(always)]
    fn shift_width(&self, w: usize) -> <Self as SimdReg<f64>>::ShiftWidth {
        unsafe {
            _mm256_set1_epi64x(w as i64)
        }
    }
}
impl SimdShl<i16> for Avx2 where Self: SimdReg<i16> {

    #[inline(always)]
    fn shl(&self, v: <Self as SimdReg<i16>>::Reg, w: <Self as SimdReg<i16>>::ShiftWidth) -> <Self as SimdReg<i16>>::Reg {
        let [lo,hi] = <Self as SimdPromote<i16,i32>>::promotion(self,v);

        let regs = [
                                 <Self as SimdShl<i32>>::shl(self,lo,w),
                                 <Self as SimdShl<i32>>::shl(self,hi,w)
                             ];

        <Self as SimdDemote<i32,i16>>::demotion(self,regs)
    }
}
impl SimdShl<i32> for Avx2 where Self: SimdReg<i32> {

    #[inline(always)]
    fn shl(&self, v: <Self as SimdReg<i32>>::Reg, w: <Self as SimdReg<i32>>::ShiftWidth) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_sllv_epi32(v, w)
        }
    }
}
impl SimdShl<i64> for Avx2 where Self: SimdReg<i64> {

    #[inline(always)]
    fn shl(&self, v: <Self as SimdReg<i64>>::Reg, w: <Self as SimdReg<i64>>::ShiftWidth) -> <Self as SimdReg<i64>>::Reg {
        unsafe {
            _mm256_sllv_epi64(v, w)
        }
    }
}
impl SimdShl<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdReinterpret<i32,f32> +
                SimdReinterpret<i32,f32> {

    #[inline(always)]
    fn shl(&self, v: <Self as SimdReg<f32>>::Reg, w: <Self as SimdReg<f32>>::ShiftWidth) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            let vr = <Self as SimdReinterpret<f32,i32>>::reinterpret(self,v);
            let rr = _mm256_sllv_epi32(vr, w);

            <Self as SimdReinterpret<i32,f32>>::reinterpret(self,rr)
        }
    }
}
impl SimdShl<f64> for Avx2 where Self: SimdReg<f64> {

    #[inline(always)]
    fn shl(&self, v: <Self as SimdReg<f64>>::Reg, w: <Self as SimdReg<f64>>::ShiftWidth) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            let vr = <Self as SimdReinterpret<f64,i64>>::reinterpret(self,v);
            let rr = _mm256_sllv_epi64(vr, w);

            <Self as SimdReinterpret<i64,f64>>::reinterpret(self,rr)
        }
    }
}
impl SimdShr<i16> for Avx2 where Self: SimdReg<i16> {

    #[inline(always)]
    fn shr(&self, v: <Self as SimdReg<i16>>::Reg, w: <Self as SimdReg<i16>>::ShiftWidth) -> <Self as SimdReg<i16>>::Reg {
        let [lo,hi] = <Self as SimdPromote<i16,i32>>::promotion(self,v);

        let regs = [
            <Self as SimdShr<i32>>::shr(self,lo,w),
            <Self as SimdShr<i32>>::shr(self,hi,w)
        ];

        <Self as SimdDemote<i32,i16>>::demotion(self,regs)
    }
}
impl SimdShr<i32> for Avx2 where Self: SimdReg<i32> {

    #[inline(always)]
    fn shr(&self, v: <Self as SimdReg<i32>>::Reg, w: <Self as SimdReg<i32>>::ShiftWidth) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_srlv_epi32(v, w)
        }
    }
}
impl SimdShr<i64> for Avx2 where Self: SimdReg<i64> {

    #[inline(always)]
    fn shr(&self, v: <Self as SimdReg<i64>>::Reg, w: <Self as SimdReg<i64>>::ShiftWidth) -> <Self as SimdReg<i64>>::Reg {
        unsafe {
            _mm256_srlv_epi64(v, w)
        }
    }
}
impl SimdShr<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdReinterpret<i32,f32> +
                SimdReinterpret<i32,f32> {

    #[inline(always)]
    fn shr(&self, v: <Self as SimdReg<f32>>::Reg, w: <Self as SimdReg<f32>>::ShiftWidth) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            let vr = <Self as SimdReinterpret<f32,i32>>::reinterpret(self,v);
            let rr = _mm256_srlv_epi32(vr, w);

            <Self as SimdReinterpret<i32,f32>>::reinterpret(self,rr)
        }
    }
}
impl SimdShr<f64> for Avx2 where Self: SimdReg<f64> {

    #[inline(always)]
    fn shr(&self, v: <Self as SimdReg<f64>>::Reg, w: <Self as SimdReg<f64>>::ShiftWidth) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            let vr = <Self as SimdReinterpret<f64,i64>>::reinterpret(self,v);
            let rr = _mm256_srlv_epi64(vr, w);

            <Self as SimdReinterpret<i64,f64>>::reinterpret(self,rr)
        }
    }
}
impl SimdTranspose<i8,1> for Avx2 {
    #[inline(always)]
    fn transpose<'a, const N: usize, const M: usize>(v: [Self::Reg; 1]) -> [Self::Reg; 1] {
        v
    }
}
impl SimdTranspose<i16,1> for Avx2 {
    #[inline(always)]
    fn transpose<'a, const N: usize, const M: usize>(v: [Self::Reg; 1]) -> [Self::Reg; 1] {
        v
    }
}
impl SimdTranspose<i32,1> for Avx2 {
    #[inline(always)]
    fn transpose<'a, const N: usize, const M: usize>(v: [Self::Reg; 1]) -> [Self::Reg; 1] {
        v
    }
}
impl SimdTranspose<i32,2> for Avx2 {
    #[inline]
    fn transpose<'a, const N: usize, const M: usize>(v: [Self::Reg; 2]) -> [Self::Reg; 2] {
        let v0 = v[0];
        let v1 = v[1];

        unsafe {
            let lo = _mm256_unpacklo_epi32(v0,v1);
            let hi = _mm256_unpackhi_epi32(v0,v1);

            [lo,hi]
        }
    }
}
impl SimdTranspose<f32,1> for Avx2 {
    #[inline(always)]
    fn transpose<'a, const N: usize, const M: usize>(v: [Self::Reg; 1]) -> [Self::Reg; 1] {
        v
    }
}
impl SimdTranspose<f32,2> for Avx2 {
    #[inline]
    fn transpose<'a, const N: usize, const M: usize>(v: [Self::Reg; 2]) -> [Self::Reg; 2] {
        let v0 = v[0];
        let v1 = v[1];

        unsafe {
            let lo = _mm256_unpacklo_ps (v0,v1);
            let hi = _mm256_unpackhi_ps (v0,v1);

            [lo,hi]
        }
    }
}
impl SimdTranspose<f64,1> for Avx2 {
    #[inline(always)]
    fn transpose<'a, const N: usize, const M: usize>(v: [Self::Reg; 1]) -> [Self::Reg; 1] {
        v
    }
}
impl SimdHSum<i32> for Avx2 {

    #[inline]
    fn hsum(&self, v: Self::Reg) -> i32 {
        unsafe {
            let lo128 = _mm256_castsi256_si128(v);
            let hi128 = _mm256_extracti128_si256(v,1);

            let sum128 = _mm_add_epi32(lo128, hi128);

            let hi64 = _mm_srli_si128(sum128, 8);

            let sum64 = _mm_add_epi32(sum128, hi64);

            let hi32 = _mm_srli_si128(sum64, 4);
            let sum32 = _mm_add_epi32(sum64, hi32);

            _mm_cvtsi128_si32(sum32)
        }
    }
}
impl SimdHSum<f32> for Avx2 {

    #[inline]
    fn hsum(&self, v: Self::Reg) -> f32 {
        unsafe {
            let lo128 = _mm256_castps256_ps128(v);
            let hi128 = _mm256_extractf128_ps(v,1);

            let sum128 = _mm_add_ps(lo128, hi128);

            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);

            let hi32 = _mm_shuffle_ps(sum64, sum64, 0x1);
            let sum32 = _mm_add_ss(sum64, hi32);

            _mm_cvtss_f32(sum32)
        }
    }
}
impl SimdHSum<f64> for Avx2 {

    #[inline]
    fn hsum(&self, v: Self::Reg) -> f64 {
        unsafe {
            let lo128 = _mm256_castpd256_pd128(v);
            let hi128 = _mm256_extractf128_pd(v,1);

            let sum128 = _mm_add_pd(lo128, hi128);

            let hi64 = _mm_unpackhi_pd(sum128, sum128);
            let sum64 = _mm_add_sd(sum128, hi64);

            _mm_cvtsd_f64(sum64)
        }
    }
}
impl SimdMulAdd<i32,i32,i32> for Avx2
    where Self: SimdMul<i32,i32,i32> +
                SimdAdd<i32,i32,i32> {
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<i32>>::Reg,
               acc: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        let [r] = <Self as SimdMul<i32,i32,i32>>::mul(self,l,r);

        <Self as SimdAdd<i32,i32,i32>>::add(self,acc,r)
    }
}
impl SimdMulAdd<f32,f32,f32> for Avx2 {
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<f32>>::Reg,
               acc:<Self as SimdReg<f32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            _mm256_fmadd_ps(l,r,acc)
        }
    }
}
impl SimdMulAdd<f64,f64,f64> for Avx2 {
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<f64>>::Reg,
               acc: <Self as SimdReg<f64>>::Reg
    ) -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            _mm256_fmadd_pd(l, r, acc)
        }
    }
}
impl SimdZero<i8> for Avx2 where Self: SimdReg<i8> {
    fn zero() -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            _mm256_setzero_si256()
        }
    }
}
impl SimdZero<i16> for Avx2 where Self: SimdReg<i16> {
    fn zero() -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            _mm256_setzero_si256()
        }
    }
}
impl SimdZero<i32> for Avx2 where Self: SimdReg<i32> {
    fn zero() -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_setzero_si256()
        }
    }
}
impl SimdZero<f32> for Avx2 where Self: SimdReg<f32> {
    fn zero() -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            _mm256_setzero_ps()
        }
    }
}
impl SimdZero<f64> for Avx2 where Self: SimdReg<f64> {
    fn zero() -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            _mm256_setzero_pd()
        }
    }
}
impl<SL,SR,SO> SimdDot<SL,SR,SO> for Avx2
    where Self: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdAdd<SO,SO,SO> + SimdHSum<SO> +
                SimdLoad<SL> + SimdLoad<SR> + SimdZero<SO> +
                SimdLanes<SL> + SimdLanes<SR> + SimdLanes<SO> +
                SimdMul<SL,SR,SO>,
          PartialDotAcc<SL,SR,SO,Self>: SimdPartialDot<SL,SR,SO,Self>,
          SL: Clone + Copy,
          SR: Clone + Copy,
          SO: From<SL> + From<SR> + Mul<SO,Output=SO> + AddAssign {
    #[inline(always)]
    fn dot<'a, const N: usize>(&self, l: &VectorView<'a, SL, N>, r: &VectorView<'a, SR, N>) -> SO {
        let mut acc = PartialDotAcc::new();

        let mut i = 0;
        let mut pa = l.as_ref().as_ptr();
        let mut pb = r.as_ref().as_ptr();

        unsafe {
            while i + <Self as SimdLanes<SL>>::LANES <= N {
                let lr = self.load(pa);
                let rr = self.load(pb);

                acc.partial_dot(self,lr,rr);

                pa = pa.add(<Self as SimdLanes<SL>>::LANES);
                pb = pb.add(<Self as SimdLanes<SR>>::LANES);

                i += <Self as SimdLanes<SL>>::LANES;
            }

            let acc = acc.finalize();

            let mut sum = <Self as SimdHSum<SO>>::hsum(self,acc.fold(self));

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                for _ in i..N {
                    sum += SO::from(*pa) * SO::from(*pb);
                    pa = pa.add(1);
                    pb = pb.add(1);
                }
            }

            sum
        }
    }
}
impl<SL,SR,SO> SimdMatVec<SL,SR,SO> for Avx2
    where Self: SimdZero<SO> +
                SimdAdd<SO,SO,SO> +
                SimdMul<SL,SR,SO> +
                SimdHSum<SO> +
                SimdLanes<SL> +
                SimdLoad<SL> +
                SimdLoad<SR>,
                SL: Add<SR,Output = SO> + Clone + Copy,
                SR: Clone + Copy,
                SO: From<SL> + From<SR> + Mul<SO,Output=SO> + AddAssign,
                PartialDotAcc<SL,SR,SO,Self>: SimdPartialDot<SL,SR,SO,Self>,
                <Self as SimdReg<SO>>::Reg: Copy {

    fn matvec<'a, const N: usize, const K: usize>(&self, l: &MatrixView<'a, SL, N, K>, r: &VectorView<'a, SR, K>, o: &mut OwnedVector<SO, N>) {
        unsafe {
            for i in 0..N {
                let mut acc = PartialDotAcc::<SL,SR,SO,Self>::new();

                for k in (0..(K - K % <Self as SimdLanes<SL>>::LANES)).step_by(<Self as SimdLanes<SL>>::LANES) {
                    let lr = self.load(l.row(i).as_ref().as_ptr().add(k));
                    let rr = self.load(r.as_ref().as_ptr().add(k));

                    acc.partial_dot(self,lr,rr);
                }

                let acc = acc.finalize();
                let mut acc = <Self as SimdHSum<SO>>::hsum(self,acc.fold(self));

                if K % <Self as SimdLanes<SL>>::LANES != 0 {
                    for k in (K - K % <Self as SimdLanes<SL>>::LANES)..K {
                        acc += SO::from(l[i][k]) * SO::from(r[k]);
                    }
                }

                o[i] = acc;
            }
        }
    }
}
derive_matmul! { Avx2,i8,i8,i32 }
derive_matmul! { Avx2,i16,i16,i32 }
derive_matmul! { Avx2,i32,i32,i32 }
derive_matmul! { Avx2,f32,f32,f32 }
derive_matmul! { Avx2,f64,f64,f64 }
impl<SL,SR,SO> SimdVMat<SL,SR,SO> for Avx2
    where Self: SimdDot<SL,SR,SO> +
                SimdLanes<SL> {
    #[inline]
    fn vmat<'a, const M: usize, const K: usize>(&self, l: &VectorView<'a, SL, K>, r: &ColumnMajorMatrix<'a, SR, K, M>, o: &mut OwnedVector<SO, M>) {
        for i in 0..M {
            let s = self.dot(l,&r.col(i).into());

            o[i] = s;
        }
    }
}
impl SimdPartialDot<i8,i8,i32,Avx2> for PartialDotAcc<i8,i8,i32,<Avx2 as SimdMul<i8,i8,i32>>::Output>
    where Avx2: SimdReg<i8> +
                SimdReg<i32> +
                SimdMul<i8,i8,i32> +
                SimdAdd<i32,i32,i32> + Backend + Sized {
    #[inline(always)]
    fn new() -> Self {
        PartialDotAcc {
            acc: [<Avx2 as SimdZero<i32>>::zero();4],
            l:PhantomData::<i8>,
            r:PhantomData::<i8>,
            o:PhantomData::<i32>
        }
    }

    #[inline(always)]
    fn partial_dot(&mut self, backend: &Avx2, l: <Avx2 as SimdReg<i8>>::Reg, r: <Avx2 as SimdReg<i8>>::Reg) {
        let regs = <Avx2 as SimdMul<i8,i8,i32>>::mul(backend,l,r);

        self.acc[0] = <Avx2 as SimdAdd<i32,i32,i32>>::add(backend,self.acc[0], regs[0]);
        self.acc[1] = <Avx2 as SimdAdd<i32,i32,i32>>::add(backend,self.acc[1], regs[1]);
        self.acc[2] = <Avx2 as SimdAdd<i32,i32,i32>>::add(backend,self.acc[2], regs[2]);
        self.acc[3] = <Avx2 as SimdAdd<i32,i32,i32>>::add(backend,self.acc[3], regs[3]);
    }

    #[inline(always)]
    fn finalize(self) -> <Avx2 as SimdMul<i8,i8,i32>>::Regs {
        Regs::new(self.acc)
    }
}
impl SimdPartialDot<i16,i16,i32,Avx2> for PartialDotAcc<i16,i16,i32,<Avx2 as SimdMul<i16,i16,i32>>::Output>
    where Avx2: SimdReg<i16> +
                SimdReg<i32> +
                SimdMul<i16,i16,i32> +
                SimdAdd<i32,i32,i32> + Backend + Sized {
    #[inline(always)]
    fn new() -> Self {
        PartialDotAcc {
            acc: [<Avx2 as SimdZero<i32>>::zero();2],
            l:PhantomData::<i16>,
            r:PhantomData::<i16>,
            o:PhantomData::<i32>
        }
    }

    #[inline(always)]
    fn partial_dot(&mut self, backend: &Avx2, l: <Avx2 as SimdReg<i16>>::Reg, r: <Avx2 as SimdReg<i16>>::Reg) {
        let regs = <Avx2 as SimdMul<i16,i16,i32>>::mul(backend,l,r);

        self.acc[0] = <Avx2 as SimdAdd<i32,i32,i32>>::add(backend,self.acc[0], regs[0]);
        self.acc[1] = <Avx2 as SimdAdd<i32,i32,i32>>::add(backend,self.acc[1], regs[1]);
    }

    #[inline(always)]
    fn finalize(self) -> <Avx2 as SimdMul<i16,i16,i32>>::Regs {
        Regs::new(self.acc)
    }
}
impl SimdPartialDot<i32,i32,i32,Avx2> for PartialDotAcc<i32,i32,i32,<Avx2 as SimdMul<i32,i32,i32>>::Output>
    where Avx2: SimdReg<i32> +
                SimdMul<i32,i32,i32> +
                SimdAdd<i32,i32,i32> + Backend + Sized {
    #[inline(always)]
    fn new() -> Self {
        PartialDotAcc {
            acc: [<Avx2 as SimdZero<i32>>::zero();1],
            l:PhantomData::<i32>,
            r:PhantomData::<i32>,
            o:PhantomData::<i32>
        }
    }

    #[inline(always)]
    fn partial_dot(&mut self, backend: &Avx2, l: <Avx2 as SimdReg<i32>>::Reg, r: <Avx2 as SimdReg<i32>>::Reg) {
        let regs = <Avx2 as SimdMul<i32,i32,i32>>::mul(backend,l,r);

        self.acc[0] = <Avx2 as SimdAdd<i32,i32,i32>>::add(backend,self.acc[0], regs[0]);
    }

    #[inline(always)]
    fn finalize(self) -> <Avx2 as SimdMul<i32,i32,i32>>::Regs {
        Regs::new(self.acc)
    }
}
impl SimdPartialDot<f32,f32,f32,Avx2> for PartialDotAcc<f32,f32,f32,<Avx2 as SimdMul<f32,f32,f32>>::Output>
where Avx2: SimdReg<f32> +
SimdMul<f32,f32,f32> +
SimdAdd<f32,f32,f32> + Backend + Sized {
    #[inline(always)]
    fn new() -> Self {
        PartialDotAcc {
            acc: [<Avx2 as SimdZero<f32>>::zero();1],
            l:PhantomData::<f32>,
            r:PhantomData::<f32>,
            o:PhantomData::<f32>
        }
    }

    #[inline(always)]
    fn partial_dot(&mut self, backend: &Avx2, l: <Avx2 as SimdReg<f32>>::Reg, r: <Avx2 as SimdReg<f32>>::Reg) {
        let regs = <Avx2 as SimdMul<f32,f32,f32>>::mul(backend,l,r);

        self.acc[0] = <Avx2 as SimdAdd<f32,f32,f32>>::add(backend,self.acc[0], regs[0]);
    }

    #[inline(always)]
    fn finalize(self) -> <Avx2 as SimdMul<f32,f32,f32>>::Regs {
        Regs::new(self.acc)
    }
}
impl SimdPartialDot<f64,f64,f64,Avx2> for PartialDotAcc<f64,f64,f64,<Avx2 as SimdMul<f64,f64,f64>>::Output>
    where Avx2: SimdReg<f64> +
                SimdMul<f64,f64,f64> +
                SimdAdd<f64,f64,f64> + Backend + Sized {
    #[inline(always)]
    fn new() -> Self {
        PartialDotAcc {
            acc: [<Avx2 as SimdZero<f64>>::zero();1],
            l:PhantomData::<f64>,
            r:PhantomData::<f64>,
            o:PhantomData::<f64>
        }
    }

    #[inline(always)]
    fn partial_dot(&mut self, backend: &Avx2, l: <Avx2 as SimdReg<f64>>::Reg, r: <Avx2 as SimdReg<f64>>::Reg) {
        let regs = <Avx2 as SimdMul<f64,f64,f64>>::mul(backend,l,r);

        self.acc[0] = <Avx2 as SimdAdd<f64,f64,f64>>::add(backend,self.acc[0], regs[0]);
    }

    #[inline(always)]
    fn finalize(self) -> <Avx2 as SimdMul<f64,f64,f64>>::Regs {
        Regs::new(self.acc)
    }
}