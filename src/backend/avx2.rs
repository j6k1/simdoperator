//! Implementation of SIMD Operations Using AVX2

use std::arch::x86_64::{__m256, __m256d, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi8, _mm256_add_pd, _mm256_add_ps, _mm256_and_pd, _mm256_and_ps, _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_blendv_pd, _mm256_blendv_ps, _mm256_castpd256_pd128, _mm256_castpd_si256, _mm256_castps256_ps128, _mm256_castps_si256, _mm256_castsi256_pd, _mm256_castsi256_ps, _mm256_castsi256_si128, _mm256_cmp_pd, _mm256_cmp_ps, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpeq_epi8, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cmpgt_epi8, _mm256_cvtepi16_epi32, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi16, _mm256_cvtps_epi32, _mm256_extractf128_pd, _mm256_extractf128_ps, _mm256_extracti128_si256, _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_mul_pd, _mm256_mul_ps, _mm256_mullo_epi16, _mm256_mullo_epi32, _mm256_or_si256, _mm256_packs_epi16, _mm256_packs_epi32, _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_set1_pd, _mm256_set1_ps, _mm256_setzero_pd, _mm256_setzero_ps, _mm256_setzero_si256, _mm256_sllv_epi32, _mm256_sllv_epi64, _mm256_srlv_epi32, _mm256_srlv_epi64, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_sub_epi8, _mm256_sub_pd, _mm256_sub_ps, _mm256_unpackhi_epi32, _mm256_unpackhi_ps, _mm256_unpacklo_epi32, _mm256_unpacklo_ps, _mm256_xor_si256, _mm_add_epi32, _mm_add_pd, _mm_add_ps, _mm_add_sd, _mm_add_ss, _mm_cvtsd_f64, _mm_cvtsi128_si32, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps, _mm_srli_si128, _mm_unpackhi_pd, _CMP_EQ_OQ, _CMP_GT_OQ, _mm256_srav_epi32, _mm256_srav_epi64, _mm256_load_si256, _mm256_load_ps, _mm256_load_pd, _mm256_store_si256, _mm256_store_ps, _mm256_store_pd};
use std::mem;
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Mul};
use std::ptr::copy_nonoverlapping;
use crate::backend::common::{mm_shuffle, Backend, Regs, SimdAddOperator, Extent, AddOperator, SimdQueryShrOperator, ShrOperator, SimdQueryAddOperator, DemoteOperator, SimdQueryDemoteOperator};
use crate::traits::{SimdCols, SimdDot, SimdHSum, SimdLanes, SimdLoad, SimdMask, SimdMatMul, SimdMatVec, SimdMulAdd, SimdPartialDot, SimdReg, SimdRows, SimdStore, SimdTranspose, SimdVMat, SimdZero, SimdAdd, SimdSub, SimdMul, SimdPromote, SimdScalarMul, SimdSplat, SimdBitOr, SimdBitAnd, SimdReinterpret, SimdBitXor, SimdBitNot, BitsBitAnd, BitsBitOr, BitsBitXor, SimdShl, SimdShr, SimdDemote, SimdConvert, FoldRegs, SimdShiftWidth, NativeBackend, SimdDotKernel, SimdMatMulKernel, SimdAddVector, SimdQueryVector, SimdMapVMat, Mapper, SimdResolveBinaryOperator, ToSimdBinaryOperator, SimdResolvePatrialBinaryOperator, SimdBinaryOperator, SimdCombinationQueryVector, SimdLoadSeq, SimdStoreSeq, SimdLoadAligned, SimdStoreAligned, SimdRegBuf, SimdLinearCombination, SimdLinearCombinationKernel};
use crate::{ColumnMajorMatrix, MatrixMutView, MatrixView, OwnedVector, VectorMutView, VectorView};
use crate::error::InstantiationError;
use crate::traits::kernel::SimdCombinationQueryVectorKernel;

/// Avx2 Backend
#[derive(Clone)]
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
    const COLS: usize = 4;
}
impl SimdCols<i16> for Avx2 {
    const COLS: usize = 2;
}
impl SimdCols<i32> for Avx2 {
    const COLS: usize = 2;
}
impl SimdCols<f32> for Avx2 {
    const COLS: usize = 2;
}
impl SimdCols<f64> for Avx2 {
    const COLS: usize = 1;
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
const BS: usize = 32;
#[repr(align(32))]
pub struct AlignedAvx2Buffer([MaybeUninit<u8>; BS]);
impl AsRef<[MaybeUninit<u8>]> for AlignedAvx2Buffer {
    fn as_ref(&self) -> &[MaybeUninit<u8>] {
        &self.0
    }
}
impl AsMut<[MaybeUninit<u8>]> for AlignedAvx2Buffer {
    fn as_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        &mut self.0
    }
}
impl SimdRegBuf for Avx2 {
    type Buffer = AlignedAvx2Buffer;

    fn uninit() -> Self::Buffer {
        let buf: [MaybeUninit<u8>; BS] = unsafe { MaybeUninit::<[MaybeUninit<u8>; BS]>::uninit().assume_init() };

        AlignedAvx2Buffer(buf)
    }

    fn initialized<T>(input: &[T], rem: usize) -> Self::Buffer {
        let mut buf: [MaybeUninit<u8>; BS] = unsafe { MaybeUninit::<[MaybeUninit<u8>; BS]>::uninit().assume_init() };

        let ptr = buf.as_mut_ptr() as *mut u8;

        unsafe {
            copy_nonoverlapping(input.as_ptr(), buf.as_mut_ptr() as *mut u8 as *mut T, rem);
            std::ptr::write_bytes(ptr.add(rem * mem::size_of::<T>()), 0, BS - rem * mem::size_of::<T>());
        }

        AlignedAvx2Buffer(buf)
    }
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
impl SimdLoadAligned<i8> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const i8) -> Self::Reg {
        unsafe {
            _mm256_load_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoadAligned<i16> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const i16) -> Self::Reg {
        unsafe {
            _mm256_load_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoadAligned<i32> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const i32) -> Self::Reg {
        unsafe {
            _mm256_load_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoadAligned<i64> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const i64) -> Self::Reg {
        unsafe {
            _mm256_load_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoadAligned<u32> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const u32) -> Self::Reg {
        unsafe {
            _mm256_load_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoadAligned<u64> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const u64) -> Self::Reg {
        unsafe {
            _mm256_load_si256(ptr as *const __m256i)
        }
    }
}
impl SimdLoadAligned<f32> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const f32) -> Self::Reg {
        unsafe {
            _mm256_load_ps(ptr)
        }
    }
}
impl SimdLoadAligned<f64> for Avx2 {
    #[inline(always)]
    unsafe fn load_aligned(&self, ptr: *const f64) -> Self::Reg {
        unsafe {
            _mm256_load_pd(ptr)
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
impl SimdStoreAligned<i8> for Avx2 {
    #[inline(always)]
    unsafe fn store_aligned(&self, ptr: *mut i8, a: Self::Reg) {
        unsafe {
            _mm256_store_si256(ptr as *mut __m256i, a);
        }
    }
}
impl SimdStoreAligned<i16> for Avx2 {
    #[inline(always)]
    unsafe fn store_aligned(&self, ptr: *mut i16, a: Self::Reg) {
        unsafe {
            _mm256_store_si256(ptr as *mut __m256i, a);
        }
    }
}
impl SimdStoreAligned<i32> for Avx2 {
    #[inline(always)]
    unsafe fn store_aligned(&self, ptr: *mut i32, a: Self::Reg) {
        unsafe {
            _mm256_store_si256(ptr as *mut __m256i, a);
        }
    }
}
impl SimdStoreAligned<f32> for Avx2 {
    #[inline(always)]
    unsafe fn store_aligned(&self, ptr: *mut f32, a: Self::Reg) {
        unsafe {
            _mm256_store_ps(ptr, a);
        }
    }
}
impl SimdStoreAligned<f64> for Avx2 {
    #[inline(always)]
    unsafe fn store_aligned(&self, ptr: *mut f64, a: Self::Reg) {
        unsafe {
            _mm256_store_pd(ptr, a);
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
    type Output = Regs<<Self as SimdReg<i16>>::Reg,2>;

    #[inline(always)]
    fn promotion(&self, reg: <Self as SimdReg<i8>>::Reg) -> <Self as SimdPromote<i8,i16>>::Output {
        unsafe {
            let lo = _mm256_castsi256_si128(reg);
            let hi = _mm256_extracti128_si256(reg,1);

            Regs::new([_mm256_cvtepi8_epi16(lo),_mm256_cvtepi8_epi16(hi)])
        }
    }
}
impl SimdPromote<i16,i32> for Avx2 where Self: SimdReg<i16> + SimdReg<i32> {
    type Output = Regs<<Self as SimdReg<i32>>::Reg,2>;

    #[inline(always)]
    fn promotion(&self, reg: <Self as SimdReg<i16>>::Reg) -> <Self as SimdPromote<i16,i32>>::Output {
        unsafe {
            let lo = _mm256_castsi256_si128(reg);
            let hi = _mm256_extracti128_si256(reg,1);

            let lo32 = _mm256_cvtepi16_epi32(lo);
            let hi32 = _mm256_cvtepi16_epi32(hi);

            Regs::new([lo32,hi32])
        }
    }
}
impl SimdPromote<i8,i32> for Avx2
    where Self: SimdReg<i8> +
                SimdReg<i16> +
                SimdReg<i32> +
                SimdPromote<i8,i16> +
                SimdPromote<i16,i32> {
    type Output = Regs<<Self as SimdReg<i32>>::Reg,4>;

    #[inline(always)]
    fn promotion(&self, reg: <Self as SimdReg<i8>>::Reg) -> <Self as SimdPromote<i8,i32>>::Output {
        let regs = <Self as SimdPromote<i8,i16>>::promotion(self, reg);

        let &[lo,hi] = regs.as_ref();

        let lo32 = <Self as SimdPromote<i16,i32>>::promotion(self, lo);
        let hi32 = <Self as SimdPromote<i16,i32>>::promotion(self, hi);

        let &[lo_lo,lo_hi] = lo32.as_ref();
        let &[hi_lo,hi_hi] = hi32.as_ref();

        Regs::new([lo_lo,lo_hi,hi_lo,hi_hi])
    }
}
impl SimdDemote<i32,i16> for Avx2 where Self: SimdReg<i32> + SimdReg<i16> {
    type Input = Regs<<Self as SimdReg<i32>>::Reg,2>;

    #[inline(always)]
    fn demotion(&self, reg: Regs<<Self as SimdReg<i32>>::Reg,2>) -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            let &[lo32,hi32] = reg.as_ref();

            _mm256_permute4x64_epi64(_mm256_packs_epi32(lo32, hi32),mm_shuffle(3,1,2,0))
        }
    }
}
impl SimdDemote<i16,i8> for Avx2 where Self: SimdReg<i16> + SimdReg<i8> {
    type Input = Regs<<Self as SimdReg<i16>>::Reg,2>;
    #[inline(always)]
    fn demotion(&self, reg: Regs<<Self as SimdReg<i16>>::Reg,2>) -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            let &[lo_16,hi_16] = reg.as_ref();

            _mm256_permute4x64_epi64(_mm256_packs_epi16(lo_16, hi_16),mm_shuffle(3,1,2,0))
        }
    }
}
impl SimdDemote<i32,i8> for Avx2
    where Self: SimdReg<i32> +
                SimdReg<i16> +
                SimdReg<i8> +
                SimdDemote<i32,i16> +
                SimdDemote<i16,i8> {
    type Input = Regs<<Self as SimdReg<i32>>::Reg,4>;

    #[inline(always)]
    fn demotion(&self, reg: Regs<<Self as SimdReg<i32>>::Reg,4>) -> <Self as SimdReg<i8>>::Reg {
        let &[lo_lo,lo_hi,hi_lo,hi_hi] = reg.as_ref();

        let lo = Regs::new([lo_lo,lo_hi]);
        let hi = Regs::new([hi_lo,hi_hi]);

        let lo_16 = <Self as SimdDemote<i32,i16>>::demotion(self,lo);
        let hi_16 = <Self as SimdDemote<i32,i16>>::demotion(self,hi);

        <Self as SimdDemote<i16,i8>>::demotion(self,Regs::new([lo_16,hi_16]))
    }
}
impl SimdConvert<i16,f32> for Avx2 where Self: SimdReg<i16> + SimdReg<f32> {
    type Input = Regs<<Self as SimdReg<i16>>::Reg,1>;
    type Output = Regs<<Self as SimdReg<f32>>::Reg,2>;

    #[inline(always)]
    fn convert(&self, reg: Regs<<Self as SimdReg<i16>>::Reg,1>) -> <Self as SimdConvert<i16,f32>>::Output {
        unsafe {
            let &[reg] = reg.as_ref();

            let lo_i16 = _mm256_castsi256_si128(reg);
            let hi_i16 = _mm256_extracti128_si256(reg, 1);

            let lo_i32 = _mm256_cvtepi16_epi32(lo_i16);
            let hi_i32 = _mm256_cvtepi16_epi32(hi_i16);

            let lo_f32 = _mm256_cvtepi32_ps(lo_i32);
            let hi_f32 = _mm256_cvtepi32_ps(hi_i32);

            Regs::new([lo_f32,hi_f32])
        }
    }
}
impl SimdConvert<i32,f32> for Avx2 where Self: SimdReg<i32> + SimdReg<f32> {
    type Input = Regs<<Self as SimdReg<i32>>::Reg,1>;
    type Output = Regs<<Self as SimdReg<f32>>::Reg,1>;

    #[inline(always)]
    fn convert(&self, reg: Regs<<Self as SimdReg<i32>>::Reg,1>) -> <Self as SimdConvert<i32,f32>>::Output {
        unsafe {
            let &[reg] = reg.as_ref();

            Regs::new([_mm256_cvtepi32_ps(reg)])
        }
    }
}
impl SimdConvert<f32,i32> for Avx2 where Self: SimdReg<i32> + SimdReg<f32> {
    type Input = Regs<<Self as SimdReg<f32>>::Reg,1>;
    type Output = Regs<<Self as SimdReg<i32>>::Reg,1>;

    #[inline(always)]
    fn convert(&self, reg: Regs<<Self as SimdReg<f32>>::Reg,1>) -> <Self as SimdConvert<f32,i32>>::Output {
        unsafe {
            let &[reg] = reg.as_ref();

            Regs::new([_mm256_cvtps_epi32(reg)])
        }
    }
}
impl SimdConvert<i8,f32> for Avx2
    where Self: SimdReg<i8> +
                SimdReg<i16> +
                SimdReg<i32> +
                SimdReg<f32> +
                SimdPromote<i8,i16> +
                SimdPromote<i16,i32> +
                SimdConvert<i32,f32> {
    type Input = Regs<<Self as SimdReg<i8>>::Reg,1>;
    type Output = Regs<<Self as SimdReg<f32>>::Reg,4>;

    #[inline(always)]
    fn convert(&self, reg: Regs<<Self as SimdReg<i8>>::Reg,1>) -> <Self as SimdConvert<i8,f32>>::Output {
        let &[reg] = reg.as_ref();

        let &[lo16,hi16] = <Self as SimdPromote<i8,i16>>::promotion(self,reg).as_ref();

        let &[lo_lo_i32,lo_hi_i32] = <Self as SimdPromote<i16,i32>>::promotion(self,lo16).as_ref();
        let &[hi_lo_i32,hi_hi_i32] = <Self as SimdPromote<i16,i32>>::promotion(self,hi16).as_ref();

        Regs::new([
            <Self as SimdConvert<i32,f32>>::convert(self,Regs::new([lo_lo_i32])).as_ref()[0],
            <Self as SimdConvert<i32,f32>>::convert(self,Regs::new([lo_hi_i32])).as_ref()[0],
            <Self as SimdConvert<i32,f32>>::convert(self,Regs::new([hi_lo_i32])).as_ref()[0],
            <Self as SimdConvert<i32,f32>>::convert(self,Regs::new([hi_hi_i32])).as_ref()[0],
        ])
    }
}
impl SimdConvert<f32,i8> for Avx2
    where Self: SimdReg<i8> +
                SimdReg<i16> +
                SimdReg<i32> +
                SimdReg<f32> +
                SimdPromote<i8,i16> +
                SimdPromote<i16,i32> +
                SimdConvert<f32,i32> {
    type Input = Regs<<Self as SimdReg<f32>>::Reg,4>;
    type Output = Regs<<Self as SimdReg<i8>>::Reg,1>;

    #[inline(always)]
    fn convert(&self, reg: Regs<<Self as SimdReg<f32>>::Reg,4>) -> <Self as SimdConvert<f32,i8>>::Output {
        let &[lo_lo_f32,lo_hi_f32,hi_lo_f32,hi_hi_f32] = reg.as_ref();

        let &[lo_lo_i32] = <Self as SimdConvert<f32,i32>>::convert(self,Regs::new([lo_lo_f32])).as_ref();
        let &[lo_hi_i32] = <Self as SimdConvert<f32,i32>>::convert(self,Regs::new([lo_hi_f32])).as_ref();
        let &[hi_lo_i32] = <Self as SimdConvert<f32,i32>>::convert(self,Regs::new([hi_lo_f32])).as_ref();
        let &[hi_hi_i32] = <Self as SimdConvert<f32,i32>>::convert(self,Regs::new([hi_hi_f32])).as_ref();

        let lo_i16 = <Self as SimdDemote<i32,i16>>::demotion(self,Regs::new([lo_lo_i32,lo_hi_i32]));
        let hi_i16 = <Self as SimdDemote<i32,i16>>::demotion(self,Regs::new([hi_lo_i32,hi_hi_i32]));

        Regs::new([<Self as SimdDemote<i16,i8>>::demotion(self,Regs::new([lo_i16,hi_i16]))])
    }
}
impl SimdAdd<i8,i8,i8> for Avx2 where Self: SimdReg<i8> {
    type Lhs = <Self as SimdReg<i8>>::Reg;
    type Rhs = <Self as SimdReg<i8>>::Reg;
    type Output = <Self as SimdReg<i8>>::Reg;

    #[target_feature(enable = "avx2")]
    unsafe fn add(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<i8>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        _mm256_add_epi8(l, r)
    }
}
impl SimdAdd<i16,i16,i16> for Avx2 where Self: SimdReg<i16> {
    type Lhs = <Self as SimdReg<i16>>::Reg;
    type Rhs = <Self as SimdReg<i16>>::Reg;
    type Output = <Self as SimdReg<i16>>::Reg;
    #[target_feature(enable = "avx2")]
    unsafe fn add(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<i16>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        _mm256_add_epi16(l, r)
    }
}
impl SimdAdd<i32,i32,i32> for Avx2 where Self: SimdReg<i32> {
    type Lhs = <Self as SimdReg<i32>>::Reg;
    type Rhs = <Self as SimdReg<i32>>::Reg;
    type Output = <Self as SimdReg<i32>>::Reg;

    #[target_feature(enable = "avx2")]
    unsafe fn add(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        _mm256_add_epi32(l, r)
    }
}
impl SimdAdd<f32,f32,f32> for Avx2 where Self: SimdReg<f32> {
    type Lhs = <Self as SimdReg<f32>>::Reg;
    type Rhs = <Self as SimdReg<f32>>::Reg;
    type Output = <Self as SimdReg<f32>>::Reg;

    #[target_feature(enable = "avx2")]
    unsafe fn add(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<f32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        _mm256_add_ps(l, r)
    }
}
impl SimdAdd<f64,f64,f64> for Avx2 where Self: SimdReg<f64> {
    type Lhs = <Self as SimdReg<f64>>::Reg;
    type Rhs = <Self as SimdReg<f64>>::Reg;
    type Output = <Self as SimdReg<f64>>::Reg;

    #[target_feature(enable = "avx2")]
    unsafe fn add(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<f64>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        _mm256_add_pd(l, r)
    }
}
impl SimdAdd<i8,i16,i16> for Avx2
    where Self: SimdReg<i8> +
                SimdReg<i16> +
                SimdAdd<i16,i16,i16> +
                SimdPromote<i8,i16> {
    type Lhs = Regs<<Self as SimdReg<i8>>::Reg,1>;
    type Rhs = Regs<<Self as SimdReg<i16>>::Reg,2>;
    type Output = Regs<<Self as SimdReg<i16>>::Reg,2>;

    #[target_feature(enable = "avx2")]
    unsafe fn add(&self, l: Regs<<Self as SimdReg<i8>>::Reg,1>, r: Regs<<Self as SimdReg<i16>>::Reg,2>)
        -> Regs<<Self as SimdReg<i16>>::Reg,2> {
        let &[l] = l.as_ref();
        let &[la,lb] = <Self as SimdPromote<i8,i16>>::promotion(self,l).as_ref();

        let &[ra,rb] = r.as_ref();

        unsafe {
            Regs::new([
                <Self as SimdAdd<i16,i16,i16>>::add(self,la,ra),
                <Self as SimdAdd<i16,i16,i16>>::add(self,lb,rb)
            ])
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
    type Output = Regs<<Self as SimdReg<i32>>::Reg,4>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<i8>>::Reg)
        -> Regs<<Self as SimdReg<i32>>::Reg,4> {
        let &[lo,hi] = <Self as SimdPromote<i8,i16>>::promotion(self,l).as_ref();
        let &[lo_r,hi_r] = <Self as SimdPromote<i8,i16>>::promotion(self,r).as_ref();

        let &[lo32_lo,lo32_hi] = <Self as SimdMul<i16,i16,i32>>::mul(self,lo,lo_r).as_ref();
        let &[hi32_lo,hi32_hi] = <Self as SimdMul<i16,i16,i32>>::mul(self,hi,hi_r).as_ref();

        Regs::new([lo32_lo,lo32_hi,hi32_lo,hi32_hi])
    }
}
impl SimdMul<i16,i16,i32> for Avx2 where Self: SimdReg<i16> + SimdReg<i32> {
    type Output = Regs<<Self as SimdReg<i32>>::Reg,2>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<i16>>::Reg)
        -> Regs<<Self as SimdReg<i32>>::Reg,2> {
        unsafe {
            let prod16 = _mm256_mullo_epi16(l, r);

            <Self as SimdPromote<i16,i32>>::promotion(self,prod16)
        }
    }
}
impl SimdMul<i32,i32,i32> for Avx2 where Self: SimdReg<i32> {
    type Output = Regs<<Self as SimdReg<i32>>::Reg,1>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<i32>>::Reg) -> Regs<<Self as SimdReg<i32>>::Reg,1> {
        unsafe {
            Regs::new([_mm256_mullo_epi32(l, r)])
        }
    }
}
impl SimdMul<f32,f32,f32> for Avx2 where Self: SimdReg<f32> {
    type Output = Regs<<Self as SimdReg<f32>>::Reg,1>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<f32>>::Reg) -> Regs<<Self as SimdReg<f32>>::Reg,1> {
        unsafe {
            Regs::new([_mm256_mul_ps(l, r)])
        }
    }
}
impl SimdMul<f64,f64,f64> for Avx2 where Self: SimdReg<f64> {
    type Output = Regs<<Self as SimdReg<f64>>::Reg,1>;

    #[inline(always)]
    fn mul(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<f64>>::Reg) -> Regs<<Self as SimdReg<f64>>::Reg,1> {
        unsafe {
            Regs::new([_mm256_mul_pd(l, r)])
        }
    }
}
impl SimdSplat<i8> for Avx2 where Self: SimdReg<i8> {

    #[inline(always)]
    fn splat(&self, v: i8) -> Self::Reg {
        unsafe {
            _mm256_set1_epi8(v)
        }
    }
}
impl SimdSplat<i16> for Avx2 where Self: SimdReg<i16> {

    #[inline(always)]
    fn splat(&self, v: i16) -> Self::Reg {
        unsafe {
            _mm256_set1_epi16(v)
        }
    }
}
impl SimdSplat<i32> for Avx2 where Self: SimdReg<i32> {

    #[inline(always)]
    fn splat(&self, v: i32) -> Self::Reg {
        unsafe {
            _mm256_set1_epi32(v)
        }
    }
}
impl SimdSplat<f32> for Avx2 where Self: SimdReg<f32> {

    #[inline(always)]
    fn splat(&self, v: f32) -> Self::Reg {
        unsafe {
            _mm256_set1_ps(v)
        }
    }
}
impl SimdSplat<f64> for Avx2 where Self: SimdReg<f64> {

    #[inline(always)]
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

    #[inline(always)]
    fn reinterpret(&self, reg: <Self as SimdReg<f32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_castps_si256(reg)
        }
    }
}
impl SimdReinterpret<f64,i64> for Avx2 where Self: SimdReg<i64> {

    #[inline(always)]
    fn reinterpret(&self, reg: <Self as SimdReg<f64>>::Reg) -> <Self as SimdReg<i64>>::Reg {
        unsafe {
            _mm256_castpd_si256(reg)
        }
    }
}
impl SimdReinterpret<i32,f32> for Avx2 where Self: SimdReg<f32> {

    #[inline(always)]
    fn reinterpret(&self, reg: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            _mm256_castsi256_ps(reg)
        }
    }
}
impl SimdReinterpret<i64,f64> for Avx2 where Self: SimdReg<f64> {

    #[inline(always)]
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
        let &[lo,hi] = <Self as SimdPromote<i16,i32>>::promotion(self,v).as_ref();

        let regs = Regs::new([
                                 <Self as SimdShl<i32>>::shl(self,lo,w),
                                 <Self as SimdShl<i32>>::shl(self,hi,w)
                             ]);

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
        let &[lo,hi] = <Self as SimdPromote<i16,i32>>::promotion(self,v).as_ref();

        let regs = Regs::new([
            <Self as SimdShr<i32>>::shr(self,lo,w),
            <Self as SimdShr<i32>>::shr(self,hi,w)
        ]);

        <Self as SimdDemote<i32,i16>>::demotion(self,regs)
    }
}
impl SimdShr<i32> for Avx2 where Self: SimdReg<i32> {

    #[inline(always)]
    fn shr(&self, v: <Self as SimdReg<i32>>::Reg, w: <Self as SimdReg<i32>>::ShiftWidth) -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_srav_epi32(v, w)
        }
    }
}
impl SimdShr<i64> for Avx2 where Self: SimdReg<i64> {

    #[inline(always)]
    fn shr(&self, v: <Self as SimdReg<i64>>::Reg, w: <Self as SimdReg<i64>>::ShiftWidth) -> <Self as SimdReg<i64>>::Reg {
        unsafe {
            _mm256_srav_epi64(v, w)
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
    #[inline(always)]
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
    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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

    #[inline(always)]
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
impl SimdMulAdd<i8,i8,i32> for Avx2
    where Self: SimdMul<i8,i8,i32> +
                SimdAdd<i32,i32,i32> {
    type Acc = Regs<<Self as SimdReg<i32>>::Reg,2>;

    #[inline(always)]
    fn zero_acc(&self) -> Self::Acc {
        Regs::new([<Self as SimdZero<i32>>::zero();2])
    }
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<i8>>::Reg, r: <Self as SimdReg<i8>>::Reg,
                acc: Self::Acc) -> Self::Acc {
        let &[acc0,acc1] = acc.as_ref();

        let &[l_lo,l_hi] = <Self as SimdPromote<i8,i16>>::promotion(self,l).as_ref();
        let &[r_lo,r_hi] = <Self as SimdPromote<i8,i16>>::promotion(self,r).as_ref();

        let &[acc0] = <Self as SimdMulAdd<i16,i16,i32>>::mul_add(self,l_lo,r_lo,Regs::new([acc0])).as_ref();
        let &[acc1] = <Self as SimdMulAdd<i16,i16,i32>>::mul_add(self,l_hi,r_hi,Regs::new([acc1])).as_ref();

        Regs::new([
            acc0,acc1
        ])
    }
}
impl SimdMulAdd<i16,i16,i32> for Avx2
    where Self: SimdMul<i16,i16,i32> +
                SimdAdd<i32,i32,i32> {
    type Acc = Regs<<Self as SimdReg<i32>>::Reg,1>;
    #[inline(always)]
    fn zero_acc(&self) -> Self::Acc {
        Regs::new([<Self as SimdZero<i32>>::zero();1])
    }
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<i16>>::Reg, r: <Self as SimdReg<i16>>::Reg,
                acc: Self::Acc) -> Self::Acc {
        unsafe {
            let rs = _mm256_madd_epi16(l,r);

            let &[acc] = acc.as_ref();

            Regs::new([
                <Self as SimdAdd<i32,i32,i32>>::add(self,acc,rs)
            ])
        }
    }
}
impl SimdMulAdd<i32,i32,i32> for Avx2
    where Self: SimdMul<i32,i32,i32> +
                SimdAdd<i32,i32,i32> {
    type Acc = Regs<<Self as SimdReg<i32>>::Reg,1>;
    #[inline(always)]
    fn zero_acc(&self) -> Self::Acc {
        Regs::new([<Self as SimdZero<i32>>::zero()])
    }
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<i32>>::Reg, r: <Self as SimdReg<i32>>::Reg,
               acc: Self::Acc) -> Self::Acc {
        let &[r] = <Self as SimdMul<i32,i32,i32>>::mul(self,l,r).as_ref();

        let &[acc] = acc.as_ref();

        unsafe { Regs::new([<Self as SimdAdd<i32,i32,i32>>::add(self,acc,r)]) }
    }
}
impl SimdMulAdd<f32,f32,f32> for Avx2 {
    type Acc = Regs<<Self as SimdReg<f32>>::Reg,1>;
    #[inline(always)]
    fn zero_acc(&self) -> <Self as SimdMul<f32, f32, f32>>::Output {
        Regs::new([<Self as SimdZero<f32>>::zero()])
    }
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<f32>>::Reg, r: <Self as SimdReg<f32>>::Reg,
               acc:Self::Acc) -> Self::Acc {
        let &[acc] = acc.as_ref();

        unsafe {
            Regs::new([_mm256_fmadd_ps(l,r,acc)])
        }
    }
}
impl SimdMulAdd<f64,f64,f64> for Avx2 {
    type Acc = Regs<<Self as SimdReg<f64>>::Reg,1>;
    #[inline(always)]
    fn zero_acc(&self) -> Self::Acc {
        Regs::new([<Self as SimdZero<f64>>::zero()])
    }
    #[inline(always)]
    fn mul_add(&self, l: <Self as SimdReg<f64>>::Reg, r: <Self as SimdReg<f64>>::Reg,
               acc: Self::Acc) -> Self::Acc {
        unsafe {
            let &[acc] = acc.as_ref();

            Regs::new([_mm256_fmadd_pd(l,r,acc)])
        }
    }
}
impl<SL,SR,SO> SimdPartialDot<SL,SR,SO> for Avx2
    where Self: SimdMul<SL,SR,SO> +
                SimdMulAdd<SL,SR,SO> {
    type Output = <Self as SimdMulAdd<SL,SR,SO>>::Acc;
    #[inline(always)]
    fn zero_acc(&self) -> <Self as SimdPartialDot<SL,SR,SO>>::Output {
        <Self as SimdMulAdd<SL,SR,SO>>::zero_acc(self)
    }
    #[inline(always)]
    fn partial_dot(&self, l: <Self as SimdReg<SL>>::Reg, r: <Self as SimdReg<SR>>::Reg,
                   acc: <Self as SimdPartialDot<SL,SR,SO>>::Output)
        -> <Self as SimdPartialDot<SL,SR,SO>>::Output {
        self.mul_add(l, r, acc)
    }
}
impl SimdZero<i8> for Avx2 where Self: SimdReg<i8> {
    #[inline(always)]
    fn zero() -> <Self as SimdReg<i8>>::Reg {
        unsafe {
            _mm256_setzero_si256()
        }
    }
}
impl SimdZero<i16> for Avx2 where Self: SimdReg<i16> {
    #[inline(always)]
    fn zero() -> <Self as SimdReg<i16>>::Reg {
        unsafe {
            _mm256_setzero_si256()
        }
    }
}
impl SimdZero<i32> for Avx2 where Self: SimdReg<i32> {
    #[inline(always)]
    fn zero() -> <Self as SimdReg<i32>>::Reg {
        unsafe {
            _mm256_setzero_si256()
        }
    }
}
impl SimdZero<f32> for Avx2 where Self: SimdReg<f32> {
    #[inline(always)]
    fn zero() -> <Self as SimdReg<f32>>::Reg {
        unsafe {
            _mm256_setzero_ps()
        }
    }
}
impl SimdZero<f64> for Avx2 where Self: SimdReg<f64> {
    #[inline(always)]
    fn zero() -> <Self as SimdReg<f64>>::Reg {
        unsafe {
            _mm256_setzero_pd()
        }
    }
}
impl<T> SimdAddVector<T,T,T> for Avx2
    where Self: SimdReg<T> +
                SimdLoad<T> +
                SimdStore<T> +
                SimdLanes<T> +
                SimdAdd<T,T,T,Lhs=<Self as SimdReg<T>>::Reg,Rhs=<Self as SimdReg<T>>::Reg,Output=<Self as SimdReg<T>>::Reg>,
          T: Add<T,Output=T> + Default + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn add_vector<'a,'b,const N: usize>(&self, l: VectorView<'a,T,N>, r: VectorView<'b,T,N>)
        -> OwnedVector<T,N> {
        unsafe { <Self as SimdQueryVector<T,SimdAddOperator<T,T,T,Avx2>>>::query_vector(self,l,r,&SimdAddOperator::new(self)) }
    }
}
impl<SL,SR,C> SimdLinearCombination<SL,SR,C> for Avx2 where Self: SimdLinearCombinationKernel<SL,SR,C> {
    #[target_feature(enable = "avx2")]
    unsafe fn linear_combination<'a, 'b, const N: usize>(&self, lines: &'a [(VectorView<'a, SL, N>, C)], acc: VectorMutView<'b,SR,N>) {
        unsafe { <Self as SimdLinearCombinationKernel<SL,SR,C>>::linear_combination(self,lines,acc) }
    }
}
impl SimdDot<i8,i8,i32> for Avx2
    where Self: SimdDotKernel<i8,i8,i32> {
    #[target_feature(enable = "avx2")]
    unsafe fn dot<'a, 'b, const N: usize>(&self, l: &VectorView<'a, i8, N>, r: VectorView<'b, i8, N>) -> i32 {
        unsafe { <Self as SimdDotKernel<i8,i8,i32>>::dot::<N,{ <Self as SimdCols<i8>>::COLS * 2 }>(self, l, r) }
    }
}
impl SimdDot<i16,i16,i32> for Avx2
    where Self: SimdDotKernel<i16,i16,i32> {
    #[target_feature(enable = "avx2")]
    unsafe fn dot<'a, 'b, const N: usize>(&self, l: &VectorView<'a, i16, N>, r: VectorView<'b, i16, N>) -> i32 {
        unsafe { <Self as SimdDotKernel<i16,i16,i32>>::dot::<N,{ <Self as SimdCols<i16>>::COLS * 2 }>(self, l, r) }
    }
}
impl SimdDot<i32,i32,i32> for Avx2
    where Self: SimdDotKernel<i32,i32,i32> {
    #[target_feature(enable = "avx2")]
    unsafe fn dot<'a, 'b, const N: usize>(&self, l: &VectorView<'a, i32, N>, r: VectorView<'b, i32, N>) -> i32 {
        unsafe { <Self as SimdDotKernel<i32,i32,i32>>::dot::<N,{ <Self as SimdCols<i32>>::COLS * 2 }>(self, l, r) }
    }
}
impl SimdDot<f32,f32,f32> for Avx2
    where Self: SimdDotKernel<f32,f32,f32> {
    #[target_feature(enable = "avx2")]
    unsafe fn dot<'a, 'b, const N: usize>(&self, l: &VectorView<'a, f32, N>, r: VectorView<'b, f32, N>) -> f32 {
        unsafe { <Self as SimdDotKernel<f32,f32,f32>>::dot::<N,{ <Self as SimdCols<f32>>::COLS * 2 }>(self, l, r) }
    }
}
impl SimdDot<f64,f64,f64> for Avx2
    where Self: SimdDotKernel<f64,f64,f64> {
    #[target_feature(enable = "avx2")]
    unsafe fn dot<'a, 'b, const N: usize>(&self, l: &VectorView<'a, f64, N>, r: VectorView<'b, f64, N>) -> f64 {
        unsafe { <Self as SimdDotKernel<f64,f64,f64>>::dot::<N,{ <Self as SimdCols<f64>>::COLS * 2 }>(self, l, r) }
    }
}
impl<SL,SR,SO> SimdMatVec<SL,SR,SO> for Avx2
    where Self: SimdZero<SO> +
                SimdPartialDot<SL,SR,SO> +
                SimdAdd<SO,SO,SO> +
                SimdHSum<SO> +
                SimdLanes<SL> +
                SimdLoad<SL> +
                SimdLoad<SR> +
                SimdPartialDot<SL,SR,SO>,
                SL: Add<SR,Output = SO> + Clone + Copy,
                SR: Clone + Copy,
                SO: From<SL> + From<SR> + Mul<SO,Output=SO> + AddAssign,
                <Self as SimdReg<SO>>::Reg: Copy {

    #[target_feature(enable = "avx2")]
    unsafe fn matvec<'a, const N: usize, const K: usize>(&self, l: MatrixView<'a, SL, N, K>, r: VectorView<'a, SR, K>, o: &mut OwnedVector<SO, N>) {
        unsafe {
            for i in 0..N {
                let mut acc = <Self as SimdPartialDot<SL,SR,SO>>::zero_acc(self);

                for k in (0..(K - K % <Self as SimdLanes<SL>>::LANES)).step_by(<Self as SimdLanes<SL>>::LANES) {
                    let lr = self.load(l.row(i).as_ref().as_ptr().add(k));
                    let rr = self.load(r.as_ref().as_ptr().add(k));

                    acc = self.partial_dot(lr,rr,acc);
                }

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
impl<SL,SR,SO> SimdVMat<SL,SR,SO> for Avx2
    where Self: SimdDot<SL,SR,SO> {
    #[target_feature(enable = "avx2")]
    unsafe fn vmat<'a,'b,const M: usize, const K: usize>(&self, l: VectorView<'a, SL, K>, r: ColumnMajorMatrix<'b, SR, K, M>, o: &mut OwnedVector<SO, M>) {
        for (o,c) in o.as_mut().iter_mut().zip(r.as_ref().chunks_exact(K)) {
            let v = VectorView::from(<& [SR;K]>::try_from(c).unwrap());

            let s = unsafe { <Self as SimdDot<SL,SR,SO>>::dot(self,&l,v) };

            *o = s;
        }
    }
}
impl<SL,SR,SO> SimdMapVMat<SL,SR,SO> for Avx2
    where Self: SimdDot<SL,SR,SO> {
    #[target_feature(enable = "avx2")]
    unsafe fn map_vmat<'a,'b,MP,MO,const M: usize, const K: usize>(&self, l: VectorView<'a, SL, K>,
                                                                   r: ColumnMajorMatrix<'b, SR, K, M>,
                                                                   mapper:MP,o: &mut OwnedVector<MO, M>) where MP: Mapper<Extent<M>,SO,MO> {
        let mut mapper = mapper;

        for (o,c) in o.as_mut().iter_mut().zip(r.as_ref().chunks_exact(K)) {
            let v = VectorView::from(<& [SR;K]>::try_from(c).unwrap());

            let s = unsafe { <Self as SimdDot<SL,SR,SO>>::dot(self,&l,v) };

            *o = mapper.map(s);
        }
    }
}
impl SimdMatMul<i8,i8,i32> for Avx2
    where Self: SimdMatMulKernel<i8,i8,i32> +
                SimdRows<i8> +
                SimdCols<i8> {
    #[target_feature(enable = "avx2")]
    unsafe fn matmul<'a, const N: usize, const M: usize, const K: usize>(&self, l: MatrixView<'a, i8, N, K>, r: ColumnMajorMatrix<'a, i8, K, M>, o: MatrixMutView<'a, i32, N, M>) {
        unsafe {
            <Self as SimdMatMulKernel<i8,i8,i32>>::matmul::<N,M,K,{ <Self as SimdRows<i8>>::ROWS },{ <Self as SimdCols<i8>>::COLS * 2}>(
                self,l,r,o
            );
        }
    }
}
impl SimdMatMul<i16,i16,i32> for Avx2
    where Self: SimdMatMulKernel<i16,i16,i32> +
                SimdRows<i16> +
                SimdCols<i16> {
    #[target_feature(enable = "avx2")]
    unsafe fn matmul<'a, const N: usize, const M: usize, const K: usize>(&self, l: MatrixView<'a, i16, N, K>, r: ColumnMajorMatrix<'a, i16, K, M>, o: MatrixMutView<'a, i32, N, M>) {
        unsafe {
            <Self as SimdMatMulKernel<i16,i16,i32>>::matmul::<N,M,K,{ <Self as SimdRows<i16>>::ROWS },{ <Self as SimdCols<i16>>::COLS * 2}>(
                self,l,r,o
            );
        }
    }
}
impl SimdMatMul<i32,i32,i32> for Avx2
    where Self: SimdMatMulKernel<i32,i32,i32> +
                SimdRows<i32> +
                SimdCols<i32> {
    #[target_feature(enable = "avx2")]
    unsafe fn matmul<'a, const N: usize, const M: usize, const K: usize>(&self, l: MatrixView<'a, i32, N, K>, r: ColumnMajorMatrix<'a, i32, K, M>, o: MatrixMutView<'a, i32, N, M>) {
        unsafe {
            <Self as SimdMatMulKernel<i32,i32,i32>>::matmul::<N,M,K,{ <Self as SimdRows<i32>>::ROWS },{ <Self as SimdCols<i32>>::COLS * 2}>(
                self,l,r,o
            );
        }
    }
}
impl<OP,SL,SR,SO> SimdCombinationQueryVector<OP,SL,SR,SO> for Avx2
    where Self: Backend + NativeBackend +
                SimdReg<SL> +
                SimdReg<SR> +
                SimdReg<SO> +
                SimdLoadSeq<SL,<OP as SimdBinaryOperator<Self,SL,SR>>::Lhs> +
                SimdLoadSeq<SR,<OP as SimdBinaryOperator<Self,SL,SR>>::Rhs> +
                SimdStoreSeq<SO,<OP as SimdBinaryOperator<Self,SL,SR>>::Output> +
                SimdCombinationQueryVectorKernel<OP,SL,SR,SO>,
          OP: SimdBinaryOperator<Self,SL,SR, ScalarOutput = SO>,
          SL: Copy,
          SR: Copy,
          SO: Default + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn combination_query_vector<'a, 'b, const N: usize>(
        &self, l: VectorView<'a, SL, N>, r: VectorView<'b, SR, N>, operator: OP
    ) -> OwnedVector<SO,N> {
        <Self as SimdCombinationQueryVectorKernel<OP,SL,SR,SO>>::combination_query_vector::<N,{ <Self as SimdLanes<i8>>::LANES }>(self,l,r,operator)
    }
}
impl SimdMatMul<f32,f32,f32> for Avx2
    where Self: SimdMatMulKernel<f32,f32,f32> +
                SimdRows<f32> +
                SimdCols<f32> {
    #[target_feature(enable = "avx2")]
    unsafe fn matmul<'a, const N: usize, const M: usize, const K: usize>(&self, l: MatrixView<'a, f32, N, K>, r: ColumnMajorMatrix<'a, f32, K, M>, o: MatrixMutView<'a, f32, N, M>) {
        unsafe {
            <Self as SimdMatMulKernel<f32,f32,f32>>::matmul::<N,M,K,{ <Self as SimdRows<f32>>::ROWS },{ <Self as SimdCols<f32>>::COLS * 2}>(
                self,l,r,o
            );
        }
    }
}
impl SimdMatMul<f64,f64,f64> for Avx2
    where Self: SimdMatMulKernel<f64,f64,f64> +
                SimdRows<f64> +
                SimdCols<f64> {
    #[target_feature(enable = "avx2")]
    unsafe fn matmul<'a, const N: usize, const M: usize, const K: usize>(&self, l: MatrixView<'a, f64, N, K>, r: ColumnMajorMatrix<'a, f64, K, M>, o: MatrixMutView<'a, f64, N, M>) {
        unsafe {
            <Self as SimdMatMulKernel<f64,f64,f64>>::matmul::<N,M,K,{ <Self as SimdRows<f64>>::ROWS },{ <Self as SimdCols<f64>>::COLS * 2}>(
                self,l,r,o
            );
        }
    }
}
impl<SL,SR,SO> SimdResolveBinaryOperator<AddOperator<SL,SR,SO>,SL,SR,SO> for Avx2
    where Self: Backend + NativeBackend + Clone,
          SO: From<SL> + From<SR> + Add<SO,Output=SO> + Copy,
          SimdQueryAddOperator<SL,SR,SO,Self>: SimdBinaryOperator<Self,SL,SR,ScalarOutput=SO> {
    type NativeOperator = SimdQueryAddOperator<SL,SR,SO,Self>;
    #[inline(always)]
    fn resolve_operator(&self,_: &AddOperator<SL,SR,SO>) -> Self::NativeOperator {
        SimdQueryAddOperator::new(self)
    }
}
impl<P,SL,SR> SimdResolvePatrialBinaryOperator<P,ShrOperator<P,i32>,SL,SR,i32> for Avx2
    where Self: Backend + NativeBackend +
                SimdReg<i32> +
                SimdShiftWidth<i32> +
                SimdShr<i32> +
                Clone,
          P: ToSimdBinaryOperator<Self>,
          <P as ToSimdBinaryOperator<Self>>::NativeOperator: SimdBinaryOperator<Self,SL,SR,Output=Regs<<Self as SimdReg<i32>>::Reg,4>,ScalarOutput=i32> {
    type NativeOperator = SimdQueryShrOperator<Self,<P as ToSimdBinaryOperator<Self>>::NativeOperator,Regs<<Self as SimdReg<i32>>::Reg,4>,i32>;
    #[inline(always)]
    fn resolve_operator(&self,parent_operator:<P as ToSimdBinaryOperator<Self>>::NativeOperator, operator: &ShrOperator<P,i32>) -> Self::NativeOperator {
        parent_operator.compose(|p| SimdQueryShrOperator::new(self, p, operator.width))
    }
}
impl<P,SL,SR> SimdResolvePatrialBinaryOperator<P,DemoteOperator<P,i8>,SL,SR,i8> for Avx2
    where Self: Backend + NativeBackend +
                SimdReg<i32> +
                Clone,
          SL: Copy,
          SR: Copy,
          P: ToSimdBinaryOperator<Self>,
          <P as ToSimdBinaryOperator<Self>>::NativeOperator: SimdBinaryOperator<Self,SL,SR,Output=Regs<<Self as SimdReg<i32>>::Reg,4>,ScalarOutput=i32> {
    type NativeOperator = SimdQueryDemoteOperator<Self,<P as ToSimdBinaryOperator<Self>>::NativeOperator,Regs<<Self as SimdReg<i32>>::Reg,4>,Regs<<Self as SimdReg<i8>>::Reg,1>,i32,i8>;
    #[inline(always)]
    fn resolve_operator(&self,parent_operator:<P as ToSimdBinaryOperator<Self>>::NativeOperator, _: &DemoteOperator<P,i8>) -> Self::NativeOperator {
        parent_operator.compose(|p| SimdQueryDemoteOperator::new(self, p))
    }
}