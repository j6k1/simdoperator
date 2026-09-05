//! Implementation of SIMD Operations Using AVX2

use std::arch::x86_64::{__m256, __m256d, __m256i, _mm256_add_epi8, _mm256_and_pd, _mm256_and_ps, _mm256_and_si256, _mm256_blendv_epi8, _mm256_blendv_pd, _mm256_blendv_ps, _mm256_castpd_si256, _mm256_castps_si256, _mm256_castsi256_pd, _mm256_castsi256_ps, _mm256_cmp_pd, _mm256_cmp_ps, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpeq_epi8, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cmpgt_epi8, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _CMP_EQ_OQ, _CMP_GT_OQ};
use crate::backend::common::Backend;
use crate::traits::{SimdAdd, SimdLanes, SimdLoad, SimdMask, SimdReg, SimdRows, SimdStore};
use crate::{OwnedVector, Vector};

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
impl SimdRows<i8> for Avx2 {
    const ROWS: usize = 1;
}
impl SimdRows<i16> for Avx2 {
    const ROWS: usize = 1;
}
impl SimdRows<i32> for Avx2 {
    const ROWS: usize = 2;
}
impl SimdRows<f32> for Avx2 {
    const ROWS: usize = 2;
}
impl SimdRows<f64> for Avx2 {
    const ROWS: usize = 1;
}
impl SimdReg<i8> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
}
impl SimdReg<i16> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
}
impl SimdReg<i32> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
}
impl SimdReg<f32> for Avx2 {
    type Reg = __m256;
    type Mask = __m256i;
}
impl SimdReg<f64> for Avx2 {
    type Reg = __m256d;
    type Mask = __m256i;
}
impl SimdLoad<i8> for Avx2 {
    unsafe fn load(&self, ptr: *const i8) -> Self::Reg {
        _mm256_loadu_si256(ptr as *const __m256i)
    }
}
impl SimdLoad<i16> for Avx2 {
    unsafe fn load(&self, ptr: *const i16) -> Self::Reg {
        _mm256_loadu_si256(ptr as *const __m256i)
    }
}
impl SimdLoad<i32> for Avx2 {
    unsafe fn load(&self, ptr: *const i32) -> Self::Reg {
        _mm256_loadu_si256(ptr as *const __m256i)
    }
}
impl SimdLoad<f32> for Avx2 {
    unsafe fn load(&self, ptr: *const f32) -> Self::Reg {
        _mm256_loadu_ps(ptr)
    }
}
impl SimdLoad<f64> for Avx2 {
    unsafe fn load(&self, ptr: *const f64) -> Self::Reg {
        _mm256_loadu_pd(ptr)
    }
}
impl SimdStore<i8> for Avx2 {
    unsafe fn store(&self, ptr: *mut i8, a: Self::Reg) {
        _mm256_storeu_si256(ptr as *mut __m256i, a);
    }
}
impl SimdStore<i16> for Avx2 {
    unsafe fn store(&self, ptr: *mut i16, a: Self::Reg) {
        _mm256_storeu_si256(ptr as *mut __m256i, a);
    }
}
impl SimdStore<i32> for Avx2 {
    unsafe fn store(&self, ptr: *mut i32, a: Self::Reg) {
        _mm256_storeu_si256(ptr as *mut __m256i, a);
    }
}
impl SimdStore<f32> for Avx2 {
    unsafe fn store(&self, ptr: *mut f32, a: Self::Reg) {
        _mm256_storeu_ps(ptr, a);
    }
}
impl SimdStore<f64> for Avx2 {
    unsafe fn store(&self, ptr: *mut f64, a: Self::Reg) {
        _mm256_storeu_pd(ptr, a);
    }
}
impl SimdMask<i8> for Avx2 where Self: SimdReg<i8> {
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpgt_epi8(a,b) }
    }

    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi8(a,b) }
    }

    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
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
impl SimdMask<i16> for Avx2 where Self: SimdReg<i16> {
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpgt_epi16(a,b) }
    }

    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi16(a,b) }
    }

    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
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
impl SimdMask<i32> for Avx2 where Self: SimdReg<i32> {
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpgt_epi32(a,b) }
    }

    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_cmpeq_epi32(a,b) }
    }

    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
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
impl SimdMask<f32> for Avx2 where Self: SimdReg<f32> {
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castps_si256(_mm256_cmp_ps(a,b, _CMP_GT_OQ)) }
    }

    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castps_si256(_mm256_cmp_ps(a,b, _CMP_EQ_OQ)) }
    }

    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_ps(a,_mm256_castsi256_ps(mask)) }
    }

    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_ps(b, a, _mm256_castsi256_ps(mask)) }
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
impl SimdMask<f64> for Avx2 where Self: SimdReg<f64> {
    fn cmp_gt(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castpd_si256(_mm256_cmp_pd(a,b, _CMP_GT_OQ)) }
    }

    fn cmp_eq(&self, a:Self::Reg,b:Self::Reg) -> Self::Mask {
        unsafe { _mm256_castpd_si256(_mm256_cmp_pd(a,b, _CMP_EQ_OQ)) }
    }

    fn mask_zero(&self, mask: Self::Mask, a: Self::Reg) -> Self::Reg {
        unsafe { _mm256_and_pd(a,_mm256_castsi256_pd(mask)) }
    }

    fn select(&self, mask: Self::Mask, a: Self::Reg, b: Self::Reg) -> Self::Reg {
        unsafe { _mm256_blendv_pd(b, a, _mm256_castsi256_pd(mask)) }
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
impl SimdAdd<i8,i8,i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn add<'a,const N: usize>(&self,l: &Vector<'a,i8,N,Self::Backend>,r: &Vector<'a,i8,N,Self::Backend>)
        -> OwnedVector<i8,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i8; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i8>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i8>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i8>>::LANES));

                    let rr = _mm256_add_epi8(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i8>>::LANES),rr);

                    i += <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i8>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_add_epi8(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i8>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i8>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] + r[j];
                }
            }
        }

        rs
    }
}