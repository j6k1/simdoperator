//! Implementation of SIMD Operations Using AVX2

use std::arch::x86_64::{__m128i, __m256, __m256d, __m256i, _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi8, _mm256_add_pd, _mm256_add_ps, _mm256_and_pd, _mm256_and_ps, _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_blendv_pd, _mm256_blendv_ps, _mm256_castpd_si256, _mm256_castps_si256, _mm256_castsi256_pd, _mm256_castsi256_ps, _mm256_cmp_pd, _mm256_cmp_ps, _mm256_cmpeq_epi16, _mm256_cmpeq_epi32, _mm256_cmpeq_epi8, _mm256_cmpgt_epi16, _mm256_cmpgt_epi32, _mm256_cmpgt_epi8, _mm256_cvtepi16_epi32, _mm256_cvtepi16_epi8, _mm256_cvtepi32_epi16, _mm256_cvtepi8_epi32, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_mul_pd, _mm256_mul_ps, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi32, _mm256_set1_pd, _mm256_set1_ps, _mm256_sll_epi32, _mm256_sll_epi64, _mm256_slli_epi32, _mm256_srl_epi32, _mm256_srl_epi64, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32, _mm256_sub_epi8, _mm256_sub_pd, _mm256_sub_ps, _mm256_xor_si256, _mm_cvtepi16_epi32, _mm_cvtepi8_epi16, _mm_loadl_epi64, _mm_loadu_si128, _mm_mullo_epi16, _mm_packus_epi16, _mm_set1_epi32, _mm_storel_epi64, _CMP_EQ_OQ, _CMP_GT_OQ};
use std::mem::transmute;
use crate::backend::common::Backend;
use crate::traits::{SimdAdd, SimdBitAnd, SimdBitNot, SimdBitOr, SimdBitXor, SimdLanes, SimdLoad, SimdMask, SimdMul, SimdReg, SimdRows, SimdScalarMul, SimdShiftLeft, SimdShiftRight, SimdStore, SimdSub};
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
impl SimdRows<i64> for Avx2 {
    const ROWS: usize = 1;
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
    type Bits = i8;
}
impl SimdReg<i16> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = i16;
}
impl SimdReg<i32> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = i32;
}
impl SimdReg<i64> for Avx2 {
    type Reg = __m256i;
    type Mask = __m256i;
    type Bits = i64;
}
impl SimdReg<f32> for Avx2 {
    type Reg = __m256;
    type Mask = __m256i;
    type Bits = i32;
}
impl SimdReg<f64> for Avx2 {
    type Reg = __m256d;
    type Mask = __m256i;
    type Bits = i64;
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
impl SimdLoad<i64> for Avx2 {
    unsafe fn load(&self, ptr: *const i64) -> Self::Reg {
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
impl SimdAdd<i16,i16,i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn add<'a,const N: usize>(&self,l: &Vector<'a,i16,N,Self::Backend>,r: &Vector<'a,i16,N,Self::Backend>)
                              -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i16>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i16>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i16>>::LANES));

                    let rr = _mm256_add_epi16(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i16>>::LANES),rr);

                    i += <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i16>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_add_epi16(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i16>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i16>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] + r[j];
                }
            }
        }

        rs
    }
}
impl SimdAdd<i32,i32,i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn add<'a,const N: usize>(&self,l: &Vector<'a,i32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let rr = _mm256_add_epi32(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),rr);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_add_epi32(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] + r[j];
                }
            }
        }

        rs
    }
}
impl SimdAdd<f32,f32,f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;
    fn add<'a,const N: usize>(&self,l: &Vector<'a,f32,N,Self::Backend>,r: &Vector<'a,f32,N,Self::Backend>)
                              -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let rr = _mm256_add_ps(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),rr);

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_add_ps(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] + r[j];
                }
            }
        }

        rs
    }
}
impl SimdAdd<f64,f64,f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn add<'a,const N: usize>(&self,l: &Vector<'a,f64,N,Self::Backend>,r: &Vector<'a,f64,N,Self::Backend>)
                              -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f64>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let rr = _mm256_add_pd(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),rr);

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_add_pd(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] + r[j];
                }
            }
        }

        rs
    }
}
impl SimdSub<i8,i8,i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn sub<'a,const N: usize>(&self,l: &Vector<'a,i8,N,Self::Backend>,r: &Vector<'a,i8,N,Self::Backend>)
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

                    let rr = _mm256_sub_epi8(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i8>>::LANES),rr);

                    i += <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i8>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_sub_epi8(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i8>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i8>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] - r[j];
                }
            }
        }

        rs
    }
}
impl SimdSub<i16,i16,i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn sub<'a,const N: usize>(&self,l: &Vector<'a,i16,N,Self::Backend>,r: &Vector<'a,i16,N,Self::Backend>)
                              -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i16>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i16>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i16>>::LANES));

                    let rr = _mm256_sub_epi16(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i16>>::LANES),rr);

                    i += <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i16>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_sub_epi16(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i16>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i16>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] - r[j];
                }
            }
        }

        rs
    }
}
impl SimdSub<i32,i32,i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn sub<'a,const N: usize>(&self,l: &Vector<'a,i32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let rr = _mm256_sub_epi32(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),rr);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_sub_epi32(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] - r[j];
                }
            }
        }

        rs
    }
}
impl SimdSub<f32,f32,f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;
    fn sub<'a,const N: usize>(&self,l: &Vector<'a,f32,N,Self::Backend>,r: &Vector<'a,f32,N,Self::Backend>)
                              -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let rr = _mm256_sub_ps(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),rr);

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_sub_ps(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] - r[j];
                }
            }
        }

        rs
    }
}
impl SimdSub<f64,f64,f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn sub<'a,const N: usize>(&self,l: &Vector<'a,f64,N,Self::Backend>,r: &Vector<'a,f64,N,Self::Backend>)
                              -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f64>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let rr = _mm256_sub_pd(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),rr);

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_sub_pd(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] - r[j];
                }
            }
        }

        rs
    }
}
impl SimdMul<i8,i8,i32> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i32> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn mul<'a,const N: usize>(&self,l: &Vector<'a,i8,N,Self::Backend>,r: &Vector<'a,i8,N,Self::Backend>)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES <= N {
                let a8 = _mm_loadl_epi64(pa.add(i) as *const __m128i);
                let b8 = _mm_loadl_epi64(pb.add(i) as *const __m128i);

                let a16 = _mm_cvtepi8_epi16(a8);
                let b16 = _mm_cvtepi8_epi16(b8);
                let prod16 = _mm_mullo_epi16(a16, b16);
                let prod32 = _mm256_cvtepi16_epi32(prod16);

                self.store(po.add(i),prod32);

                i += <Self as SimdLanes::<i8>>::LANES;
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] as i32 * r[j] as i32;
                }
            }
        }

        rs
    }
}
impl SimdMul<i8,i16,i32> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i32> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn mul<'a,const N: usize>(&self,l: &Vector<'a,i8,N,Self::Backend>,r: &Vector<'a,i16,N,Self::Backend>)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES <= N {
                let a8 = _mm_loadl_epi64(pa.add(i) as *const __m128i);

                let a16 = _mm_cvtepi8_epi16(a8);
                let b16 = _mm_loadu_si128(pb.add(i) as *const __m128i);

                let prod16 = _mm_mullo_epi16(a16, b16);
                let prod32 = _mm256_cvtepi16_epi32(prod16);

                self.store(po.add(i),prod32);

                i += <Self as SimdLanes::<i8>>::LANES;
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] as i32 * r[j] as i32;
                }
            }
        }

        rs
    }
}
impl SimdMul<i16,i16,i32> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i32> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn mul<'a,const N: usize>(&self,l: &Vector<'a,i16,N,Self::Backend>,r: &Vector<'a,i16,N,Self::Backend>)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES <= N {
                let a16 = _mm_loadu_si128(pa.add(i) as *const __m128i);
                let b16 = _mm_loadu_si128(pb.add(i) as *const __m128i);

                let prod16 = _mm_mullo_epi16(a16, b16);
                let prod32 = _mm256_cvtepi16_epi32(prod16);

                self.store(po.add(i),prod32);

                i += <Self as SimdLanes::<i8>>::LANES;
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] as i32 * r[j] as i32;
                }
            }
        }

        rs
    }
}
impl SimdMul<i32,i32,i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn mul<'a,const N: usize>(&self,l: &Vector<'a,i32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let rr = _mm256_mul_epi32(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),rr);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_mul_epi32(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] * r[j];
                }
            }
        }

        rs
    }
}
impl SimdMul<f32,f32,f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;
    fn mul<'a,const N: usize>(&self,l: &Vector<'a,f32,N,Self::Backend>,r: &Vector<'a,f32,N,Self::Backend>)
                              -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let rr = _mm256_mul_ps(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),rr);

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_mul_ps(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] * r[j];
                }
            }
        }

        rs
    }
}
impl SimdMul<f64,f64,f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn mul<'a,const N: usize>(&self,l: &Vector<'a,f64,N,Self::Backend>,r: &Vector<'a,f64,N,Self::Backend>)
                              -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f64>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let rr = _mm256_mul_pd(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),rr);

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_mul_pd(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] * r[j];
                }
            }
        }

        rs
    }
}
impl SimdScalarMul<i32,i32,i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn scalarmul<'a, const N: usize>(&self, l:i32,r: &Vector<'a, i32, N, Self::Backend>) -> OwnedVector<i32, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let s32 = _mm256_set1_epi32(l);
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v32 = self.load(pb.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let o = _mm256_mullo_epi32(s32,v32);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),o);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v32 = self.load(pb.add(i));

                    let o = _mm256_mullo_epi32(s32,v32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l as i32 * r[j] as i32;
                }
            }
        }

        rs
    }
}
impl SimdScalarMul<i8,i8,i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn scalarmul<'a, const N: usize>(&self, l:i8,r: &Vector<'a, i8, N, Self::Backend>) -> OwnedVector<i32, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let s32 = _mm256_set1_epi32(l as i32);
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v8 = _mm_loadl_epi64(
                        pb.add(i + <Self as SimdLanes::<i32>>::LANES * j) as *const __m128i
                    );

                    let v32 = _mm256_cvtepi8_epi32(v8);

                    let o = _mm256_mullo_epi32(v32, s32);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),o);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v8 = _mm_loadl_epi64(
                        pb.add(i) as *const __m128i
                    );

                    let v32 = _mm256_cvtepi8_epi32(v8);

                    let o = _mm256_mullo_epi32(v32, s32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l as i32 * r[j] as i32;
                }
            }
        }

        rs
    }
}
impl SimdScalarMul<i8,i16,i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn scalarmul<'a, const N: usize>(&self, l:i8,r: &Vector<'a, i16, N, Self::Backend>) -> OwnedVector<i32, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let s32 = _mm256_set1_epi32(l as i32);
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v16 = _mm_loadu_si128(
                        pb.add(i + <Self as SimdLanes::<i32>>::LANES * j) as *const __m128i
                    );

                    let v32 = _mm256_cvtepi16_epi32(v16);

                    let o = _mm256_mullo_epi32(v32, s32);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),o);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v16 = _mm_loadu_si128(
                        pb.add(i) as *const __m128i
                    );

                    let v32 = _mm256_cvtepi16_epi32(v16);

                    let o = _mm256_mullo_epi32(v32, s32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l as i32 * r[j] as i32;
                }
            }
        }

        rs
    }
}
impl SimdScalarMul<i16,i16,i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn scalarmul<'a, const N: usize>(&self, l:i16,r: &Vector<'a, i16, N, Self::Backend>) -> OwnedVector<i32, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let s32 = _mm256_set1_epi32(l as i32);
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v16 = _mm_loadu_si128(
                        pb.add(i + <Self as SimdLanes::<i32>>::LANES * j) as *const __m128i
                    );

                    let v32 = _mm256_cvtepi16_epi32(v16);

                    let o = _mm256_mullo_epi32(v32, s32);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),o);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v16 = _mm_loadu_si128(
                        pb.add(i) as *const __m128i
                    );

                    let v32 = _mm256_cvtepi16_epi32(v16);

                    let o = _mm256_mullo_epi32(v32, s32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l as i32 * r[j] as i32;
                }
            }
        }

        rs
    }
}
impl SimdScalarMul<f32,f32,f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;
    fn scalarmul<'a, const N: usize>(&self, l:f32,r: &Vector<'a, f32, N, Self::Backend>) -> OwnedVector<f32, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let s32 = _mm256_set1_ps(l);
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let v32 = self.load(pb.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let o = _mm256_mul_ps(s32,v32);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),o);

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let v32 = self.load(pb.add(i));

                    let o = _mm256_mul_ps(s32,v32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l as f32 * r[j] as f32;
                }
            }
        }

        rs
    }
}
impl SimdScalarMul<f64,f64,f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn scalarmul<'a, const N: usize>(&self, l:f64,r: &Vector<'a, f64, N, Self::Backend>) -> OwnedVector<f64, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let s32 = _mm256_set1_pd(l);
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let v32 = self.load(pb.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let o = _mm256_mul_pd(s32,v32);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),o);

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let v32 = self.load(pb.add(i));

                    let o = _mm256_mul_pd(s32,v32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l as f64 * r[j] as f64;
                }
            }
        }

        rs
    }
}
impl SimdBitAnd<i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn bitand<'a,const N: usize>(&self,l: &Vector<'a,i8,N,Self::Backend>,r: &Vector<'a,i8,N,Self::Backend>)
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

                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i8>>::LANES),rr);

                    i += <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i8>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i8>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i8>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] & r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitAnd<i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn bitand<'a,const N: usize>(&self,l: &Vector<'a,i16,N,Self::Backend>,r: &Vector<'a,i16,N,Self::Backend>)
                                 -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i16>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i16>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i16>>::LANES));

                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i16>>::LANES),rr);

                    i += <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i16>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i16>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i16>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] & r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitAnd<i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn bitand<'a,const N: usize>(&self,l: &Vector<'a,i32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                                 -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),rr);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] & r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitAnd<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;
    fn bitand<'a,const N: usize>(&self,l: &Vector<'a,f32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                                 -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f32::from_bits(l[j].to_bits() & r[j] as u32);
                }
            }
        }

        rs
    }
}
impl SimdBitAnd<f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn bitand<'a,const N: usize>(&self,l: &Vector<'a,f64,N,Self::Backend>,r: &Vector<'a,i64,N,Self::Backend>)
        -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f64>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_and_si256(ra,rb);

                    self.store(po.add(i),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f64::from_bits(l[j].to_bits() & r[j] as u64);
                }
            }
        }

        rs
    }
}
impl SimdBitOr<i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn bitor<'a,const N: usize>(&self,l: &Vector<'a,i8,N,Self::Backend>,r: &Vector<'a,i8,N,Self::Backend>)
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

                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i8>>::LANES),rr);

                    i += <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i8>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i8>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i8>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] | r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitOr<i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn bitor<'a,const N: usize>(&self,l: &Vector<'a,i16,N,Self::Backend>,r: &Vector<'a,i16,N,Self::Backend>)
                                 -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i16>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i16>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i16>>::LANES));

                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i16>>::LANES),rr);

                    i += <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i16>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i16>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i16>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] | r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitOr<i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn bitor<'a,const N: usize>(&self,l: &Vector<'a,i32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                                 -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),rr);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] | r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitOr<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;

    fn bitor<'a,const N: usize>(&self,l: &Vector<'a,f32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                                 -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f32::from_bits(l[j].to_bits() | r[j] as u32);
                }
            }
        }

        rs
    }
}
impl SimdBitOr<f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn bitor<'a,const N: usize>(&self,l: &Vector<'a,f64,N,Self::Backend>,r: &Vector<'a,i64,N,Self::Backend>)
                                 -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f64>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_or_si256(ra,rb);

                    self.store(po.add(i),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f64::from_bits(l[j].to_bits() | r[j] as u64);
                }
            }
        }

        rs
    }
}
impl SimdBitXor<i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn bitxor<'a,const N: usize>(&self,l: &Vector<'a,i8,N,Self::Backend>,r: &Vector<'a,i8,N,Self::Backend>)
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

                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i8>>::LANES),rr);

                    i += <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i8>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i8>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i8>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] ^ r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitXor<i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn bitxor<'a,const N: usize>(&self,l: &Vector<'a,i16,N,Self::Backend>,r: &Vector<'a,i16,N,Self::Backend>)
                                 -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i16>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i16>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i16>>::LANES));

                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i16>>::LANES),rr);

                    i += <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i16>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i16>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i16>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] ^ r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitXor<i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn bitxor<'a,const N: usize>(&self,l: &Vector<'a,i32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                                 -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),rr);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] ^ r[j];
                }
            }
        }

        rs
    }
}
impl SimdBitXor<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;

    fn bitxor<'a,const N: usize>(&self,l: &Vector<'a,f32,N,Self::Backend>,r: &Vector<'a,i32,N,Self::Backend>)
                                 -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f32>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f32::from_bits(l[j].to_bits() ^ r[j] as u32);
                }
            }
        }

        rs
    }
}
impl SimdBitXor<f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn bitxor<'a,const N: usize>(&self,l: &Vector<'a,f64,N,Self::Backend>,r: &Vector<'a,i64,N,Self::Backend>)
                                 -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f64>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_xor_si256(ra,rb);

                    self.store(po.add(i),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f64::from_bits(l[j].to_bits() ^ r[j] as u64);
                }
            }
        }

        rs
    }
}
impl SimdBitNot<i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn bitnot<'a,const N: usize>(&self,v: &Vector<'a,i8,N,Self::Backend>)
                                 -> OwnedVector<i8,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i8; N]);

        unsafe {
            let mask = _mm256_set1_epi32(-1);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i8>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i8>>::LANES));

                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i + j * <Self as SimdLanes::<i8>>::LANES),rr);

                    i += <Self as SimdLanes::<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i8>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i8>>::LANES <= N {
                    let ra = self.load(pa.add(i));

                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i8>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i8>>::LANES != 0 {
                for j in i..N {
                    rs[j] = !v[j];
                }
            }
        }

        rs
    }
}
impl SimdBitNot<i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn bitnot<'a,const N: usize>(&self,v: &Vector<'a,i16,N,Self::Backend>)
                                 -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let mask = _mm256_set1_epi32(-1);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i16>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i16>>::LANES));

                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i + j * <Self as SimdLanes::<i16>>::LANES),rr);

                    i += <Self as SimdLanes::<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i16>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i16>>::LANES <= N {
                    let ra = self.load(pa.add(i));

                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i16>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i16>>::LANES != 0 {
                for j in i..N {
                    rs[j] = !v[j];
                }
            }
        }

        rs
    }
}
impl SimdBitNot<i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn bitnot<'a,const N: usize>(&self,v: &Vector<'a,i32,N,Self::Backend>)
                                 -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let mask = _mm256_set1_epi32(-1);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));

                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),rr);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let ra = self.load(pa.add(i));

                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = !v[j];
                }
            }
        }

        rs
    }
}
impl SimdBitNot<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;

    fn bitnot<'a,const N: usize>(&self,v: &Vector<'a,f32,N,Self::Backend>)
                                 -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let mask = _mm256_set1_epi32(-1);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f32>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f32>>::LANES));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i + j * <Self as SimdLanes::<f32>>::LANES),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f32>>::LANES * <Self as SimdRows::<f32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f32>>::LANES <= N {
                    let ra = self.load(pa.add(i));

                    let ra = _mm256_castps_si256(ra);
                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i),_mm256_castsi256_ps(rr));

                    i += <Self as SimdLanes::<f32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f32::from_bits(!v[j].to_bits());
                }
            }
        }

        rs
    }
}
impl SimdBitNot<f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn bitnot<'a,const N: usize>(&self,v: &Vector<'a,f64,N,Self::Backend>)
                                 -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let mask = _mm256_set1_epi32(-1);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<f64>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes::<f64>>::LANES));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i + j * <Self as SimdLanes::<f64>>::LANES),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<f64>>::LANES * <Self as SimdRows::<f64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<f64>>::LANES <= N {
                    let ra = self.load(pa.add(i));

                    let ra = _mm256_castpd_si256(ra);
                    let rr = _mm256_andnot_si256(ra,mask);

                    self.store(po.add(i),_mm256_castsi256_pd(rr));

                    i += <Self as SimdLanes::<f64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<f64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f64::from_bits(!v[j].to_bits());
                }
            }
        }

        rs
    }
}
impl SimdShiftLeft<i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn shl<'a,const N: usize>(&self,v: &Vector<'a,i8,N,Self::Backend>,w:usize)
        -> OwnedVector<i8,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i8; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v8 = _mm_loadl_epi64(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *const __m128i);
                    let v16 = _mm_cvtepi8_epi16(v8);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_sll_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);
                    let s8 = _mm_packus_epi16(s16, s16);

                    _mm_storel_epi64(po.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *mut __m128i,s8);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v8 = _mm_loadl_epi64(pa.add(i) as *const __m128i);
                    let v16 = _mm_cvtepi8_epi16(v8);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_sll_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);
                    let s8 = _mm_packus_epi16(s16, s16);

                    _mm_storel_epi64(po.add(i) as *mut __m128i,s8);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j] << w;
                }
            }
        }

        rs
    }
}
impl SimdShiftLeft<i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn shl<'a,const N: usize>(&self,v: &Vector<'a,i16,N,Self::Backend>,w:usize)
        -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v16 = _mm_loadl_epi64(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *const __m128i);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_sll_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);

                    _mm_storel_epi64(po.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *mut __m128i,s16);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v8 = _mm_loadl_epi64(pa.add(i) as *const __m128i);
                    let v16 = _mm_cvtepi8_epi16(v8);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_sll_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);

                    _mm_storel_epi64(po.add(i) as *mut __m128i,s16);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j] << w;
                }
            }
        }

        rs
    }
}
impl SimdShiftLeft<i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn shl<'a,const N: usize>(&self,v: &Vector<'a,i32,N,Self::Backend>,w:usize)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v32 = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let s32 = _mm256_sll_epi32(v32,rw);

                   self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),s32);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v32 = self.load(pa.add(i));
                    let s32 = _mm256_sll_epi32(v32,rw);

                    self.store(po.add(i),s32);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j] << w;
                }
            }
        }

        rs
    }
}
impl SimdShiftLeft<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;
    fn shl<'a,const N: usize>(&self,v: &Vector<'a,f32,N,Self::Backend>,w:usize)
                              -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v32 = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let bits = _mm256_castps_si256(v32);
                    let s32 = _mm256_sll_epi32(bits,rw);
                    let o = _mm256_castsi256_ps(s32);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),o);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v32 = self.load(pa.add(i));
                    let bits = _mm256_castps_si256(v32);
                    let s32 = _mm256_sll_epi32(bits,rw);
                    let o = _mm256_castsi256_ps(s32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f32::from_bits(rs[j].to_bits() << w);
                }
            }
        }

        rs
    }
}
impl SimdShiftLeft<f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn shl<'a,const N: usize>(&self,v: &Vector<'a,f64,N,Self::Backend>,w:usize)
        -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i64>>::LANES * <Self as SimdRows::<i64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i64>>::ROWS {
                    let v64 = self.load(pa.add(i + j * <Self as SimdLanes::<i64>>::LANES));
                    let bits = _mm256_castpd_si256(v64);
                    let s64 = _mm256_sll_epi64(bits,rw);
                    let o = _mm256_castsi256_pd(s64);

                    self.store(po.add(i + j * <Self as SimdLanes::<i64>>::LANES),o);

                    i += <Self as SimdLanes::<i64>>::LANES * <Self as SimdRows::<i64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i64>>::LANES * <Self as SimdRows::<i64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i64>>::LANES <= N {
                    let v64 = self.load(pa.add(i));
                    let bits = _mm256_castpd_si256(v64);
                    let s64 = _mm256_sll_epi64(bits,rw);
                    let o = _mm256_castsi256_pd(s64);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f64::from_bits(rs[j].to_bits() << w);
                }
            }
        }

        rs
    }
}
impl SimdShiftRight<i8> for Avx2
    where Self: SimdReg<i8> +
                SimdLanes<i8> +
                SimdRows<i8> +
                SimdMask<i8> {
    type Backend = Avx2;
    fn shr<'a,const N: usize>(&self,v: &Vector<'a,i8,N,Self::Backend>,w:usize)
                              -> OwnedVector<i8,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i8; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v8 = _mm_loadl_epi64(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *const __m128i);
                    let v16 = _mm_cvtepi8_epi16(v8);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_srl_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);
                    let s8 = _mm_packus_epi16(s16, s16);

                    _mm_storel_epi64(po.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *mut __m128i,s8);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i8>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v8 = _mm_loadl_epi64(pa.add(i) as *const __m128i);
                    let v16 = _mm_cvtepi8_epi16(v8);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_srl_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);
                    let s8 = _mm_packus_epi16(s16, s16);

                    _mm_storel_epi64(po.add(i) as *mut __m128i,s8);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j] >> w;
                }
            }
        }

        rs
    }
}
impl SimdShiftRight<i16> for Avx2
    where Self: SimdReg<i16> +
                SimdLanes<i16> +
                SimdRows<i16> +
                SimdMask<i16> {
    type Backend = Avx2;
    fn shr<'a,const N: usize>(&self,v: &Vector<'a,i16,N,Self::Backend>,w:usize)
                              -> OwnedVector<i16,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i16; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v16 = _mm_loadl_epi64(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *const __m128i);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_srl_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);

                    _mm_storel_epi64(po.add(i + j * <Self as SimdLanes::<i32>>::LANES) as *mut __m128i,s16);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i16>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v8 = _mm_loadl_epi64(pa.add(i) as *const __m128i);
                    let v16 = _mm_cvtepi8_epi16(v8);
                    let v32 = _mm256_cvtepi16_epi32(v16);
                    let s32 = _mm256_srl_epi32(v32,rw);
                    let s16 = _mm256_cvtepi32_epi16(s32);

                    _mm_storel_epi64(po.add(i) as *mut __m128i,s16);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j] >> w;
                }
            }
        }

        rs
    }
}
impl SimdShiftRight<i32> for Avx2
    where Self: SimdReg<i32> +
                SimdLanes<i32> +
                SimdRows<i32> +
                SimdMask<i32> {
    type Backend = Avx2;
    fn shr<'a,const N: usize>(&self,v: &Vector<'a,i32,N,Self::Backend>,w:usize)
                              -> OwnedVector<i32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0i32; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v32 = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let s32 = _mm256_srl_epi32(v32,rw);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),s32);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v32 = self.load(pa.add(i));
                    let s32 = _mm256_srl_epi32(v32,rw);

                    self.store(po.add(i),s32);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j] >> w;
                }
            }
        }

        rs
    }
}
impl SimdShiftRight<f32> for Avx2
    where Self: SimdReg<f32> +
                SimdLanes<f32> +
                SimdRows<f32> +
                SimdMask<f32> {
    type Backend = Avx2;
    fn shr<'a,const N: usize>(&self,v: &Vector<'a,f32,N,Self::Backend>,w:usize)
                              -> OwnedVector<f32,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f32; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i32>>::ROWS {
                    let v32 = self.load(pa.add(i + j * <Self as SimdLanes::<i32>>::LANES));
                    let bits = _mm256_castps_si256(v32);
                    let s32 = _mm256_srl_epi32(bits,rw);
                    let o = _mm256_castsi256_ps(s32);

                    self.store(po.add(i + j * <Self as SimdLanes::<i32>>::LANES),o);

                    i += <Self as SimdLanes::<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i32>>::LANES * <Self as SimdRows::<i32>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i32>>::LANES <= N {
                    let v32 = self.load(pa.add(i));
                    let bits = _mm256_castps_si256(v32);
                    let s32 = _mm256_srl_epi32(bits,rw);
                    let o = _mm256_castsi256_ps(s32);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i32>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i32>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f32::from_bits(rs[j].to_bits() >> w);
                }
            }
        }

        rs
    }
}
impl SimdShiftRight<f64> for Avx2
    where Self: SimdReg<f64> +
                SimdLanes<f64> +
                SimdRows<f64> +
                SimdMask<f64> {
    type Backend = Avx2;
    fn shr<'a,const N: usize>(&self,v: &Vector<'a,f64,N,Self::Backend>,w:usize)
                              -> OwnedVector<f64,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([0f64; N]);

        unsafe {
            let rw = _mm_set1_epi32(w as i32);

            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes::<i64>>::LANES * <Self as SimdRows::<i64>>::ROWS <= N {
                for j in 0..<Self as SimdRows::<i64>>::ROWS {
                    let v64 = self.load(pa.add(i + j * <Self as SimdLanes::<i64>>::LANES));
                    let bits = _mm256_castpd_si256(v64);
                    let s64 = _mm256_srl_epi64(bits,rw);
                    let o = _mm256_castsi256_pd(s64);

                    self.store(po.add(i + j * <Self as SimdLanes::<i64>>::LANES),o);

                    i += <Self as SimdLanes::<i64>>::LANES * <Self as SimdRows::<i64>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<i64>>::LANES * <Self as SimdRows::<i64>>::ROWS) != 0 {
                while i + <Self as SimdLanes::<i64>>::LANES <= N {
                    let v64 = self.load(pa.add(i));
                    let bits = _mm256_castpd_si256(v64);
                    let s64 = _mm256_srl_epi64(bits,rw);
                    let o = _mm256_castsi256_pd(s64);

                    self.store(po.add(i),o);

                    i += <Self as SimdLanes::<i64>>::LANES;
                }
            }

            if N % <Self as SimdLanes::<i64>>::LANES != 0 {
                for j in i..N {
                    rs[j] = f64::from_bits(rs[j].to_bits() >> w);
                }
            }
        }

        rs
    }
}
