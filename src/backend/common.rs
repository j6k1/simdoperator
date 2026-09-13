//! Common Backend Implementation

use std::arch::x86_64::{__m128i, _mm256_cvtepi16_epi32, _mm256_cvtepi32_epi16, _mm256_sll_epi32, _mm_cvtepi8_epi16, _mm_loadl_epi64, _mm_packus_epi16, _mm_set1_epi32, _mm_storel_epi64};
use std::ops::{Add, Mul, Sub};
use crate::traits::{SimdAddVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdMulVector, SimdSubVector, SimdMask, SimdScalarMulVector, SimdLoad, SimdStore, SimdReg, SimdLanes, SimdRows, SimdAdd, SimdSub, SimdMul, SimdStoreSeq, SimdSplat, SimdCols, BitsBitAnd, BitsBitOr, BitsBitXor, BitsBitNot, SimdBitAndVector, SimdBitAnd, SimdBitOr, SimdBitXor, SimdBitNot, BitsShl, BitsShr, SimdShlVector, SimdShl, SimdShrVector, SimdShr};
use crate::{OwnedVector, Vector};
use crate::backend::avx2::Avx2;

pub trait Backend {
    fn new() -> Self;
}
impl<T,BE> SimdAddVector<T,T,T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdAdd<T,T,T> +
              SimdLoad<T> +
              SimdStore<T>,
              T: Default + Add<Output=T> + Copy {
    type Backend = BE;
    #[inline]
    fn add_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>, r: &Vector<'a,T,N,Self::Backend>)
        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.add(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] + r[j];
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdSubVector<T,T,T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdSub<T,T,T> +
              SimdLoad<T> +
              SimdStore<T>,
              T: Default + Sub<Output=T> + Copy {
    type Backend = BE;
    #[inline]
    fn sub_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>, r: &Vector<'a,T,N,Self::Backend>)
                                     -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.sub(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] - r[j];
                }
            }
        }

        rs
    }
}
impl<SL,SR,SO,BE> SimdMulVector<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdRows<SL> +
              SimdMul<SL,SR,SO> +
              SimdLoad<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output> +
              SimdStore<SO>,
              SO: Default + Copy,
              SL: Mul<SR,Output=SO> + Copy,
              SR: Copy {
    type Backend = BE;
    #[inline]
    fn mul_vector<'a,const N: usize>(&self, l: &Vector<'a,SL,N,Self::Backend>, r: &Vector<'a,SR,N,Self::Backend>)
                                     -> OwnedVector<SO,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([SO::default(); N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SL>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let prod = self.mul(ra,rb);

                self.store_seq(po.add(i),prod);

                i += <Self as SimdLanes<SL>>::LANES;
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j] * r[j];
                }
            }
        }

        rs
    }
}
impl<SL,SR,SO,BE> SimdScalarMulVector<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SO> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output> +
              SimdStore<SO>,
              SO: Default + Copy,
              SL: Mul<SR,Output=SO> + Copy,
              SR: Copy,
              <Self as SimdReg<SL>>::Reg: Copy {
    type Backend = BE;
    #[inline]
    fn scalarmul_vector<'a, const N: usize>(&self, l:SL, r: &Vector<'a, SR, N, Self::Backend>) -> OwnedVector<SO, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([SO::default(); N]);

        unsafe {
            let s = self.splat(l);
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS <= N {
                for j in 0..<Self as SimdCols<SL>>::COLS {
                    let rr = self.load(pb.add(i + j * <Self as SimdLanes<SL>>::LANES));

                    let o = self.mul(s,rr);

                    self.store_seq(po.add(i + j * <Self as SimdLanes<SL>>::LANES),o);
                }

                i += <Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS;
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                while i + <Self as SimdLanes<SL>>::LANES <= N {
                    let rr = self.load(pb.add(i));

                    let o = self.mul(s,rr);

                    self.store_seq(po.add(i),o);

                    i += <Self as SimdLanes<SL>>::LANES;
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l * r[j];
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdBitAndVector<T> for BE
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitAnd>::Bits> +
              SimdReg<<T as BitsBitAnd>::Bits> +
              SimdLoad<T> +
              SimdLoad<<T as BitsBitAnd>::Bits> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitAnd<T> +,
              T: BitsBitAnd + Default + Copy,
              <T as BitsBitAnd>::Bits: Copy {
    type Backend = BE;
    #[inline]
    fn bitand_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>, r: &Vector<'a,<T as BitsBitAnd>::Bits,N,Self::Backend>)
        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.bitand(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitand(r[j]);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdBitOrVector<T> for BE
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitOr>::Bits> +
              SimdReg<<T as BitsBitOr>::Bits,Reg=<Self as SimdReg<<T as BitsBitOr>::Bits>>::Mask> +
              SimdLoad<T> +
              SimdLoad<<Self as SimdReg<T>>::Bits> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitOr<T>,
              T: BitsBitOr + Default + Copy,
              <T as BitsBitOr>::Bits: Copy {
    type Backend = BE;
    #[inline]
    fn bitor_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>, r: &Vector<'a,<T as BitsBitOr>::Bits,N,Self::Backend>)
                                        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.bitor(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitor(r[j]);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdBitXorVector<T> for BE
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitXor>::Bits> +
              SimdReg<<T as BitsBitXor>::Bits,Reg=<Self as SimdReg<<T as BitsBitXor>::Bits>>::Mask> +
              SimdLoad<T> +
              SimdLoad<<Self as SimdReg<T>>::Bits> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitXor<T>,
              T: BitsBitXor + Default + Copy,
              <T as BitsBitXor>::Bits: Copy {
    type Backend = BE;
    #[inline]
    fn bitxor_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>, r: &Vector<'a,<T as BitsBitXor>::Bits,N,Self::Backend>)
                                       -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.bitxor(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitxor(r[j]);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdBitNotVector<T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLoad<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitNot<T>,
              T: BitsBitNot + Default + Copy {
    type Backend = BE;
    #[inline]
    fn bitnot_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>)
        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = l.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));

                let rr = self.bitnot(ra);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitnot();
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdShlVector<T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLoad<T> +
              SimdStore<T> +
              SimdShl<T> +
              SimdLanes<T>,
              T: BitsShl + Default + Copy {
    type Backend = BE;
    #[inline]
    fn shl_vector<'a,const N: usize>(&self, v: &Vector<'a,T,N,Self::Backend>, w:usize)
                                     -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let vr = self.load(pa.add(i));
                let rr = self.shl(vr,w);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j].bits_shl(w);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdShrVector<T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLoad<T> +
              SimdStore<T> +
              SimdShr<T> +
              SimdLanes<T>,
              T: BitsShr + Default + Copy {
    type Backend = BE;
    #[inline]
    fn shr_vector<'a,const N: usize>(&self, v: &Vector<'a,T,N,Self::Backend>, w:usize)
                                     -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from([T::default(); N]);

        unsafe {
            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let vr = self.load(pa.add(i));
                let rr = self.shr(vr,w);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = rs[j].bits_shr(w);
                }
            }
        }

        rs
    }
}
impl BitsBitAnd for i8 {
    type Bits = i8;

    #[inline(always)]
    fn bits_bitand(self,m:i8) -> Self {
        self & m
    }
}
impl BitsBitAnd for i16 {
    type Bits = i16;
    #[inline(always)]
    fn bits_bitand(self,m:i16) -> Self {
        self & m
    }
}
impl BitsBitAnd for i32 {
    type Bits = i32;
    #[inline(always)]
    fn bits_bitand(self,m:i32) -> Self {
        self & m
    }
}
impl BitsBitAnd for i64 {
    type Bits = i64;
    #[inline(always)]
    fn bits_bitand(self,m:i64) -> Self {
        self & m
    }
}
impl BitsBitAnd for f32 {
    type Bits = u32;

    #[inline(always)]
    fn bits_bitand(self, m: Self::Bits) -> Self {
        f32::from_bits(self.to_bits() & m)
    }
}
impl BitsBitAnd for f64 {
    type Bits = u64;
    #[inline(always)]
    fn bits_bitand(self, m: Self::Bits) -> Self {
        f64::from_bits(self.to_bits() & m)
    }
}
impl BitsBitOr for i8 {
    type Bits = i8;

    #[inline(always)]
    fn bits_bitor(self,m:i8) -> Self {
        self | m
    }
}
impl BitsBitOr for i16 {
    type Bits = i16;
    fn bits_bitor(self,m:i16) -> Self {
        self | m
    }
}
impl BitsBitOr for i32 {
    type Bits = i32;
    #[inline(always)]
    fn bits_bitor(self,m:i32) -> Self {
        self | m
    }
}
impl BitsBitOr for i64 {
    type Bits = i64;
    #[inline(always)]
    fn bits_bitor(self,m:i64) -> Self {
        self | m
    }
}
impl BitsBitOr for f32 {
    type Bits = u32;

    #[inline(always)]
    fn bits_bitor(self, m: Self::Bits) -> Self {
        f32::from_bits(self.to_bits() | m)
    }
}
impl BitsBitOr for f64 {
    type Bits = u64;
    #[inline(always)]
    fn bits_bitor(self, m: Self::Bits) -> Self {
        f64::from_bits(self.to_bits() | m)
    }
}
impl BitsBitXor for i8 {
    type Bits = i8;
    #[inline(always)]
    fn bits_bitxor(self,m:i8) -> Self {
        self ^ m
    }
}
impl BitsBitXor for i16 {
    type Bits = i16;
    #[inline(always)]
    fn bits_bitxor(self,m:i16) -> Self {
        self ^ m
    }
}
impl BitsBitXor for i32 {
    type Bits = i32;
    #[inline(always)]
    fn bits_bitxor(self,m:i32) -> Self {
        self ^ m
    }
}
impl BitsBitXor for i64 {
    type Bits = i64;
    #[inline(always)]
    fn bits_bitxor(self,m:i64) -> Self {
        self ^ m
    }
}
impl BitsBitXor for f32 {
    type Bits = u32;
    #[inline(always)]
    fn bits_bitxor(self, m: Self::Bits) -> Self {
        f32::from_bits(self.to_bits() ^ m)
    }
}
impl BitsBitXor for f64 {
    type Bits = u64;
    #[inline(always)]
    fn bits_bitxor(self, m: Self::Bits) -> Self {
        f64::from_bits(self.to_bits() ^ m)
    }
}
impl BitsBitNot for i8 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for i16 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for i32 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for i64 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for f32 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        f32::from_bits(!self.to_bits())
    }
}
impl BitsBitNot for f64 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        f64::from_bits(!self.to_bits())
    }
}
impl BitsShl for i16 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        self << w as i16
    }
}
impl BitsShl for i32 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        self << w as i32
    }
}
impl BitsShl for f32 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        f32::from_bits(self.to_bits() << w as u32)
    }
}
impl BitsShl for f64 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        f64::from_bits(self.to_bits() << w as u64)
    }
}
impl BitsShr for i16 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        self >> w as i16
    }
}
impl BitsShr for i32 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        self >> w as i32
    }
}
impl BitsShr for f32 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        f32::from_bits(self.to_bits() >> w as u32)
    }
}
impl BitsShr for f64 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        f64::from_bits(self.to_bits() >> w as u64)
    }
}
