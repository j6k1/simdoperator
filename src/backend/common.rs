//! Common Backend Implementation

use std::arch::x86_64::{_mm256_mullo_epi32, _mm256_set1_epi32};
use std::ops::{Add, Mul, Sub};
use crate::backend::avx2::Avx2;
use crate::traits::{SimdAddVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdDot, SimdHSum, SimdMatMul, SimdMatVec, SimdHMax, SimdMulVector, SimdShlVector, SimdShrVector, SimdSubVector, SimdVMat, SimdHMin, SimdMask, SimdScalarMulVector, SimdOuterProduct, SimdLoad, SimdStore, SimdReg, SimdMulAdd, SimdZero, SimdLanes, SimdRows, SimdAdd, SimdSub, SimdMul, SimdStoreSeq, SimdSplat, SimdCols};
use crate::{OwnedVector, Vector};

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

            while i + <Self as SimdLanes<T>>::LANES * <Self as SimdRows<T>>::ROWS <= N {
                for j in 0..<Self as SimdRows<T>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes<T>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes<T>>::LANES));

                    let rr = self.add(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes<T>>::LANES),rr);

                    i += <Self as SimdLanes<T>>::LANES * <Self as SimdRows<T>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<T>>::LANES * <Self as SimdRows<T>>::ROWS) != 0 {
                while i + <Self as SimdLanes<T>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = self.add(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes<T>>::LANES;
                }
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

            while i + <Self as SimdLanes<T>>::LANES * <Self as SimdRows<T>>::ROWS <= N {
                for j in 0..<Self as SimdRows<T>>::ROWS {
                    let ra = self.load(pa.add(i + j * <Self as SimdLanes<T>>::LANES));
                    let rb = self.load(pb.add(i + j * <Self as SimdLanes<T>>::LANES));

                    let rr = self.sub(ra,rb);

                    self.store(po.add(i + j * <Self as SimdLanes<T>>::LANES),rr);

                    i += <Self as SimdLanes<T>>::LANES * <Self as SimdRows<T>>::ROWS;
                }
            }

            if N % (<Self as SimdLanes<T>>::LANES * <Self as SimdRows<T>>::ROWS) != 0 {
                while i + <Self as SimdLanes<T>>::LANES <= N {
                    let ra = self.load(pa.add(i));
                    let rb = self.load(pb.add(i));

                    let rr = self.sub(ra,rb);

                    self.store(po.add(i),rr);

                    i += <Self as SimdLanes<T>>::LANES;
                }
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
