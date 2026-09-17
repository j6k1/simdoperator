//! Common Backend Implementation

use std::ops::{Add, Mul, Sub};
use crate::traits::{SimdAddVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdMulVector, SimdSubVector, SimdMask, SimdScalarMulVector, SimdLoad, SimdStore, SimdReg, SimdLanes, SimdRows, SimdAdd, SimdSub, SimdMul, SimdStoreSeq, SimdSplat, SimdCols, BitsBitAnd, BitsBitOr, BitsBitXor, BitsBitNot, SimdBitAndVector, SimdBitAnd, SimdBitOr, SimdBitXor, SimdBitNot, BitsShl, BitsShr, SimdShlVector, SimdShl, SimdShrVector, SimdShr, SimdPromote, SimdPromoteVector, Assume, SimdDemoteVector, SimdDemote, SimdConvertVector, SimdConvert, SupportMul, FoldRegs, SimdOuterProduct, SimdMatMul, SimdMulAssignVector, SimdAddAssignVector, SimdSubAssignVector, SimdShiftWidth, SimdScalarMulAssignVector, SimdAddAssignMatrix, SimdScalarMulAssignMatrix, SimdScalarMulMatrix, SimdConvertMatrix};
use crate::{ColumnMajorMatrix, Matrix, MatrixMut, OwnedMatrix, OwnedVector, Vector, VectorMut};

pub trait Backend {
    fn new() -> Self;
}
#[derive(Clone,Copy)]
pub struct Regs<R,const N:usize> where R: Copy {
    regs:[R;N]
}
impl<R,const N:usize> Regs<R,N> where R: Copy {
    pub fn new(regs:[R;N]) -> Self {
        Regs { regs }
    }
}
impl<R,const N:usize> AsRef<[R;N]> for Regs<R,N>
    where R: Copy {
    fn as_ref(&self) -> &[R;N] {
        &self.regs
    }
}
impl<S,BE: SimdReg<S> + SimdAdd<S,S,S>> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,1>
    where <BE as SimdReg<S>>::Reg: Copy {
    fn fold(&self,_: &BE) -> <BE as SimdReg<S>>::Reg {
        self.regs[0]
    }
}
impl<S,BE: SimdReg<S> + SimdAdd<S,S,S>> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,2>
    where <BE as SimdReg<S>>::Reg: Copy {
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg {
        backend.add(self.regs[0],self.regs[1])
    }
}
impl<S,BE: SimdReg<S> + SimdAdd<S,S,S>> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,4>
    where <BE as SimdReg<S>>::Reg: Copy {
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg {
        backend.add(backend.add(self.regs[0],self.regs[1]),backend.add(self.regs[2],self.regs[3]))
    }
}
impl<T,BE> SimdStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,4>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> {
    #[inline(always)]
    unsafe fn store_seq(&self, ptr: *mut T, regs: Regs<<Self as SimdReg<T>>::Reg,4>) {
        let &[a,b,c,d] = regs.as_ref();

        unsafe {
            self.store(ptr, a);
            self.store(ptr.add(1 * <Self as SimdLanes<T>>::LANES),b);
            self.store(ptr.add(2 * <Self as SimdLanes<T>>::LANES), c);
            self.store(ptr.add(3 * <Self as SimdLanes<T>>::LANES),d);
        }
    }
}
impl<T,BE> SimdStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,2>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> {
    #[inline(always)]
    unsafe fn store_seq(&self, ptr: *mut T, regs: Regs<<Self as SimdReg<T>>::Reg,2>) {
        let &[a,b] = regs.as_ref();

        unsafe {
            self.store(ptr, a);
            self.store(ptr.add(1 * <Self as SimdLanes<T>>::LANES),b);
        }
    }
}
impl<T,BE> SimdStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,1>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> {
    #[inline(always)]
    unsafe fn store_seq(&self, ptr: *mut T, regs: Regs<<Self as SimdReg<T>>::Reg,1>) {
        let &[reg] = regs.as_ref();

        unsafe {
            self.store(ptr, reg);
        }
    }
}
impl<T,BE> SimdAddAssignVector<T,T> for BE
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
    fn add_assign_vector<'a,const N: usize>(&self, l: &mut VectorMut<'a,T,N>, r: &Vector<'a,T,N,Self::Backend>) {
        let mut i = 0;

        unsafe {
            let pa = l.as_mut().as_mut_ptr();
            let pb = r.as_ref().as_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.add(ra,rb);

                self.store(pa.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    l[j] = l[j] + r[j];
                }
            }
        }
    }
}
impl<T,BE> SimdAddVector<T,T,T> for BE
    where BE: Backend +
              SimdAddAssignVector<T,T,Backend=BE>,
              T: Copy {
    type Backend = BE;
    #[inline]
    fn add_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>, r: &Vector<'a,T,N,Self::Backend>)
        -> OwnedVector<T,N> {
        let mut acc = OwnedVector::from(Box::<[T;N]>::from(l));

        let mut l = VectorMut::<T,N>::from(&mut acc);

        <Self as SimdAddAssignVector<T,T>>::add_assign_vector(self,&mut l,r);

        acc
    }
}
impl<T,BE> SimdSubAssignVector<T,T> for BE
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
    fn sub_assign_vector<'a,const N: usize>(&self, l: &mut VectorMut<'a,T,N>, r: &Vector<'a,T,N,Self::Backend>) {
        let mut i = 0;

        unsafe {
            let pa = l.as_mut().as_mut_ptr();
            let pb = r.as_ref().as_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.sub(ra,rb);

                self.store(pa.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    l[j] = l[j] - r[j];
                }
            }
        }
    }
}
impl<T,BE> SimdSubVector<T,T,T> for BE
    where BE: Backend +
              SimdSubAssignVector<T,T,Backend=BE>,
      T: Copy {
    type Backend = BE;
    #[inline]
    fn sub_vector<'a,const N: usize>(&self, l: &Vector<'a,T,N,Self::Backend>, r: &Vector<'a,T,N,Self::Backend>)
        -> OwnedVector<T,N> {
        let mut acc = OwnedVector::from(Box::<[T;N]>::from(l));

        let mut l = VectorMut::<T,N>::from(&mut acc);

        <Self as SimdSubAssignVector<T,T>>::sub_assign_vector(self,&mut l,r);

        acc
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
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
              SO: Default + Copy,
              SL: Mul<SR,Output=SO> + Copy,
              SR: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Heterogeneous> {
    type Backend = BE;
    #[inline]
    fn mul_vector<'a,const N: usize>(&self, l: &Vector<'a,SL,N,Self::Backend>, r: &Vector<'a,SR,N,Self::Backend>)
        -> OwnedVector<SO,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

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
impl<SL,SR,BE> SimdMulAssignVector<SL,SR> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdLanes<SL> +
              SimdRows<SL> +
              SimdMul<SL,SR,SL> +
              SimdLoad<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SL,<Self as SimdMul<SL,SR,SL>>::Output>,
          SL: Mul<SR,Output=SL> + Copy,
          SR: Copy,
          <Self as SimdMul<SL,SR,SL>>::Output: Copy,
          (SL,SL): SupportMul<Homogeneous> {
    type Backend = BE;
    #[inline]
    fn mul_assign_vector<'a,const N: usize>(&self, l: &mut VectorMut<'a,SL,N>, r: &Vector<'a,SR,N,Self::Backend>) {
        let mut i = 0;

        unsafe {
            let pa = l.as_mut().as_mut_ptr();
            let pb = r.as_ref().as_ptr();

            while i + <Self as SimdLanes<SL>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let prod = self.mul(ra,rb);

                self.store_seq(pa.add(i),prod);

                i += <Self as SimdLanes<SL>>::LANES;
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                for j in i..N {
                    l[j] = l[j] * r[j];
                }
            }
        }
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
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
              SO: Default + Copy,
              SL: Mul<SR,Output=SO> + Copy,
              SR: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Homogeneous> {
    type Backend = BE;
    #[inline]
    fn mul_vector<'a,const N: usize>(&self, l: &Vector<'a,SL,N,Self::Backend>, r: &Vector<'a,SR,N,Self::Backend>)
        -> OwnedVector<SO,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

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
              <Self as SimdReg<SL>>::Reg: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Heterogeneous> {
    type Backend = BE;
    #[inline]
    fn scalarmul_vector<'a, const N: usize>(&self, l:SL, r: &Vector<'a, SR, N, Self::Backend>) -> OwnedVector<SO, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

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
impl<SL,SR,BE> SimdScalarMulAssignVector<SL,SR> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SR> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SR,<Self as SimdMul<SL,SR,SR>>::Output>,
      SL: Mul<SR,Output=SR> + Copy,
      SR: Copy,
      <Self as SimdReg<SR>>::Reg: Copy,
      <Self as SimdMul<SL,SR,SR>>::Output: Copy,
      (SL,SR): SupportMul<Homogeneous> {
    type Backend = BE;
    #[inline]
    fn scalarmul_assign_vector<'a, const N: usize>(&self, l:SL, r: &mut VectorMut<'a, SR, N>) {
        let mut i = 0;

        unsafe {
            let s = self.splat(l);
            let pb = r.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS <= N {
                for j in 0..<Self as SimdCols<SL>>::COLS {
                    let rr = self.load(pb.add(i + j * <Self as SimdLanes<SL>>::LANES));

                    let o = self.mul(s,rr);

                    self.store_seq(pb.add(i + j * <Self as SimdLanes<SL>>::LANES),o);
                }

                i += <Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS;
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                while i + <Self as SimdLanes<SL>>::LANES <= N {
                    let rr = self.load(pb.add(i));

                    let o = self.mul(s,rr);

                    self.store_seq(pb.add(i),o);

                    i += <Self as SimdLanes<SL>>::LANES;
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                for j in i..N {
                    r[j] = l * r[j];
                }
            }
        }
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
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
          SO: Default + Copy,
          SL: Mul<SR,Output=SO> + Copy,
          SR: Copy,
          <Self as SimdReg<SL>>::Reg: Copy,
          <Self as SimdMul<SL,SR,SO>>::Output: Copy,
          (SL,SO): SupportMul<Homogeneous> {
    type Backend = BE;
    #[inline]
    fn scalarmul_vector<'a, const N: usize>(&self, l:SL, r: &Vector<'a, SR, N, Self::Backend>) -> OwnedVector<SO, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

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

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

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

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

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

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

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

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

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
              SimdLanes<T> +
              SimdShiftWidth<T>,
              T: BitsShl + Default + Copy {
    type Backend = BE;
    #[inline]
    fn shl_vector<'a,const N: usize>(&self, v: &Vector<'a,T,N,Self::Backend>, w:usize)
                                     -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        let rw = <BE as SimdShiftWidth<T>>::shift_width(self,w);

        unsafe {
            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let vr = self.load(pa.add(i));
                let rr = self.shl(vr,rw);

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
              SimdLanes<T> +
              SimdShiftWidth<T>,
              T: BitsShr + Default + Copy {
    type Backend = BE;
    #[inline]
    fn shr_vector<'a,const N: usize>(&self, v: &Vector<'a,T,N,Self::Backend>, w:usize)
                                     -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        let rw = <BE as SimdShiftWidth<T>>::shift_width(self,w);

        unsafe {
            let pa = v.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let vr = self.load(pa.add(i));
                let rr = self.shr(vr,rw);

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
impl<SS,SD,BE> SimdPromoteVector<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SS> +
              SimdLanes<SS> +
              SimdPromote<SS,SD> +
              SimdLoad<SS> +
              SimdStoreSeq<SD,<Self as SimdPromote<SS,SD>>::Output> +
              SimdStore<SD>,
          SS: Copy,
          SD: Default + Copy + From<SS>,
          <Self as SimdPromote<SS,SD>>::Output: Copy,
          <Self as SimdReg<SS>>::Reg: Copy {
    type Backend = BE;
    #[inline]
    fn promotion_vector<'a, const N: usize>(&self, s: &Vector<'a, SS, N, Self::Backend>) -> OwnedVector<SD, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SD::default(); N]));

        unsafe {
            let pb = s.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SS>>::LANES <= N {
                let sr = self.load(pb.add(i));

                let o = self.promotion(sr);

                self.store_seq(po.add(i),o);

                i += <Self as SimdLanes<SS>>::LANES;
            }

            if N % <Self as SimdLanes<SS>>::LANES != 0 {
                for j in i..N {
                    rs[j] = s[j].into();
                }
            }
        }

        rs
    }
}
impl<SS,SD,BE> SimdDemoteVector<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SS> +
              SimdLanes<SS> +
              SimdDemote<SS,SD> +
              SimdLoad<SS> +
              SimdStoreSeq<SD,<Self as SimdDemote<SS,SD>>::Output> +
              SimdStore<SD>,
          SS: Assume<SD> + Copy,
          SD: Default + Copy,
          <Self as SimdDemote<SS,SD>>::Output: Copy,
          <Self as SimdReg<SS>>::Reg: Copy {
    type Backend = BE;
    #[inline]
    fn demotion_vector<'a, const N: usize>(&self, s: &Vector<'a, SS, N, Self::Backend>) -> OwnedVector<SD, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SD::default(); N]));

        unsafe {
            let pb = s.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SS>>::LANES <= N {
                let sr = self.load(pb.add(i));

                let o = self.demotion(sr);

                self.store_seq(po.add(i),o);

                i += <Self as SimdLanes<SS>>::LANES;
            }

            if N % <Self as SimdLanes<SS>>::LANES != 0 {
                for j in i..N {
                    rs[j] = s[j].assume();
                }
            }
        }

        rs
    }
}
impl<SS,SD,BE> SimdConvertVector<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SS> +
              SimdLanes<SS> +
              SimdConvert<SS,SD> +
              SimdLoad<SS> +
              SimdStoreSeq<SD,<Self as SimdConvert<SS,SD>>::Output> +
              SimdStore<SD>,
          SS: Assume<SD> + Copy,
          SD: Default + Copy,
          <Self as SimdConvert<SS,SD>>::Output: Copy,
          <Self as SimdReg<SS>>::Reg: Copy {
    type Backend = BE;
    #[inline]
    fn convert_vector<'a, const N: usize>(&self, s: &Vector<'a, SS, N, Self::Backend>) -> OwnedVector<SD, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SD::default(); N]));

        unsafe {
            let pb = s.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SS>>::LANES <= N {
                let sr = self.load(pb.add(i));

                let o = self.convert(sr);

                self.store_seq(po.add(i),o);

                i += <Self as SimdLanes<SS>>::LANES;
            }

            if N % <Self as SimdLanes<SS>>::LANES != 0 {
                for j in i..N {
                    rs[j] = s[j].assume();
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdAddAssignMatrix<T,T> for BE
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
    fn add_assign_matrix<'a,const N: usize,const M: usize>(&self, l: &'a mut MatrixMut<'a,T,N,M>, r: &Matrix<'a,T,N,M,Self::Backend>) {
        unsafe {
            for i in 0..N {
                let rb = r.row(i);
                let pa = l.as_mut().as_mut_ptr().add(i * M);
                let pb = rb.as_ref().as_ptr();

                let mut j = 0;

                while j + <Self as SimdLanes<T>>::LANES <= M {
                    let ra = self.load(pa.add(j));
                    let rb = self.load(pb.add(j));

                    let rr = self.add(ra,rb);

                    self.store(pa.add(j),rr);

                    j += <Self as SimdLanes<T>>::LANES;
                }

                if M % <Self as SimdLanes<T>>::LANES != 0 {
                    for k in j..M {
                        l[(i,k)] = l[(i,k)] + r[i][k];
                    }
                }
            }
        }
    }
}
impl<T,BE> SimdScalarMulAssignMatrix<T,T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdMul<T,T,T> +
              SimdSplat<T> +
              SimdLoad<T> +
              SimdStoreSeq<T,<Self as SimdMul<T,T,T>>::Output>,
          T: Default + Mul<Output=T> + Copy,
          <Self as SimdMul<T,T,T>>::Output: Copy,
          (T,T): SupportMul<Homogeneous> {
    type Backend = BE;
    #[inline]
    fn scalar_mul_assign_matrix<'a,const N: usize,const M: usize>(&self, l: T, r: &'a mut MatrixMut<'a,T,N,M>) {
        unsafe {
            let rl = <Self as SimdSplat<T>>::splat(self,l);

            for i in 0..N {
                let pb = r.as_mut().as_mut_ptr().add(i * M);

                let mut j = 0;

                while j + <Self as SimdLanes<T>>::LANES <= M {
                    let rb = self.load(pb.add(j));

                    let rr = self.mul(rl,rb);

                    self.store_seq(pb.add(j),rr);

                    j += <Self as SimdLanes<T>>::LANES;
                }

                if M % <Self as SimdLanes<T>>::LANES != 0 {
                    for k in j..M {
                        r[(i,k)] = l * r[(i,k)];
                    }
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdScalarMulMatrix<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SO> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdLoad<SO> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
          SO: Default + Copy,
          SL: Mul<SR,Output=SO> + Copy,
          SR: Copy,
          <Self as SimdReg<SL>>::Reg: Copy,
          <Self as SimdMul<SL,SR,SO>>::Output: Copy,
          (SL,SO): SupportMul<Homogeneous> {
    type Backend = BE;
    #[inline]
    fn scalar_mul_matrix<'a,const N: usize,const M: usize>(&self, l: SL, r: &Matrix<'a,SR,N,M,BE>, acc:&'a mut MatrixMut<'a,SO,N,M>) {
        unsafe {
            let rl = <Self as SimdSplat<SL>>::splat(self,l);

            for i in 0..N {
                let rrow = r.row(i);
                let pacc = acc.as_mut().as_mut_ptr().add(i * M);

                let mut j = 0;

                while j + <Self as SimdLanes<SL>>::LANES <= M {
                    let rb = self.load(rrow.as_ref().as_ptr().add(j));

                    let rr = self.mul(rl,rb);

                    self.store_seq(pacc.add(j),rr);

                    j += <Self as SimdLanes<SL>>::LANES;
                }

                if M % <Self as SimdLanes<SL>>::LANES != 0 {
                    for k in j..M {
                        acc[(i,k)] = l * r[i][k];
                    }
                }
            }
        }
    }
}
impl<SS,SD,BE> SimdConvertMatrix<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SS> +
              SimdLanes<SS> +
              SimdConvert<SS,SD> +
              SimdLoad<SS> +
              SimdStoreSeq<SD,<Self as SimdConvert<SS,SD>>::Output> +
              SimdStore<SD>,
          SS: Assume<SD> + Copy,
          SD: Default + Copy,
          <Self as SimdConvert<SS,SD>>::Output: Copy,
          <Self as SimdReg<SS>>::Reg: Copy {
    type Backend = BE;
    #[inline]
    fn convert_matrix<'a, const N: usize,const M: usize>(&self, s: &Matrix<'a, SS, N, M, Self::Backend>, acc:&'a mut MatrixMut<'a,SD,N,M>) {
        unsafe {
            for i in 0..N {
                let lr = s.row(i);
                let pa = lr.as_ref().as_ptr();
                let mut po = acc.as_mut().as_mut_ptr().add(i * M);

                let mut j = 0;

                while j + <Self as SimdLanes<SS>>::LANES <= N {
                    let sr = self.load(pa.add(j));

                    let o = self.convert(sr);

                    self.store_seq(po.add(i),o);

                    j += <Self as SimdLanes<SS>>::LANES;
                }

                if M % <Self as SimdLanes<SS>>::LANES != 0 {
                    for k in j..M {
                        acc[(i,k)] = lr[k].assume();
                    }
                }
            }
        }
    }
}
impl<SL, SR,SO,BE> SimdOuterProduct<SL,SR,SO> for BE
    where BE: Backend +
              SimdMatMul<SL,SR,SO,Backend=BE> {
    type Backend = BE;

    fn outer_product<'a, const N: usize, const M: usize>(&self, l: &Vector<'a, SL, N, Self::Backend>,
                                                         r: &Vector<'a, SR, M, Self::Backend>,
                                                         o: &mut OwnedMatrix<SO, N, M>) {
        let mut o = o.into();
        let l = Matrix::from(l);
        let r = ColumnMajorMatrix::from(r);

        <Self as SimdMatMul<SL,SR,SO>>::matmul::<N,M,1>(self,&l,&r,&mut o)
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
impl<T> Assume<T> for T {
    #[inline(always)]
    fn assume(self) -> T {
        self
    }
}
impl<T> Assume<T> for &T where T: Assume<T> + Copy {
    #[inline(always)]
    fn assume(self) -> T {
        *self
    }
}
impl Assume<f32> for f64 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i8> for f64 {
    #[inline(always)]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<i16> for f64 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<f64> for f32 {
    #[inline(always)]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<i16> for f32 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<i8> for f32 {
    #[inline(always)]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<f64> for i8 {
    #[inline(always)]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<f32> for i8 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i16> for i8 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<i32> for i8 {
    #[inline(always)]
    fn assume(self) -> i32 {
        self as i32
    }
}
impl Assume<f32> for i16 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<f64> for i16 {
    #[inline(always)]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<i32> for i16 {
    #[inline(always)]
    fn assume(self) -> i32 {
        self as i32
    }
}
impl Assume<i8> for i16 {
    #[inline(always)]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<i16> for i32 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
pub enum Homogeneous {}
pub enum Heterogeneous {}
impl SupportMul<Heterogeneous> for (i8,i32) {}
impl SupportMul<Heterogeneous> for (i16,i32) {}
impl<T> SupportMul<Homogeneous> for (T,T) {}