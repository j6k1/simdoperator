//! Common Backend Implementation

use std::ops::{Add, Mul, Sub};
use crate::traits::{SimdAddVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdMulVector, SimdSubVector, SimdMask, SimdScalarMulVector, SimdLoad, SimdStore, SimdReg, SimdLanes, SimdRows, SimdAdd, SimdSub, SimdMul, SimdStoreSeq, SimdSplat, SimdCols, BitsBitAnd, BitsBitOr, BitsBitXor, BitsBitNot, SimdBitAndVector, SimdBitAnd, SimdBitOr, SimdBitXor, SimdBitNot, BitsShl, BitsShr, SimdShlVector, SimdShl, SimdShrVector, SimdShr, SimdPromote, SimdPromoteVector, Assume, SimdDemoteVector, SimdDemote, SimdConvertVector, SimdConvert, SupportMul, FoldRegs, SimdOuterProduct, SimdMatMul, SimdMulAssignVector, SimdAddAssignVector, SimdSubAssignVector, SimdShiftWidth, SimdScalarMulAssignVector, SimdAddAssignMatrix, SimdScalarMulAssignMatrix, SimdScalarMulMatrix, SimdConvertMatrix, SimdLoadSeq, SimdScalarMulVectorInto};
use crate::{ColumnMajorMatrix, MatrixMut, MatrixView, OwnedMatrix, OwnedVector, VectorMut, VectorView};
use crate::error::InstantiationError;

/// A trait that defines the instantiation of a SIMD arithmetic backend
pub trait Backend: 'static + Sized {
    fn new() -> Result<Self,InstantiationError>;
}
/// A set of one or more registers
#[derive(Clone,Copy)]
pub struct Regs<R,const N:usize> where R: Copy {
    regs:[R;N]
}
impl<R,const N:usize> Regs<R,N> where R: Copy {
    #[inline(always)]
    pub fn new(regs:[R;N]) -> Self {
        Regs { regs }
    }
}
impl<R,const N:usize> AsRef<[R;N]> for Regs<R,N>
    where R: Copy {
    #[inline(always)]
    fn as_ref(&self) -> &[R;N] {
        &self.regs
    }
}
impl<S,BE: SimdReg<S> + SimdAdd<S,S,S>> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,1>
    where <BE as SimdReg<S>>::Reg: Copy {
    #[inline(always)]
    fn fold(&self,_: &BE) -> <BE as SimdReg<S>>::Reg {
        self.regs[0]
    }
}
impl<S,BE: SimdReg<S> + SimdAdd<S,S,S>> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,2>
    where <BE as SimdReg<S>>::Reg: Copy {
    #[inline(always)]
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg {
        backend.add(self.regs[0],self.regs[1])
    }
}
impl<S,BE: SimdReg<S> + SimdAdd<S,S,S>> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,4>
    where <BE as SimdReg<S>>::Reg: Copy {
    #[inline(always)]
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg {
        backend.add(backend.add(self.regs[0],self.regs[1]),backend.add(self.regs[2],self.regs[3]))
    }
}
impl<T,BE> SimdLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,1>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> {
    #[inline(always)]
    unsafe fn load_seq(&self, ptr: *const T) -> Regs<<Self as SimdReg<T>>::Reg,1> {
        let reg = unsafe { self.load(ptr) };
        Regs::new([reg])
    }
}
impl<T,BE> SimdLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,2>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> {
    #[inline(always)]
    unsafe fn load_seq(&self, ptr: *const T) -> Regs<<Self as SimdReg<T>>::Reg,2> {
        let a = unsafe { self.load(ptr) };
        let b = unsafe { self.load(ptr.add(<Self as SimdLanes<T>>::LANES)) };
        Regs::new([a,b])
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
    #[target_feature(enable = "avx2")]
    unsafe fn add_assign_vector<'a,const N: usize>(&self, l: &mut VectorMut<'a,T,N>, r: &VectorView<'a,T,N>) {
        unsafe {
            let mut ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let mut chunks_l = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES);

            for (mut cl,cr) in chunks_l.zip(chunks_r) {
                let ra = self.load(cl.as_mut_ptr());
                let rb = self.load(cr.as_ptr());

                let rr = self.add(ra,rb);

                self.store(cl.as_mut_ptr(),rr);
            }

            let mut chuks_l_remainder = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();
            let chuks_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for (l,r) in chuks_l_remainder.iter_mut().zip(chuks_r_remainder.iter()) {
                    *l = *l + *r;
                }
            }
        }
    }
}
impl<T,BE> SimdAddVector<T,T,T> for BE
    where BE: Backend +
              SimdAddAssignVector<T,T>,
              T: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn add_vector<'a,const N: usize>(&self, l: &VectorView<'a,T,N>, r: &VectorView<'a,T,N>)
        -> OwnedVector<T,N> {
        let mut acc = OwnedVector::from(Box::<[T;N]>::from(l));

        let mut l = VectorMut::<T,N>::from(&mut acc);

        unsafe { <Self as SimdAddAssignVector<T,T>>::add_assign_vector(self,&mut l,&r) };

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
    #[target_feature(enable = "avx2")]
    unsafe fn sub_assign_vector<'a,const N: usize>(&self, l: &mut VectorMut<'a,T,N>, r: &VectorView<'a,T,N>) {
        unsafe {
            let mut ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let mut chunks_l = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES);

            for (mut cl,cr) in chunks_l.zip(chunks_r) {
                let ra = self.load(cl.as_mut_ptr());
                let rb = self.load(cr.as_ptr());

                let rr = self.sub(ra,rb);

                self.store(cl.as_mut_ptr(),rr);
            }

            let mut chuks_l_remainder = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();
            let chuks_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for (l,r) in chuks_l_remainder.iter_mut().zip(chuks_r_remainder.iter()) {
                    *l = *l - *r;
                }
            }
        }
    }
}
impl<T,BE> SimdSubVector<T,T,T> for BE
    where BE: Backend +
              SimdSubAssignVector<T,T>,
      T: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn sub_vector<'a,const N: usize>(&self, l: &VectorView<'a,T,N>, r: &VectorView<'a,T,N>)
        -> OwnedVector<T,N> {
        let mut acc = OwnedVector::from(Box::<[T;N]>::from(l));

        let mut l = VectorMut::<T,N>::from(&mut acc);

        unsafe { <Self as SimdSubAssignVector<T,T>>::sub_assign_vector(self,&mut l,r) };

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
              SO: Default + From<SL> + From<SR> + Mul<SO,Output=SO> + Copy,
              SL: Copy,
              SR: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Heterogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn mul_vector<'a,const N: usize>(&self, l: &VectorView<'a,SL,N>, r: &VectorView<'a,SR,N>)
        -> OwnedVector<SO,N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let ref_l = l.as_ref();
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_l = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);

            for (co,(cl,cr)) in chunks_o.zip(chunks_l.zip(chunks_r)) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let prod = self.mul(ra,rb);

                self.store_seq(co.as_mut_ptr(),prod);
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunck_l_remainder = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunck_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let mut chunck_o_remainder = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,(&l,&r)) in chunck_o_remainder.iter_mut().zip(chunck_l_remainder.iter().zip(chunck_r_remainder.iter())) {
                    *o = SO::from(l) * SO::from(r);
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
    #[target_feature(enable = "avx2")]
    unsafe fn mul_assign_vector<'a,const N: usize>(&self, l: &mut VectorMut<'a,SL,N>, r: &VectorView<'a,SR,N>) {
        unsafe {
            let mut ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let mut chunks_l = ref_l.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES);

            for (mut cl,cr) in chunks_l.zip(chunks_r) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let prod = self.mul(ra,rb);

                self.store_seq(cl.as_mut_ptr(),prod);
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let mut chunck_l_remainder = ref_l.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();
                let chunck_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();

                for (l,r) in chunck_l_remainder.iter_mut().zip(chunck_r_remainder.iter()) {
                    *l = *l * *r;
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
    #[target_feature(enable = "avx2")]
    unsafe fn mul_vector<'a,const N: usize>(&self, l: &VectorView<'a,SL,N>, r: &VectorView<'a,SR,N>)
        -> OwnedVector<SO,N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let ref_l = l.as_ref();
            let ref_r = r.as_ref();
            let mut ref_o = rs.as_mut();

            let chunks_l = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);

            for (co,(cl,cr)) in chunks_o.zip(chunks_l.zip(chunks_r)) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let prod = self.mul(ra,rb);

                self.store_seq(co.as_mut_ptr(),prod);
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunck_l_remainder = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunck_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let mut chunck_o_remainder = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,(&l,&r)) in chunck_o_remainder.iter_mut().zip(chunck_l_remainder.iter().zip(chunck_r_remainder.iter())) {
                    *o = l * r;
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
              SO: Default + From<SL> + From<SR> + Mul<SO,Output=SO> + Copy,
              SL: Copy,
              SR: Copy,
              <Self as SimdReg<SL>>::Reg: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Heterogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector<'a, const N: usize>(&self, l:SL, r: &VectorView<'a, SR, N>) -> OwnedVector<SO, N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let s = self.splat(l);
            let ref_r = r.as_ref();
            let mut ref_o = rs.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            for (co,cr) in chunks_o.zip(chunks_r) {
                for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (o,r) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(r.as_ptr());

                    let rs = self.mul(s,rr);

                    self.store_seq(o.as_mut_ptr(),rs);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,r) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *o = SO::from(l) * SO::from(*r);
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
      SL: Copy,
      SR: From<SL> + Mul<Output=SR> + Copy,
      SR: Copy,
      <Self as SimdReg<SR>>::Reg: Copy,
      <Self as SimdMul<SL,SR,SR>>::Output: Copy,
      (SL,SR): SupportMul<Homogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_assign_vector<'a, const N: usize>(&self, l:SL, r: &mut VectorMut<'a, SR, N>) {
        unsafe {
            let s = self.splat(l);
            let ref_r = r.as_mut();

            let mut chunks_r = ref_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);

            for cr in chunks_r {
                for cr in cr.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES) {
                    let rr = self.load(cr.as_ptr());

                    let rs = self.mul(s,rr);

                    self.store_seq(cr.as_mut_ptr(),rs);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let mut chunks_r = ref_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for cr in chunks_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(cr.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let mut chunks_r = ref_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for r in chunks_r.iter_mut() {
                    *r = SR::from(l) * *r;
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
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector<'a, const N: usize>(&self, l:SL, r: &VectorView<'a, SR, N>) -> OwnedVector<SO, N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            let s = self.splat(l);

            for (co,cr) in chunks_o.zip(chunks_r) {
                for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (co,cr) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (co,cr) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *co = l * *cr;
                }
            }
        }

        rs
    }
}
impl<SL,SR,SO,BE> SimdScalarMulVectorInto<SL,SR,SO> for BE
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
          SO: Default + From<SL> + From<SR> + Mul<SO,Output=SO> + Copy,
          SL: Copy,
          SR: Copy,
          <Self as SimdReg<SL>>::Reg: Copy,
          <Self as SimdMul<SL,SR,SO>>::Output: Copy,
          (SL,SO): SupportMul<Heterogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector_into<'a, const N: usize>(&self, l:SL, r: &VectorView<'a, SR, N>, o: &mut VectorMut<'a, SO, N>) {
        unsafe {
            let s = self.splat(l);
            let ref_r = r.as_ref();
            let ref_o = o.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            for (co,cr) in chunks_o.zip(chunks_r) {
                for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (co,cr) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,r) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *o = SO::from(l) * SO::from(*r);
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdScalarMulVectorInto<SL,SR,SO> for BE
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
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector_into<'a, const N: usize>(&self, l:SL, r: &VectorView<'a, SR, N>, o: &mut VectorMut<'a, SO, N>) {
        unsafe {
            let s = self.splat(l);
            let ref_r = r.as_ref();
            let ref_o = o.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let mut chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            for (co,cr) in chunks_o.zip(chunks_r) {
                 for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let mut chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (co,cr) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let mut chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,r) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *o = l * *r;
                }
            }
        }
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
    #[target_feature(enable = "avx2")]
    unsafe fn bitand_vector<'a,const N: usize>(&self, l: &VectorView<'a,T,N>, r: &VectorView<'a,<T as BitsBitAnd>::Bits,N>)
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
    #[target_feature(enable = "avx2")]
    unsafe fn bitor_vector<'a,const N: usize>(&self, l: &VectorView<'a,T,N>, r: &VectorView<'a,<T as BitsBitOr>::Bits,N>)
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
    #[target_feature(enable = "avx2")]
    unsafe fn bitxor_vector<'a,const N: usize>(&self, l: &VectorView<'a,T,N>, r: &VectorView<'a,<T as BitsBitXor>::Bits,N>)
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
    #[target_feature(enable = "avx2")]
    unsafe fn bitnot_vector<'a,const N: usize>(&self, l: &VectorView<'a,T,N>)
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
    #[target_feature(enable = "avx2")]
    unsafe fn shl_vector<'a,const N: usize>(&self, v: &VectorView<'a,T,N>, w:usize)
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
                    rs[j] = v[j].bits_shl(w);
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
    #[target_feature(enable = "avx2")]
    unsafe fn shr_vector<'a,const N: usize>(&self, v: &VectorView<'a,T,N>, w:usize)
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
                    rs[j] = v[j].bits_shr(w);
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
    #[target_feature(enable = "avx2")]
    unsafe fn promotion_vector<'a, const N: usize>(&self, s: &VectorView<'a, SS, N>) -> OwnedVector<SD, N> {
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
              SimdLanes<SD> +
              SimdDemote<SS,SD> +
              SimdLoadSeq<SS,<Self as SimdDemote<SS,SD>>::Input> +
              SimdStore<SD>,
          SS: Assume<SD> + Copy,
          SD: Default + Copy,
          <Self as SimdDemote<SS,SD>>::Input: Copy,
          <Self as SimdReg<SD>>::Reg: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn demotion_vector<'a, const N: usize>(&self, s: &VectorView<'a, SS, N>) -> OwnedVector<SD, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SD::default(); N]));

        unsafe {
            let pb = s.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SD>>::LANES <= N {
                let sr = self.load_seq(pb.add(i));

                let o = self.demotion(sr);

                self.store(po.add(i),o);

                i += <Self as SimdLanes<SD>>::LANES;
            }

            if N % <Self as SimdLanes<SD>>::LANES != 0 {
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
    #[target_feature(enable = "avx2")]
    unsafe fn convert_vector<'a, const N: usize>(&self, s: &VectorView<'a, SS, N>) -> OwnedVector<SD, N> {
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
    #[target_feature(enable = "avx2")]
    unsafe fn add_assign_matrix<'a,const N: usize,const M: usize>(&self, l: &mut MatrixMut<'a,T,N,M>, r: &MatrixView<'a,T,N,M>) {
        unsafe {
            let mut ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let mut chunks_l = ref_l.chunks_exact_mut(M);
            let chunks_r = ref_r.chunks_exact(M);

            for (cl,cr) in chunks_l.zip(chunks_r) {
                let mut chunks_l = cl.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);
                let chunks_r = cr.chunks_exact(<Self as SimdLanes<T>>::LANES);

                for (cl,cr) in chunks_l.zip(chunks_r) {
                    let ra = self.load(cl.as_ptr());
                    let rb = self.load(cr.as_ptr());

                    let rr = self.add(ra,rb);

                    self.store(cl.as_mut_ptr(),rr);
                }

                if M % <Self as SimdLanes<T>>::LANES != 0 {
                    let chunks_l = cl.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();
                    let chunks_r = cr.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();

                    for (l,r) in chunks_l.iter_mut().zip(chunks_r.iter()) {
                        *l = *l + *r;
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
    #[target_feature(enable = "avx2")]
    unsafe fn scalar_mul_assign_matrix<'a,const N: usize,const M: usize>(&self, l: T, r: &mut MatrixMut<'a,T,N,M>) {
        unsafe {
            let rl = <Self as SimdSplat<T>>::splat(self,l);

            let r_ref = r.as_mut();

            let mut chunks_r = r_ref.chunks_exact_mut(M);

            for cr in chunks_r {
                let mut chunks_r = cr.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);

                for cr in chunks_r {
                    let rb = self.load(cr.as_ptr());

                    let rr = self.mul(rl,rb);

                    self.store_seq(cr.as_mut_ptr(),rr);
                }

                if M % <Self as SimdLanes<T>>::LANES != 0 {
                    let mut chunks_r = cr.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();

                    for r in chunks_r {
                        *r = l * *r;
                    }
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdScalarMulMatrix<SL,SR> for BE
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
    type OutputScalar = SO;
    #[target_feature(enable = "avx2")]
    unsafe fn scalar_mul_matrix<'a,const N: usize,const M: usize>(&self, l: SL, r: &MatrixView<'a,SR,N,M>, acc:&'a mut MatrixMut<'a,SO,N,M>) {
        unsafe {
            let rl = <Self as SimdSplat<SL>>::splat(self,l);

            let r_ref = r.as_ref();
            let mut ref_acc = acc.as_mut();

            let chunks_r = r_ref.chunks_exact(M);
            let acc_chunks_r = ref_acc.chunks_exact_mut(M);

            for (cr,acc_cr) in chunks_r.zip(acc_chunks_r) {
                let chunks_cr = cr.chunks_exact(<Self as SimdLanes<SL>>::LANES);
                let acc_chunks_cr = acc_cr.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);

                for (cr,acc_cr) in chunks_cr.zip(acc_chunks_cr) {
                    let rb = self.load(cr.as_ptr());

                    let rr = self.mul(rl,rb);

                    self.store_seq(acc_cr.as_mut_ptr(),rr);
                }

                if M % <Self as SimdLanes<SL>>::LANES != 0 {
                    let chunks_r = cr.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                    let chunks_acc = acc_cr.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                    for (acc,r) in chunks_acc.iter_mut().zip(chunks_r.iter()) {
                        *acc = l * *r;
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
    #[target_feature(enable = "avx2")]
    unsafe fn convert_matrix<'a, const N: usize,const M: usize>(&self, s: &MatrixView<'a, SS, N, M>, acc:&'a mut MatrixMut<'a,SD,N,M>) {
        unsafe {
            for i in 0..N {
                let lr = s.row(i);
                let pa = lr.as_ref().as_ptr();
                let po = acc.as_mut().as_mut_ptr().add(i * M);

                let mut j = 0;

                while j + <Self as SimdLanes<SS>>::LANES <= M {
                    let sr = self.load(pa.add(j));

                    let o = self.convert(sr);

                    self.store_seq(po.add(j),o);

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
impl<SL,SR,SO,BE> SimdOuterProduct<SL,SR,SO> for BE
    where BE: Backend +
              SimdScalarMulVectorInto<SL,SR,SO>
              /* SimdMatMul<SL,SR,SO> */,
          SL: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn outer_product<'a, const N: usize, const M: usize>(&self, l: &VectorView<'a, SL, N>,
                                                         r: &VectorView<'a, SR, M>,
                                                         o: &mut OwnedMatrix<SO, N, M>) {
        /*
        let mut o = o.into();
        let l = MatrixView::from(l);
        let r = ColumnMajorMatrix::from(r);

        unsafe { <Self as SimdMatMul<SL,SR,SO>>::matmul::<N,M,1>(self,&l,&r,&mut o) }

         */
        unsafe {
            for (row,&l) in o.as_mut().chunks_exact_mut(M).zip(l.as_ref().iter()) {
                <Self as SimdScalarMulVectorInto<SL,SR,SO>>::scalarmul_vector_into(self,l,r,&mut row.try_into().unwrap());
            }
        }
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
        self << w
    }
}
impl BitsShl for i32 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        self << w
    }
}
impl BitsShl for f32 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        f32::from_bits(self.to_bits() << w)
    }
}
impl BitsShl for f64 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        f64::from_bits(self.to_bits() << w)
    }
}
impl BitsShr for i16 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        self >> w
    }
}
impl BitsShr for i32 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        self >> w
    }
}
impl BitsShr for f32 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        f32::from_bits(self.to_bits() >> w)
    }
}
impl BitsShr for f64 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        f64::from_bits(self.to_bits() >> w)
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
impl Assume<f32> for i32 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i16> for i32 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
/// Operations Between Elements of the Same Type
pub enum Homogeneous {}
/// Operations Between Different Types
pub enum Heterogeneous {}
impl SupportMul<Heterogeneous> for (i8,i32) {}
impl SupportMul<Heterogeneous> for (i16,i32) {}
impl<T> SupportMul<Homogeneous> for (T,T) {}
