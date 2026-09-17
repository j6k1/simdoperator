//! Trait and data type features for abstracting SIMD operations

use crate::{ColumnMajorMatrix, Matrix, MatrixMut, OwnedMatrix, OwnedVector, Vector, VectorMut};
use crate::backend::common::{Backend};
pub trait SimdAdd<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    type Backend: Backend;

    fn add(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
pub trait SimdSub<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    type Backend: Backend;
    fn sub(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
pub trait SimdMul<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdAdd<SO,SO,SO> + Sized {
    type Backend: Backend;
    type Output: FoldRegs<SO,Self>;
    fn mul(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> Self::Output;
}
pub trait SimdScalarMul<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdMul<SL,SR,SO> {
    type Backend: Backend;
    fn scalarmul(&self,l:SL,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdMul<SL,SR,SO>>::Output;
}
pub trait SimdBitOr<S>: SimdReg<S>
    where S: BitsBitOr,
          Self: SimdReg<<S as BitsBitOr>::Bits> {
    type Backend: Backend;
    fn bitor(&self,l:<Self as SimdReg<S>>::Reg,r:<Self as SimdReg<<S as BitsBitOr>::Bits>>::Reg) -> <Self as SimdReg<S>>::Reg;
}
pub  trait SimdBitAnd<S>: SimdReg<S>
    where S: BitsBitAnd,
          Self: SimdReg<<S as BitsBitAnd>::Bits>{
    type Backend: Backend;
    fn bitand(&self,l:<Self as SimdReg<S>>::Reg,r:<Self as SimdReg<<S as BitsBitAnd>::Bits>>::Reg) -> <Self as SimdReg<S>>::Reg;
}
pub trait SimdBitXor<S>: SimdReg<S>
    where S: BitsBitXor,
          Self: SimdReg<<S as BitsBitXor>::Bits> {
    type Backend: Backend;
    fn bitxor(&self,l:<Self as SimdReg<S>>::Reg,r:<Self as SimdReg<<S as BitsBitXor>::Bits>>::Reg) -> <Self as SimdReg<S>>::Reg;
}
pub trait SimdBitNot<S>: SimdReg<S> {
    type Backend: Backend;
    fn bitnot(&self,v:Self::Reg) -> Self::Reg;
}
pub trait SimdShiftWidth<S>: SimdReg<S> {
    type Backend: Backend;
    fn shift_width(&self,w:usize) -> Self::ShiftWidth;
}
pub trait SimdShl<S>: SimdReg<S> {
    type Backend: Backend;
    fn shl(&self,v:Self::Reg,w:<Self as SimdReg<S>>::ShiftWidth) -> Self::Reg;
}
pub trait SimdShr<S>: SimdReg<S> {
    type Backend: Backend;
    fn shr(&self,v:Self::Reg,w:<Self as SimdReg<S>>::ShiftWidth) -> Self::Reg;
}
pub trait SimdSplat<S>: SimdReg<S> {
    type Backend: Backend;
    fn splat(&self,v:S) -> Self::Reg;
}
pub trait SimdAddAssignVector<SL,SR> {
    type Backend: Backend;
    fn add_assign_vector<'a,const N: usize>(&self, l:&mut VectorMut<'a,SL,N>, r:&Vector<'a,SR,N,Self::Backend>);
}
pub trait SimdAddVector<SL,SR,SO> {
    type Backend: Backend;
    fn add_vector<'a,const N: usize>(&self, l:&Vector<'a,SL,N,Self::Backend>, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdSubAssignVector<SL,SR> {
    type Backend: Backend;
    fn sub_assign_vector<'a,const N: usize>(&self, l:&mut VectorMut<'a,SL,N>, r:&Vector<'a,SR,N,Self::Backend>);
}
pub trait SimdSubVector<SL,SR,SO> {
    type Backend: Backend;
    fn sub_vector<'a,const N: usize>(&self, l:&Vector<'a,SL,N,Self::Backend>, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdMulAssignVector<SL,SR> {
    type Backend: Backend;
    fn mul_assign_vector<'a,const N: usize>(&self, l:&mut VectorMut<'a,SL,N>, r:&Vector<'a,SR,N,Self::Backend>);
}
pub trait SimdMulVector<SL,SR,SO> {
    type Backend: Backend;
    fn mul_vector<'a,const N: usize>(&self, l:&Vector<'a,SL,N,Self::Backend>, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdScalarMulAssignVector<SL,SR> {
    type Backend: Backend;
    fn scalarmul_assign_vector<'a,const N: usize>(&self, l:SL, r:&mut VectorMut<'a,SR,N>);
}
pub trait SimdScalarMulVector<SL,SR,SO> {
    type Backend: Backend;
    fn scalarmul_vector<'a,const N: usize>(&self, l:SL, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdBitXorVector<S> where Self: SimdReg<S> {
    type Backend: Backend;
    fn bitxor_vector<'a,const N: usize>(&self, l:&Vector<'a,S,N,Self::Backend>, r:&Vector<'a,<Self as SimdReg<S>>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitAndVector<S>
    where Self: SimdReg<S> + SimdReg<<S as BitsBitAnd>::Bits>,
          S: BitsBitAnd {
    type Backend: Backend;
    fn bitand_vector<'a,const N: usize>(&self, l:&Vector<'a,S,N,Self::Backend>, r:&Vector<'a,<S as BitsBitAnd>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitOrVector<S> where Self: SimdReg<S> {
    type Backend: Backend;
    fn bitor_vector<'a,const N: usize>(&self, l:&Vector<'a,S,N,Self::Backend>, r:&Vector<'a,<Self as SimdReg<S>>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitNotVector<S> {
    type Backend: Backend;
    fn bitnot_vector<'a,const N: usize>(&self, v:&Vector<'a,S,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdShlVector<S> {
    type Backend: Backend;
    fn shl_vector<'a,const N: usize>(&self, v:&Vector<'a,S,N,Self::Backend>, w:usize) -> OwnedVector<S,N>;
}
pub trait SimdShrVector<S> {
    type Backend: Backend;
    fn shr_vector<'a,const N: usize>(&self, v:&Vector<'a,S,N,Self::Backend>, w:usize) -> OwnedVector<S,N>;
}
pub trait SimdAddAssignMatrix<SL,SR> {
    type Backend: Backend;
    fn add_assign_matrix<'a,const N: usize,const M: usize>(&self, l:&'a mut MatrixMut<'a,SL,N,M>, r:& Matrix<'a,SR,N,M,Self::Backend>);
}
pub trait SimdScalarMulAssignMatrix<SL,SR> {
    type Backend: Backend;
    fn scalar_mul_assign_matrix<'a,const N: usize,const M: usize>(&self, l: SL, r:&'a mut MatrixMut<'a,SR,N,M>);
}
pub trait SimdScalarMulMatrix<SL,SR,SO> {
    type Backend: Backend;
    fn scalar_mul_matrix<'a,const N: usize,const M: usize>(&self, l: SL, r:&Matrix<'a,SR,N,M,Self::Backend>) -> OwnedMatrix<SO,N,M>;
}
pub trait SimdDot<SL,SR,SO> {
    type Backend: Backend;
    fn dot<'a,const N: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,r:&Vector<'a,SR,N,Self::Backend>) -> SO;
}
pub trait SimdOuterProduct<SL,SR,SO> {
    type Backend: Backend;
    fn outer_product<'a,const N: usize,const M: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,
                                                       r:&Vector<'a,SR,M,Self::Backend>,
                                                       o:&mut OwnedMatrix<SO,N,M>);
}
pub trait SimdVMat<SL,SR,SO> {
    type Backend: Backend;
    fn vmat<'a,const M: usize,const K: usize>(&self,
                                              l:&Vector<'a,SL,K,Self::Backend>,
                                              r:&ColumnMajorMatrix<'a,SR,K,M,Self::Backend>,
                                              o:&mut OwnedVector<SO,M>);
}
pub trait SimdMatVec<SL,SR,SO> {
    type Backend: Backend;
    fn matvec<'a,const N: usize,const K: usize>(&self,
                                                l:&Matrix<'a,SL,N,K,Self::Backend>,
                                                r:&Vector<'a,SR,K,Self::Backend>,
                                                o:&mut OwnedVector<SO,N>);
}
pub trait SimdMatMul<SL,SR,SO> {
    type Backend: Backend;
    fn matmul<'a,const N: usize,const M: usize,const K: usize>(&self,
                                                               l:&Matrix<'a,SL,N,K,Self::Backend>,
                                                               r:&ColumnMajorMatrix<'a,SR,K,M,Self::Backend>,
                                                               o:&mut MatrixMut<'a,SO,N,M>);
    fn matmul_tile<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&Matrix<'a,SL,N,K,Self::Backend>,
        r:&ColumnMajorMatrix<'a,SR,K,M,Self::Backend>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );
    fn matmul_tile_tail_rows<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&Matrix<'a,SL,N,K,Self::Backend>,
        r:&ColumnMajorMatrix<'a,SR,K,M,Self::Backend>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );

    fn matmul_tile_tail_cols<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&Matrix<'a,SL,N,K,Self::Backend>,
        r:&ColumnMajorMatrix<'a,SR,K,M,Self::Backend>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );
    fn matmul_tile_tail_rows_cols<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&Matrix<'a,SL,N,K,Self::Backend>,
        r:&ColumnMajorMatrix<'a,SR,K,M,Self::Backend>,
        i:usize,
        j:usize,
        acc:&mut MatrixMut<'a,SO,N,M>
    );
}
pub trait SimdHSum<S>: SimdReg<S> {
    type Backend: Backend;
    fn hsum(&self,v:Self::Reg) -> S;
}
pub trait SimdHMax<S>: SimdReg<S> {
    type Backend: Backend;
    fn hmax(&self,v:Self::Reg) -> S;
}
pub trait SimdHMin<S>: SimdReg<S> {
    type Backend: Backend;
    fn hmin(&self,v:Self::Reg) -> S;
}
pub trait SimdTranspose<S,const ROWS: usize> where Self: SimdReg<S> {
    type Backend: Backend;
    fn transpose<'a,const N: usize,const M: usize>(v:[Self::Reg; ROWS]) -> [Self::Reg; ROWS];
}
pub trait SimdLanes<S> {
    const LANES: usize;
}
pub trait SimdRows<S> {
    const ROWS: usize;
}
pub trait SimdCols<S> {
    const COLS: usize;
}
pub trait SimdReg<S> {
    type Reg: Copy;
    type Mask: Copy;
    type Bits: Copy;
    type ShiftWidth: Copy;
}
pub trait SimdLoad<S>: SimdReg<S> {
    unsafe fn load(&self,ptr: *const S) -> Self::Reg;
}
pub trait SimdStore<S>: SimdReg<S> {
    unsafe fn store(&self, ptr: *mut S, reg: Self::Reg);
}

pub trait SimdStoreSeq<S,R>
    where R: Copy {
    unsafe fn store_seq(&self, ptr: *mut S, reg: R);
}
pub trait SimdMask<S>: SimdReg<S> {

    fn cmp_gt(&self,a:Self::Reg,b:Self::Reg) -> Self::Mask;
    fn cmp_eq(&self,a:Self::Reg,b:Self::Reg) -> Self::Mask;
    fn select(&self,mask:Self::Mask,a:Self::Reg,b:Self::Reg) -> Self::Reg;
    fn mask_zero(&self,m:Self::Mask,a:Self::Reg) -> Self::Reg;
    fn tail_mask(&self,index: usize, total: usize) -> Self::Mask;
}
pub trait SimdReinterpret<SS,SD>: SimdReg<SS> + SimdReg<SD> {
    type Backend: Backend;
    fn reinterpret(&self,reg:<Self as SimdReg<SS>>::Reg) -> <Self as SimdReg<SD>>::Reg;
}
pub trait SimdPromote<SS,SD>: SimdReg<SS> + SimdReg<SD> + SimdAdd<SD,SD,SD> + Sized {
    type Backend: Backend;
    type Output: FoldRegs<SD,Self>;
    fn promotion(&self, reg:<Self as SimdReg<SS>>::Reg) -> Self::Output;
}
pub trait SimdDemote<SS,SD>: SimdReg<SS> + SimdReg<SD> + SimdAdd<SD,SD,SD> + Sized {
    type Backend: Backend;
    type Output: FoldRegs<SD,Self>;
    fn demotion(&self, reg:<Self as SimdReg<SS>>::Reg) -> Self::Output;
}
pub trait SimdConvert<SS,SD>: SimdReg<SS> + SimdReg<SD> + SimdAdd<SD,SD,SD> + Sized {
    type Backend: Backend;
    type Output: FoldRegs<SD,Self>;
    fn convert(&self,reg:<Self as SimdReg<SS>>::Reg) -> Self::Output;
}
pub trait SimdPromoteVector<SS,SD> {
    type Backend: Backend;
    fn promotion_vector<'a,const N: usize>(&self, s:&Vector<'a,SS,N,Self::Backend>) -> OwnedVector<SD,N>;
}
pub trait SimdDemoteVector<SS,SD> {
    type Backend: Backend;
    fn demotion_vector<'a,const N: usize>(&self, s:&Vector<'a,SS,N,Self::Backend>) -> OwnedVector<SD,N>;
}
pub trait SimdConvertVector<SS,SD> {
    type Backend: Backend;
    fn convert_vector<'a,const N: usize>(&self, s:&Vector<'a,SS,N,Self::Backend>) -> OwnedVector<SD,N>;
}
pub trait SimdNeg<S>: SimdReg<S> {}
pub trait SimdMulAdd<SL,SR,SO>:
    SimdReg<SL> +
    SimdReg<SR> +
    SimdReg<SO> + SimdMul<SL,SR,SO>
    where <Self as SimdReg<SO>>::Reg: Copy {
    type Backend: Backend;
    fn zero_acc(&self) -> <Self as SimdMul<SL,SR,SO>>::Output;
    fn mul_add(&self,l:<Self as SimdReg<SL>>::Reg,
               r:<Self as SimdReg<SR>>::Reg,
               acc:<Self as SimdMul<SL,SR,SO>>::Output) -> <Self as SimdMul<SL,SR,SO>>::Output;
}
pub trait SimdPartialDot<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdAdd<SO,SO,SO> + Sized {
    type Backend: Backend;
    type Output: FoldRegs<SO,Self>;
    fn zero_acc(&self) -> Self::Output;
    fn partial_dot(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg,acc:Self::Output) -> Self::Output;
}
pub trait SimdZero<S>: SimdReg<S> {
    fn zero() -> <Self as SimdReg<S>>::Reg;
}
pub trait SupportMul<K> {}
pub trait Dot<R,O> {
    fn dot(&self,r:R) -> O;
}
pub trait Product<R,O> {
    fn product(&self, r: R) -> O;
}
pub trait HSum<S> {
    fn hsum(&self) -> S;
}
pub trait HMax<S> {
    fn hmax(&self) -> S;
}
pub trait HMin<S> {
    fn hmin(&self) -> S;
}
pub trait Transpose<T,const N: usize, const M: usize> where Self: Dims<N,M> {
    type Output: Dims<M,N>;
    fn transpose(self) -> Self::Output;
}
pub trait ToColumnMajor<T,const N: usize, const M: usize> where Self: Dims<N,M> {
    type Output: Dims<N,M>;
    fn to_column_major(self) -> Self::Output;
}
pub trait Dims<const N: usize,const M: usize> {
}
pub trait BitsBitAnd {
    type Bits;

    fn bits_bitand(self,m:Self::Bits) -> Self;
}
pub trait BitsBitOr {
    type Bits;

    fn bits_bitor(self,m:Self::Bits) -> Self;
}
pub trait BitsBitXor {
    type Bits;

    fn bits_bitxor(self,m:Self::Bits) -> Self;
}
pub trait BitsBitNot {

    fn bits_bitnot(self) -> Self;
}
pub trait BitsShl {
    fn bits_shl(self,w:usize) -> Self;
}
pub trait BitsShr {
    fn bits_shr(self,w:usize) -> Self;
}
pub trait Assume<T> {
    fn assume(self) -> T;
}
pub trait FoldRegs<S,BE: SimdReg<S> + SimdAdd<S,S,S>> where <BE as SimdReg<S>>::Reg: Copy {
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg;
}
