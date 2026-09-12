//! Trait and data type features for abstracting SIMD operations

use crate::{ColumnMajorMatrix, Matrix, MatrixMut, OwnedMatrix, OwnedVector, Vector};
use crate::backend::common::Backend;
pub trait SimdAdd<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    type Backend: Backend;

    fn add(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
pub trait SimdSub<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    type Backend: Backend;
    fn sub(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
pub trait SimdMul<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    type Backend: Backend;
    type Output;
    fn mul(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg) -> Self::Output;
}
pub trait SimdScalarMul<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdMul<SL,SR,SO> {
    type Backend: Backend;
    fn scalarmul(&self,l:SL,r:<Self as SimdReg<SR>>::Reg) -> <Self as SimdMul<SL,SR,SO>>::Output;
}
pub trait SimdBitOr<S>: SimdReg<S> {
    type Backend: Backend;
    fn bitor(&self,v:Self::Reg) -> Self::Reg;
}
pub  trait SimdBitAnd<S>: SimdReg<S> {
    type Backend: Backend;
    fn bitand(&self,v:Self::Reg) -> Self::Reg;
}
pub trait SimdBitXor<S>: SimdReg<S> {
    type Backend: Backend;
    fn bitxor(&self,v:Self::Reg) -> Self::Reg;
}
pub trait SimdBitNot<S>: SimdReg<S> {
    type Backend: Backend;
    fn bitnot(&self,v:Self::Reg) -> Self::Reg;
}
pub trait SimdShl<S>: SimdReg<S> {
    type Backend: Backend;
    fn shl(&self,v:Self::Reg) -> Self::Reg;
}
pub trait SimdShr<S>: SimdReg<S> {
    type Backend: Backend;
    fn shr(&self,v:Self::Reg) -> Self::Reg;
}
pub trait SimdSplat<S>: SimdReg<S> {
    type Backend: Backend;
    fn splat(&self,v:S) -> Self::Reg;
}
pub trait SimdSplatVector<S,const N: usize>: SimdReg<S> {}
pub trait SimdAddVector<SL,SR,SO> {
    type Backend: Backend;
    fn add_vector<'a,const N: usize>(&self, l:&Vector<'a,SL,N,Self::Backend>, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdSubVector<SL,SR,SO> {
    type Backend: Backend;
    fn sub_vector<'a,const N: usize>(&self, l:&Vector<'a,SL,N,Self::Backend>, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdMulVector<SL,SR,SO> {
    type Backend: Backend;
    fn mul_vector<'a,const N: usize>(&self, l:&Vector<'a,SL,N,Self::Backend>, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdScalarMulVector<SL,SR,SO> {
    type Backend: Backend;
    fn scalarmul_vector<'a,const N: usize>(&self, l:SL, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdBitXorVector<S> where Self: SimdReg<S> {
    type Backend: Backend;
    fn bitxor_vector<'a,const N: usize>(&self, l:&Vector<'a,S,N,Self::Backend>, r:&Vector<'a,<Self as SimdReg<S>>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitAndVector<S> where Self: SimdReg<S> {
    type Backend: Backend;
    fn bitand_vector<'a,const N: usize>(&self, l:&Vector<'a,S,N,Self::Backend>, r:&Vector<'a,<Self as SimdReg<S>>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
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
    type Reg;
    type Mask;
    type Bits;
}
pub trait SimdLoad<S>: SimdReg<S> {
    unsafe fn load(&self,ptr: *const S) -> Self::Reg;
}
pub trait SimdStore<S>: SimdReg<S> {
    unsafe fn store(&self, ptr: *mut S, reg: Self::Reg);
}
pub trait SimdStoreSeq<S,I> {
    unsafe fn store_seq(&self, ptr: *mut S, reg: I);
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
pub trait SimdPromote<SS,SD>: SimdReg<SS> + SimdReg<SD> {
    type Backend: Backend;
    type Output;
    fn promotion(&self, reg:<Self as SimdReg<SS>>::Reg) -> Self::Output;
}
pub trait SimdMulAdd<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    type Backend: Backend;
    fn mul_add(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg,acc:<Self as SimdReg<SO>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
pub trait SimdPartialDot<SL,SR,SO>: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> {
    type Backend: Backend;
    fn partial_dot(&self,l:<Self as SimdReg<SL>>::Reg,r:<Self as SimdReg<SR>>::Reg,acc:<Self as SimdReg<SO>>::Reg) -> <Self as SimdReg<SO>>::Reg;
}
pub trait SimdZero<S>: SimdReg<S> {
    fn zero() -> <Self as SimdReg<S>>::Reg;
}
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