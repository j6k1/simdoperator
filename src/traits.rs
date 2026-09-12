//! Trait and data type features for abstracting SIMD operations

use crate::{ColumnMajorMatrix, Matrix, MatrixMut, OwnedMatrix, OwnedVector, Vector};
use crate::backend::common::Backend;

pub trait SimdAdd<SL,SR,SO> {
    type Backend: Backend;
    fn add<'a,const N: usize>(&self, l:&Vector<'a,SL,N,Self::Backend>, r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdSub<SL,SR,SO> {
    type Backend: Backend;
    fn sub<'a,const N: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdMul<SL,SR,SO> {
    type Backend: Backend;
    fn mul<'a,const N: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdScalarMul<SL,SR,SO> {
    type Backend: Backend;
    fn scalarmul<'a,const N: usize>(&self,l:SL,r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdBitXor<S> where Self: SimdReg<S> {
    type Backend: Backend;
    fn bitxor<'a,const N: usize>(&self,l:&Vector<'a,S,N,Self::Backend>,r:&Vector<'a,<Self as SimdReg<S>>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitAnd<S> where Self: SimdReg<S> {
    type Backend: Backend;
    fn bitand<'a,const N: usize>(&self,l:&Vector<'a,S,N,Self::Backend>,r:&Vector<'a,<Self as SimdReg<S>>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitOr<S> where Self: SimdReg<S> {
    type Backend: Backend;
    fn bitor<'a,const N: usize>(&self,l:&Vector<'a,S,N,Self::Backend>,r:&Vector<'a,<Self as SimdReg<S>>::Bits,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitNot<S> {
    type Backend: Backend;
    fn bitnot<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdShiftLeft<S> {
    type Backend: Backend;
    fn shl<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>,w:usize) -> OwnedVector<S,N>;
}
pub trait SimdShiftRight<S> {
    type Backend: Backend;
    fn shr<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>,w:usize) -> OwnedVector<S,N>;
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
    const ROWS_GEMV: usize;
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
pub trait SimdConvert<SS,SD>: SimdReg<SS> + SimdReg<SD> {
    type Backend: Backend;
    fn convert(&self,reg:<Self as SimdReg<SS>>::Reg) -> <Self as SimdReg<SD>>::Reg;
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