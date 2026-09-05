//! Trait and data type features for abstracting SIMD operations

use crate::{Matrix, OwnedMatrix, OwnedVector, Vector, VectorMask};
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
    fn scalarmul<'a,const N: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,r:SR) -> OwnedVector<SO,N>;
}
pub trait SimdDiv<SL,SR,SO> {
    type Backend: Backend;
    fn div<'a,const N: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,N>;
}
pub trait SimdBitXor<S> {
    type Backend: Backend;
    fn bitxor<'a,const N: usize>(&self,l:&Vector<'a,S,N,Self::Backend>,r:&Vector<'a,S,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitAnd<S> {
    type Backend: Backend;
    fn bitand<'a,const N: usize>(&self,l:&Vector<'a,S,N,Self::Backend>,r:&Vector<'a,S,N,Self::Backend>) -> OwnedVector<S,N>;
}
pub trait SimdBitOr<S> {
    type Backend: Backend;
    fn bitor<'a,const N: usize>(&self,l:&Vector<'a,S,N,Self::Backend>,r:&Vector<'a,S,N,Self::Backend>) -> OwnedVector<S,N>;
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
    fn outer_product<'a,const N: usize,const M: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,r:&Vector<'a,SR,M,Self::Backend>) -> OwnedMatrix<SO,N,M>;
}
pub trait SimdVMat<SL,SR,SO> {
    type Backend: Backend;
    fn vmat<'a,const N: usize,const M: usize>(&self,l:&Vector<'a,SL,N,Self::Backend>,r:&Matrix<'a,SR,N,M,Self::Backend>) -> OwnedMatrix<SO,N,M>;
}
pub trait SimdMatVec<SL,SR,SO> {
    type Backend: Backend;
    fn matvec<'a,const N: usize,const M: usize>(&self,l:&Matrix<'a,SL,N,M,Self::Backend>,r:&Vector<'a,SR,N,Self::Backend>) -> OwnedVector<SO,M>;
}
pub trait SimdMatMul<SL,SR,SO> {
    type Backend: Backend;
    fn matmul<'a,const N: usize,const M: usize,const K: usize>(&self,l:&Matrix<'a,SL,M,N,Self::Backend>,r:&Matrix<'a,SR,N,K,Self::Backend>) -> OwnedMatrix<SO,M,K>;
}
pub trait SimdHSum<S> {
    type Backend: Backend;
    fn hsum<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>) -> S;
}
pub trait SimdHMax<S> {
    type Backend: Backend;
    fn hmax<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>) -> S;
}
pub trait SimdHMin<S> {
    type Backend: Backend;
    fn hmin<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>) -> S;
}
pub trait SimdHOr<S> {
    type Backend: Backend;
    fn hor<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>) -> S;
}
pub trait SimdHAnd<S> {
    type Backend: Backend;
    fn hand<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>) -> S;
}
pub trait SimdTranspose<S> {
    type Backend: Backend;
    fn transpose<'a,const N: usize,const M: usize>(v:&Matrix<'a,S,N,M,Self::Backend>) -> OwnedMatrix<S,M,N>;
}
pub trait SimdLanes<S> {
    const LANES: usize;
}
pub trait SimdMask<S> {
    type Backend: Backend;
    type Mask;
    type TailMask;

    fn cmp_gt<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>,w:&Vector<'a,S,N,Self::Backend>) -> VectorMask<Self::Mask,N>;
    fn cmp_eq<'a,const N: usize>(&self,v:&Vector<'a,S,N,Self::Backend>,w:&Vector<'a,S,N,Self::Backend>) -> VectorMask<Self::Mask,N>;
    fn select<'a,const N: usize>(&self,m:&VectorMask<Self::Mask,N>,a:&Vector<'a,S,N,Self::Backend>,b:&Vector<'a,S,N,Self::Backend>,) -> OwnedVector<S,N>;
    fn mask_zero<'a,const N: usize>(&self,m:&VectorMask<Self::Mask,N>,a:&Vector<'a,S,N,Self::Backend>) -> OwnedVector<S,N>;
    fn tail_mask(index: usize, total: usize) -> Self::TailMask;
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
pub trait HOr<S> {
    fn hor(&self) -> S;
}
pub trait HAnd<S> {
    fn hand(&self) -> S;
}
pub trait Transpose<T,const N: usize, const M: usize> where Self: Dims<N,M> {
    type Output: Dims<M,N>;
    fn transpose(v:Self) -> Self::Output;
}
pub trait Dims<const N: usize,const M: usize> {
}