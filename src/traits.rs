//! Trait and data type features for abstracting SIMD operations

use crate::{Matrix, OwnedMatrix, OwnedVector, Vector, VectorMask};

pub trait SimdAdd<SL,SR,SO> {
    fn add<'a,const N: usize>(l:&Vector<'a,SL,N>,r:&Vector<'a,SR,N>) -> OwnedVector<SO,N>;
}
pub trait SimdSub<SL,SR,SO> {
    fn sub<'a,const N: usize>(l:&Vector<'a,SL,N>,r:&Vector<'a,SR,N>) -> OwnedVector<SO,N>;
}
pub trait SimdMul<SL,SR,SO> {
    fn mul<'a,const N: usize>(l:&Vector<'a,SL,N>,r:&Vector<'a,SR,N>) -> OwnedVector<SO,N>;
}
pub trait SimdSclarMul<SL,SR,SO> {
    fn scalarmul<'a,const N: usize>(s:SL,v:&Vector<'a,SR,N>) -> OwnedVector<SO,N>;
}
pub trait SimdDiv<SL,SR,SO> {
    fn div<'a,const N: usize>(l:&Vector<'a,SL,N>,r:&Vector<'a,SR,N>) -> OwnedVector<SO,N>;
}
pub trait SimdBitXor<S> {
    fn bitxor<'a,const N: usize>(l:&Vector<'a,S,N>,r:&Vector<'a,S,N>) -> OwnedVector<S,N>;
}
pub trait SimdBitAnd<S> {
    fn bitand<'a,const N: usize>(l:&Vector<'a,S,N>,r:&Vector<'a,S,N>) -> OwnedVector<S,N>;
}
pub trait SimdBitOr<S> {
    fn bitor<'a,const N: usize>(l:&Vector<'a,S,N>,r:&Vector<'a,S,N>) -> OwnedVector<S,N>;
}
pub trait SimdBitNot<S> {
    fn bitnot<'a,const N: usize>(v:&Vector<'a,S,N>) -> OwnedVector<S,N>;
}
pub trait SimdShiftLeft<S> {
    fn shl<'a,const N: usize>(v:&Vector<'a,S,N>,w:usize) -> OwnedVector<S,N>;
}
pub trait SimdShiftRight<S> {
    fn shr<'a,const N: usize>(v:&Vector<'a,S,N>,w:usize) -> OwnedVector<S,N>;
}
pub trait SimdDot<SL,SR,SO> {
    fn dot<'a,const N: usize>(l:&Vector<'a,SL,N>,r:&Vector<'a,SR,N>) -> SO;
}
pub trait SimdVMat<SL,SR,SO> {
    fn vmat<'a,const N: usize,const M: usize>(l:&Vector<'a,SL,N>,r:&Matrix<'a,SR,N,M>) -> OwnedMatrix<SO,N,M>;
}
pub trait SimdMatVect<SL,SR,SO> {
    fn matvect<'a,const N: usize,const M: usize>(l:&Matrix<'a,SL,N,M>,r:&Vector<'a,SR,N>) -> OwnedVector<SO,M>;
}
pub trait SimdMatMul<SL,SR,SO> {
    fn matmul<'a,const N: usize,const M: usize,const K: usize>(l:&Matrix<'a,SL,M,N>,r:&Matrix<'a,SR,N,K>) -> OwnedMatrix<SO,M,K>;
}
pub trait SimdHSum<S> {
    fn hsum<'a,const N: usize>(v:&Vector<'a,S,N>) -> S;
}
pub trait SimdMax<S> {
    fn hmax<'a,const N: usize>(v:&Vector<'a,S,N>) -> S;
}
pub trait SimdHMin<S> {
    fn hmin<'a,const N: usize>(v:&Vector<'a,S,N>) -> S;
}
pub trait SimdHOr<S> {
    fn hor<'a,const N: usize>(v:&Vector<'a,S,N>) -> S;
}
pub trait SimdHAnd<S> {
    fn hand<'a,const N: usize>(v:&Vector<'a,S,N>) -> S;
}
pub trait SimdTranspose<S> {
    fn transpose<'a,const N: usize,const M: usize>(v:&Matrix<'a,S,N,M>) -> OwnedMatrix<S,M,N>;
}
pub trait SimdLanes<S> {
    const LANES: usize;
}
pub trait SimdMask<S> {
    type Mask;
    type TailMask;

    fn cmp_gt<'a,const N: usize>(v:&Vector<'a,S,N>,w:&Vector<'a,S,N>) -> VectorMask<Self::Mask,N>;
    fn cmp_eq<'a,const N: usize>(v:&Vector<'a,S,N>,w:&Vector<'a,S,N>) -> VectorMask<Self::Mask,N>;
    fn select<'a,const N: usize>(m:&VectorMask<Self::Mask,N>,a:&Vector<'a,S,N>,b:&Vector<'a,S,N>,) -> OwnedVector<S,N>;
    fn mask_zero<'a,const N: usize>(m:&VectorMask<Self::Mask,N>,a:&Vector<'a,S,N>) -> OwnedVector<S,N>;
    fn tail_mask(index: usize, total: usize) -> Self::TailMask;
}
