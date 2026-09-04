//! Trait and data type features for abstracting SIMD operations

pub struct Vector<T,const N: usize> {
    data: [T; N]
}
pub struct Matrix<T,const N: usize,const M: usize> {
    data: [[T; N]; M]
}
pub struct VectorMask<M,const N: usize> {
    data: M
}
pub trait SimdAdd<SL,SR,SO> {
    fn add<const N: usize>(l:&Vector<SL,N>,r:&Vector<SR,N>) -> Vector<SO,N>;
}
pub trait SimdSub<SL,SR,SO> {
    fn sub<const N: usize>(l:&Vector<SL,N>,r:&Vector<SR,N>) -> Vector<SO,N>;
}
pub trait SimdMul<SL,SR,SO> {
    fn mul<const N: usize>(l:&Vector<SL,N>,r:&Vector<SR,N>) -> Vector<SO,N>;
}
pub trait SimdDiv<SL,SR,SO> {
    fn div<const N: usize>(l:&Vector<SL,N>,r:&Vector<SR,N>) -> Vector<SO,N>;
}
pub trait SimdBitXor<S> {
    fn bitxor<const N: usize>(l:&Vector<S,N>,r:&Vector<S,N>) -> Vector<S,N>;
}
pub trait SimdBitAnd<S> {
    fn bitand<const N: usize>(l:&Vector<S,N>,r:&Vector<S,N>) -> Vector<S,N>;
}
pub trait SimdBitOr<S> {
    fn bitor<const N: usize>(l:&Vector<S,N>,r:&Vector<S,N>) -> Vector<S,N>;
}
pub trait SimdBitNot<S> {
    fn bitnot<const N: usize>(v:&Vector<S,N>) -> Vector<S,N>;
}
pub trait SimdShiftLeft<S> {
    fn shl<const N: usize>(v:&Vector<S,N>,w:usize) -> Vector<S,N>;
}
pub trait SimdShiftRight<S> {
    fn shr<const N: usize>(v:&Vector<S,N>,w:usize) -> Vector<S,N>;
}
pub trait SimdDot<SL,SR,SO> {
    fn dot<const N: usize>(l:&Vector<SL,N>,r:&Vector<SR,N>) -> SO;
}
pub trait SimdVMat<SL,SR,SO> {
    fn vmat<const N: usize,const M: usize>(l:&Vector<SL,N>,r:&Matrix<SR,N,M>) -> Matrix<SO,N,M>;
}
pub trait SimdMatVect<SL,SR,SO> {
    fn matvect<const N: usize,const M: usize>(l:&Matrix<SL,N,M>,r:&Vector<SR,N>) -> Vector<SO,M>;
}
pub trait SimdMatMul<SL,SR,SO> {
    fn matmul<const N: usize,const M: usize,const K: usize>(l:&Matrix<SL,M,N>,r:&Matrix<SR,N,K>) -> Matrix<SO,M,K>;
}
pub trait SimdHSum<S> {
    fn hsum<const N: usize>(v:&Vector<S,N>) -> S;
}
pub trait SimdMax<S> {
    fn hmax<const N: usize>(v:&Vector<S,N>) -> S;
}
pub trait SimdHMin<S> {
    fn hmin<const N: usize>(v:&Vector<S,N>) -> S;
}
pub trait SimdHOr<S> {
    fn hor<const N: usize>(v:&Vector<S,N>) -> S;
}
pub trait SimdHAnd<S> {
    fn hand<const N: usize>(v:&Vector<S,N>) -> S;
}
pub trait SimdSplat<S> {
    fn splat<const N: usize>(v:S) -> Vector<S,N>;
}
pub trait SimdTranspose<S> {
    fn transposee<const N: usize,const M: usize>(v:&Matrix<S,N,M>) -> Matrix<S,M,N>;
}
pub trait SimdLanes<S> {
    const LANES: usize;
}
pub trait SimdMask<S> {
    type Mask;
    type TailMask;

    fn cmp_gt<const N: usize>(v:&Vector<S,N>,w:&Vector<S,N>) -> VectorMask<Self::Mask,N>;
    fn cmp_eq<const N: usize>(v:&Vector<S,N>,w:&Vector<S,N>) -> VectorMask<Self::Mask,N>;
    fn select<const N: usize>(m:&VectorMask<Self::Mask,N>,a:&Vector<S,N>,b:&Vector<S,N>,) -> Vector<S,N>;
    fn mask_zero<const N: usize>(m:&VectorMask<Self::Mask,N>,a:&Vector<S,N>) -> Vector<S,N>;
    fn tail_mas(index: usize, total: usize) -> Self::TailMask;
}
