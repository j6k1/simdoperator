//! Trait and data type features for abstracting SIMD operations

use std::ops::{Add, BitAnd, BitOr, BitXor, Index, IndexMut, Mul, Not, Shl, Shr, Sub};
use crate::error::TryFromSliceError;
use crate::backend::common::{Backend};
use crate::traits::{Dims, Dot, HAnd, HMax, HMin, HOr, HSum, Product, SimdAdd, SimdBitAnd, SimdBitNot, SimdBitOr, SimdBitXor, SimdDot, SimdHAnd, SimdHMax, SimdHMin, SimdHOr, SimdHSum, SimdMatMul, SimdMatVec, SimdMul, SimdOuterProduct, SimdReg, SimdScalarMul, SimdShiftLeft, SimdShiftRight, SimdSub, SimdVMat};

pub mod backend;
pub mod traits;
pub mod error;

pub struct Vector<'a,T,const N: usize,BE: Backend> {
    data: &'a [T; N],
    backend: BE
}
pub struct Matrix<'a,T,const N: usize,const M: usize,BE: Backend> {
    data: &'a [T],
    backend: BE
}
pub struct OwnedVector<T,const N: usize> {
    data: [T; N]
}
pub struct OwnedMatrix<T,const N: usize,const M: usize> {
    data: Box<[T]>
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> Dims<N,M> for Matrix<'a,T,N,M,BE> {}
impl<T,const N: usize,const M:usize> Dims<N,M> for OwnedMatrix<T,N,M> {}
impl<'a,BE: Backend,T,const N: usize,const M: usize> Matrix<'a,T,N,M,BE> {
    #[inline]
    pub fn row(&self,index:usize) -> Vector<'a,T,N,BE> {
        let view = &self.data[index * N..(index + 1) * N];

        Vector {
            data: view.try_into().unwrap(),
            backend: BE::new()
        }
    }
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> TryFrom<&'a [T]> for Matrix<'a,T,M,N,BE> {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(TryFromSliceError)
        } else {
            Ok(Matrix {
                data: value,
                backend: BE::new()
            })
        }
    }
}
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a [T]> for Vector<'a,T,N,BE> {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(TryFromSliceError)
        } else {
            Ok(Vector {
                data: value.try_into()?,
                backend: BE::new()
            })
        }
    }
}
impl<T,BE: Backend,const N: usize> Index<usize> for Vector<'_,T,N,BE> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,BE: Backend,const N: usize> AsRef<[T;N]> for Vector<'_,T,N,BE> {
    fn as_ref(&self) -> &[T;N] {
        &self.data
    }
}
impl<T,BE: Backend,const N: usize,const M: usize> Index<usize> for Matrix<'_,T,N,M,BE> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * N)..(index * N + N)]
    }
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> From<&'a OwnedMatrix<T,N,M>> for Matrix<'a,T,N,M,BE> {
    fn from(value: &'a OwnedMatrix<T,N,M>) -> Self {
        Matrix {
            data: &value.data,
            backend: BE::new()
        }
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a OwnedVector<T,N>> for Vector<'a,T,N,BE> {
    fn from(value: &'a OwnedVector<T,N>) -> Self {
        Vector {
            data: &value.data,
            backend: BE::new()
        }
    }
}
impl<T,const N: usize> From<OwnedVector<T,N>> for [T;N] {
    fn from(value: OwnedVector<T,N>) -> Self {
        value.data
    }
}
impl<T,const N: usize> From<[T;N]> for OwnedVector<T,N> {
    fn from(value: [T;N]) -> Self {
        OwnedVector {
            data: value
        }
    }
}
impl<T,const N: usize> AsMut<[T;N]> for OwnedVector<T,N> {
    fn as_mut(&mut self) -> &mut [T;N] {
        &mut self.data
    }
}
impl<T,const N: usize,const M: usize> AsMut<[T]> for OwnedMatrix<T,N,M> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
impl<T,const N: usize> Index<usize> for OwnedVector<T,N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,const N: usize> IndexMut<usize> for OwnedVector<T,N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}
impl<T,const N: usize,const M: usize> From<OwnedMatrix<T,N,M>> for Box<[T]> {
    fn from(value: OwnedMatrix<T,N,M>) -> Self {
        value.data
    }
}
impl<'a,BE,T,const N: usize> Add<&'a Vector<'a,T,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdAdd<T,T,T,Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn add(self, rhs: &'a Vector<'a,T,N,BE>) -> Self::Output {
        self.backend.add(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> Sub<&'a Vector<'a,T,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdSub<T,T,T,Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn sub(self, rhs: &'a Vector<'a,T,N,BE>) -> Self::Output {
        self.backend.sub(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i8,N,BE>> for &'a Vector<'a,i8,N,BE>
    where BE: Backend + SimdMul<i8,i8,i32,Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i8,N,BE>) -> Self::Output {
        self.backend.mul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i16,N,BE>> for &'a Vector<'a,i8,N,BE>
    where BE: Backend + SimdMul<i8,i16,i32,Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,BE>) -> Self::Output {
        self.backend.mul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i16,N,BE>> for &'a Vector<'a,i16,N,BE>
    where BE: Backend + SimdMul<i16,i16,i32,Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,BE>) -> Self::Output {
        self.backend.mul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i8,N,BE>> for i8
    where BE: Backend + SimdScalarMul<i8,i8,i32,Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i8,N,BE>) -> Self::Output {
        rhs.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i16,N,BE>> for i8
    where BE: Backend + SimdScalarMul<i8,i16,i32,Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,BE>) -> Self::Output {
        rhs.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i16,N,BE>> for i16
    where BE: Backend + SimdScalarMul<i16,i16,i32,Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,BE>) -> Self::Output {
        rhs.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i32,N,BE>> for i32
    where BE: Backend + SimdScalarMul<i32,i32,i32,Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i32,N,BE>) -> Self::Output {
        rhs.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,f32,N,BE>> for f32
    where BE: Backend + SimdScalarMul<f32,f32,f32,Backend = BE> {
    type Output = OwnedVector<f32,N>;

    fn mul(self, rhs: &'a Vector<'a,f32,N,BE>) -> Self::Output {
        rhs.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,f64,N,BE>> for f64
    where BE: Backend + SimdScalarMul<f64,f64,f64,Backend = BE> {
    type Output = OwnedVector<f64,N>;

    fn mul(self, rhs: &'a Vector<'a,f64,N,BE>) -> Self::Output {
        rhs.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> BitXor<&'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdReg<T> + SimdBitXor<T,Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn bitxor(self, rhs: &'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>) -> Self::Output {
        self.backend.bitxor(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> BitOr<&'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdReg<T> + SimdBitOr<T,Backend = BE> {

    type Output = OwnedVector<T,N>;

    fn bitor(self, rhs: &'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>) -> Self::Output {
        self.backend.bitor(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> BitAnd<&'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdReg<T> + SimdBitAnd<T,Backend = BE> {

    type Output = OwnedVector<T,N>;

    fn bitand(self, rhs: &'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>) -> Self::Output {
        self.backend.bitand(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> Not for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdBitNot<T,Backend = BE> {

    type Output = OwnedVector<T,N>;

    fn not(self) -> Self::Output {
        self.backend.bitnot(self)
    }
}
impl<'a,BE,T,const N: usize> Shl<usize> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdShiftLeft<T,Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn shl(self, rhs: usize) -> Self::Output {
        self.backend.shl(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> Shr<usize> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdShiftRight<T,Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn shr(self, rhs: usize) -> Self::Output {
        self.backend.shr(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> HSum<T> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdHSum<T,Backend = BE> {
    fn hsum(&self) -> T {
        self.backend.hsum(self)
    }
}
impl<'a,BE,T,const N: usize> HAnd<T> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdHAnd<T,Backend = BE> {
    fn hand(&self) -> T {
        self.backend.hand(self)
    }
}
impl<'a,BE,T,const N: usize> HOr<T> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdHOr<T,Backend = BE> {
    fn hor(&self) -> T {
        self.backend.hor(self)
    }
}
impl<'a,BE,T,const N: usize> HMax<T> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdHMax<T,Backend = BE> {
    fn hmax(&self) -> T {
        self.backend.hmax(self)
    }
}
impl<'a,BE,T,const N: usize> HMin<T> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdHMin<T,Backend = BE> {
    fn hmin(&self) -> T {
        self.backend.hmin(self)
    }
}
impl<'a,BE,SL,SR,SO,const N: usize> Dot<&'a Vector<'a,SR,N,BE>,SO> for &'a Vector<'a,SL,N,BE>
    where BE: Backend + SimdDot<SL,SR,SO,Backend = BE> {
    fn dot(&self,r:&'a Vector<'a,SR,N,BE>) -> SO {
        self.backend.dot(self,r)
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize> Product<&'a Vector<'a,SR,M,BE>,OwnedMatrix<SO,N,M>>
    for &'a Vector<'a,SL,N,BE> where BE: Backend + SimdOuterProduct<SL,SR,SO,Backend = BE> {
    fn product(&self,r:&'a Vector<'a,SR,M,BE>) -> OwnedMatrix<SO,N,M> {
        self.backend.outer_product(self,r)
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize> Product<&'a Matrix<'a,SR,N,M,BE>,OwnedMatrix<SO,N,M>>
    for &'a Vector<'a,SL,N,BE> where BE: Backend + SimdVMat<SL,SR,SO,Backend = BE> {
    fn product(&self,r:&'a Matrix<'a,SR,N,M,BE>) -> OwnedMatrix<SO,N,M> {
        self.backend.vmat(self,r)
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize> Product<&'a Vector<'a,SR,N,BE>,OwnedVector<SO,M>>
    for &'a Matrix<'a,SL,N,M,BE> where BE: Backend + SimdMatVec<SL,SR,SO,Backend = BE> {
    fn product(&self, r: &'a Vector<'a,SR,N,BE>) -> OwnedVector<SO, M> {
        self.backend.matvec(self, r)
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize,const K: usize> Product<&'a Matrix<'a,SR,N,K,BE>,OwnedMatrix<SO,M,K>>
    for &'a Matrix<'a,SL,M,N,BE> where BE: Backend + SimdMatMul<SL,SR,SO,Backend = BE> {
    fn product(&self, r: &'a Matrix<'a,SR,N,K,BE>) -> OwnedMatrix<SO,M,K> {
        self.backend.matmul(self, r)
    }
}