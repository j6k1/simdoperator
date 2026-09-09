//! Trait and data type features for abstracting SIMD operations

use std::ops::{Add, BitAnd, BitOr, BitXor, Index, IndexMut, Mul, Not, Shl, Shr, Sub};
use crate::error::TryFromSliceError;
use crate::backend::common::{Backend};
use crate::traits::{Dims, Dot, HMax, HMin, HSum, Product, SimdAdd, SimdBitAnd, SimdBitNot, SimdBitOr, SimdBitXor, SimdDot, SimdHMax, SimdHMin, SimdHSum, SimdMatMul, SimdMatVec, SimdMul, SimdOuterProduct, SimdReg, SimdScalarMul, SimdShiftLeft, SimdShiftRight, SimdSub, SimdVMat, ToColumnMajor, Transpose};

pub mod backend;
pub mod traits;
pub mod error;

pub struct Vector<'a,T,const N: usize,BE: Backend> {
    data: &'a [T; N],
    backend: BE
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
impl<'a,BE: Backend,T,const N: usize> From<&'a OwnedVector<T,N>> for Vector<'a,T,N,BE> {
    fn from(value: &'a OwnedVector<T,N>) -> Self {
        Vector {
            data: &value.data,
            backend: BE::new()
        }
    }
}
pub struct OwnedVector<T,const N: usize> {
    data: [T; N]
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
impl<T,const N: usize> Default for OwnedVector<T,N> where T: Default + Clone + Copy {
    fn default() -> Self {
        Self {
            data: [T::default();N].into()
        }
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
pub struct Matrix<'a,T,const N: usize,const M: usize,BE: Backend> {
    data: &'a [T],
    backend: BE
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> Dims<N,M> for Matrix<'a,T,N,M,BE> {}
impl<'a,BE: Backend,T,const N: usize,const M: usize> Matrix<'a,T,N,M,BE> {
    #[inline]
    pub fn row(&self,index:usize) -> Vector<'a,T,N,BE> {
        let view = &self.data[index * M..(index + 1) * M];

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
impl<T,BE: Backend,const N: usize,const M: usize> Index<usize> for Matrix<'_,T,N,M,BE> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * M)..(index * M + M)]
    }
}
impl<'a,T,BE: Backend,const N: usize,const M: usize> Transpose<T,N,M> for Matrix<'a,T,N,M,BE>
    where T: Default + Clone + Copy {
    type Output = OwnedMatrix<T,M,N>;
    fn transpose(self) -> OwnedMatrix<T,M,N> {
        let mut r = OwnedMatrix::<T,M,N>::default();

        const BLOCK:usize = 64;

        for row in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
            for col in (0..((M + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
                for x in 0..BLOCK {
                    for y in 0..BLOCK {
                        if row + y >= N || col + x >= M {
                            continue;
                        }
                        r[(col + y,row + x)] = self.data[(row + x) * M + col + y];
                    }
                }
            }
        }

        r
    }
}
impl<'a,T,BE: Backend,const N: usize,const M: usize> ToColumnMajor<T,N,M> for Matrix<'a,T,N,M,BE>
    where T: Default + Clone + Copy {
    type Output = OwnedColumnMajorMatrix<T,N,M>;
    fn to_column_major(self) -> OwnedColumnMajorMatrix<T,N,M> {
        let mut r = vec![T::default();N * M].into_boxed_slice();

        const BLOCK:usize = 64;

        for row in (0..((N + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
            for col in (0..((M + BLOCK - 1) / BLOCK * BLOCK)).step_by(BLOCK) {
                for x in 0..BLOCK {
                    for y in 0..BLOCK {
                        if row + y >= N || col + x >= M {
                            continue;
                        }
                        r[(col + y) * N + (row + x)] = self.data[(row + x) * M + col + y];
                    }
                }
            }
        }

        OwnedColumnMajorMatrix { data: r }
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
pub struct OwnedMatrix<T,const N: usize,const M: usize> {
    data: Box<[T]>
}
impl<T,const N: usize,const M: usize> Default for OwnedMatrix<T,N,M> where T: Default + Clone {
    fn default() -> Self {
        Self {
            data: vec![T::default();N*M].into_boxed_slice()
        }
    }
}
impl<T,const N: usize,const M: usize> Index<(usize,usize)> for OwnedMatrix<T,N,M> {
    type Output = T;

    fn index(&self, (row,col): (usize, usize)) -> &Self::Output {
        &self.data[(row * M) + col]
    }
}
impl<T,const N: usize,const M: usize> IndexMut<(usize,usize)> for OwnedMatrix<T,N,M> {
    fn index_mut(&mut self, (row,col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[(row * M) + col]
    }
}
impl<T,const N: usize,const M: usize> From<OwnedMatrix<T,N,M>> for Box<[T]> {
    fn from(value: OwnedMatrix<T,N,M>) -> Self {
        value.data
    }
}
impl<T,const N: usize,const M: usize> From<Box<[T]>> for OwnedMatrix<T,N,M> {
    fn from(value: Box<[T]>) -> Self {
        OwnedMatrix { data: value }
    }
}
impl<T,const N: usize,const M: usize> AsMut<[T]> for OwnedMatrix<T,N,M> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}
impl<T,const N: usize,const M:usize> Dims<N,M> for OwnedMatrix<T,N,M> {}
impl<'a,BE: Backend,T,const N: usize,const M: usize> ColumnMajorMatrix<'a,T,N,M,BE> {
    #[inline]
    pub fn col(&self,index:usize) -> Vector<'a,T,N,BE> {
        let view = &self.data[index * N..(index + 1) * N];

        Vector {
            data: view.try_into().unwrap(),
            backend: BE::new()
        }
    }
}
pub struct ColumnMajorMatrix<'a,T,const N: usize,const M: usize,BE: Backend> {
    data: &'a [T],
    backend: BE
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> Dims<N,M> for ColumnMajorMatrix<'a,T,N,M,BE> {}
impl<'a,BE: Backend,T,const N: usize,const M: usize> TryFrom<&'a [T]> for ColumnMajorMatrix<'a,T,M,N,BE> {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(TryFromSliceError)
        } else {
            Ok(ColumnMajorMatrix {
                data: value,
                backend: BE::new()
            })
        }
    }
}
pub struct OwnedColumnMajorMatrix<T,const M: usize,const N: usize> {
    data: Box<[T]>
}
impl<T,const N: usize,const M: usize> From<OwnedColumnMajorMatrix<T,N,M>> for Box<[T]> {
    fn from(value: OwnedColumnMajorMatrix<T,N,M>) -> Self {
        value.data
    }
}
impl<T,const N: usize,const M: usize> From<Box<[T]>> for OwnedColumnMajorMatrix<T,M,N> {
    fn from(value: Box<[T]>) -> Self {
        OwnedColumnMajorMatrix { data: value }
    }
}
impl<T,const N: usize,const M: usize> Dims<N,M> for OwnedColumnMajorMatrix<T,N,M> {}
impl<'a,BE: Backend,T,const N: usize,const M: usize> From<&'a OwnedColumnMajorMatrix<T,N,M>> for ColumnMajorMatrix<'a,T,N,M,BE> {
    fn from(value: &'a OwnedColumnMajorMatrix<T,N,M>) -> Self {
        ColumnMajorMatrix {
            data: &value.data,
            backend: BE::new()
        }
    }
}
impl<'a,BE: Backend,T,const N: usize,const M: usize> From<&'a ColumnMajorMatrix<'a,T,N,M,BE>>
    for OwnedColumnMajorMatrix<T,M,N>
    where T: Clone + Copy {
    fn from(value: &'a ColumnMajorMatrix<'a,T,N,M,BE>) -> Self {
        OwnedColumnMajorMatrix {
            data: value.data.to_vec().into_boxed_slice()
        }
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
impl<'a,BE,SL,SR,SO,const M: usize,const K: usize> Product<&'a ColumnMajorMatrix<'a,SR,M,K,BE>,OwnedVector<SO,M>>
    for &'a Vector<'a,SL,K,BE>
    where BE: Backend + SimdVMat<SL,SR,SO,Backend = BE>,
          SO: Default + Copy + Clone {
    fn product(&self,r:&'a ColumnMajorMatrix<'a,SR,M,K,BE>) -> OwnedVector<SO,M> {
        let mut o = OwnedVector::<SO,M>::default();

        self.backend.vmat(self,r,&mut o);

        o
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const K: usize> Product<&'a Vector<'a,SR,K,BE>,OwnedVector<SO,N>>
    for &'a Matrix<'a,SL,N,K,BE>
        where BE: Backend + SimdMatVec<SL,SR,SO,Backend = BE>,
              SO: Default + Copy + Clone {
    fn product(&self, r: &'a Vector<'a,SR,K,BE>) -> OwnedVector<SO,N> {
        let mut o = OwnedVector::<SO,N>::default();

        self.backend.matvec(self, r, &mut o);

        o
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize,const K: usize> Product<&'a ColumnMajorMatrix<'a,SR,K,M,BE>,OwnedMatrix<SO,N,M>>
    for &'a Matrix<'a,SL,N,K,BE>
    where BE: Backend + SimdMatMul<SL,SR,SO,Backend = BE>,
          SO: Default + Copy + Clone {
    fn product(&self, r: &'a ColumnMajorMatrix<'a,SR,K,M,BE>) -> OwnedMatrix<SO,N,M> {
        let mut o = OwnedMatrix::<SO,N,M>::default();
        self.backend.matmul(self, r, &mut o);

        o
    }
}