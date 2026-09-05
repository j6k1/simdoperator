//! Trait and data type features for abstracting SIMD operations

use std::ops::{Add, Div, Index, Mul, Sub};
use crate::error::TryFromSliceError;
use crate::backend::common::{Backend};
use crate::traits::{SimdAdd, SimdDiv, SimdMul, SimdScalarMul, SimdSub};

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
pub struct VectorMask<M,const N: usize> {
    data: M
}
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
    fn try_from(value: &'a [T]) -> Result<Self, Self::Error> {
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
    fn try_from(value: &'a [T]) -> Result<Self, Self::Error> {
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
impl<T,const N: usize,const M: usize> From<OwnedMatrix<T,N,M>> for Box<[T]> {
    fn from(value: OwnedMatrix<T,N,M>) -> Self {
        value.data
    }
}
impl<'a,BE,T,const N: usize> Add<&'a Vector<'a,T,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdAdd<T,T,T, Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn add(self, rhs: &'a Vector<'a,T,N,BE>) -> Self::Output {
        self.backend.add(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> Sub<&'a Vector<'a,T,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdSub<T,T,T, Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn sub(self, rhs: &'a Vector<'a,T,N,BE>) -> Self::Output {
        self.backend.sub(self, rhs)
    }
}
impl<'a,BE,T,const N: usize> Div<&'a Vector<'a,T,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdDiv<T,T,T, Backend = BE> {
    type Output = OwnedVector<T,N>;

    fn div(self, rhs: &'a Vector<'a,T,N,BE>) -> Self::Output {
        self.backend.div(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i8,N,BE>> for &'a Vector<'a,i8,N,BE>
    where BE: Backend + SimdMul<i8,i8,i32, Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i8,N,BE>) -> Self::Output {
        self.backend.mul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i16,N,BE>> for &'a Vector<'a,i8,N,BE>
    where BE: Backend + SimdMul<i8,i16,i32, Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,BE>) -> Self::Output {
        self.backend.mul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i16,N,BE>> for &'a Vector<'a,i16,N,BE>
    where BE: Backend + SimdMul<i16,i16,i32, Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,BE>) -> Self::Output {
        self.backend.mul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<i8> for &'a Vector<'a,i8,N,BE>
    where BE: Backend + SimdScalarMul<i8,i8,i32, Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: i8) -> Self::Output {
        self.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<i16> for &'a Vector<'a,i8,N,BE>
    where BE: Backend + SimdScalarMul<i8,i16,i32, Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: i16) -> Self::Output {
        self.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<i16> for &'a Vector<'a,i16,N,BE>
    where BE: Backend + SimdScalarMul<i16,i16,i32, Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: i16) -> Self::Output {
        self.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<i32> for &'a Vector<'a,i32,N,BE>
    where BE: Backend + SimdScalarMul<i32,i32,i32, Backend = BE> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: i32) -> Self::Output {
        self.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<i64> for &'a Vector<'a,i64,N,BE>
    where BE: Backend + SimdScalarMul<i64,i64,i64, Backend = BE> {
    type Output = OwnedVector<i64,N>;

    fn mul(self, rhs: i64) -> Self::Output {
        self.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<f32> for &'a Vector<'a,f32,N,BE>
    where BE: Backend + SimdScalarMul<f32,f32,f32, Backend = BE> {
    type Output = OwnedVector<f32,N>;

    fn mul(self, rhs: f32) -> Self::Output {
        self.backend.scalarmul(self, rhs)
    }
}
impl<'a,BE,const N: usize> Mul<f64> for &'a Vector<'a,f64,N,BE>
    where BE: Backend + SimdScalarMul<f64,f64,f64, Backend = BE> {
    type Output = OwnedVector<f64,N>;

    fn mul(self, rhs: f64) -> Self::Output {
        self.backend.scalarmul(self, rhs)
    }
}
