//! Trait and data type features for abstracting SIMD operations

use std::ops::{Index, IndexMut};
use crate::error::TryFromSliceError;

pub mod backend;
pub mod traits;
mod error;

pub struct Vector<'a,T,const N: usize> {
    data: &'a [T; N]
}
pub struct Matrix<'a,T,const N: usize,const M: usize> {
    data: &'a [T]
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
impl<'a,T,const N: usize,const M: usize> Matrix<'a,T,N,M> {
    #[inline]
    pub fn row(&self,index:usize) -> Vector<'a,T,N> {
        let view = &self.data[index * N..(index + 1) * N];

        Vector {
            data: view.try_into().unwrap()
        }
    }
}
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a [T]> for Matrix<'a,T,M,N> {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self, Self::Error> {
        if value.len() != N * M {
            Err(TryFromSliceError)
        } else {
            Ok(Matrix {
                data: value
            })
        }
    }
}
impl<'a,T,const N: usize> TryFrom<&'a [T]> for Vector<'a,T,N> {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self, Self::Error> {
        if value.len() != N {
            Err(TryFromSliceError)
        } else {
            Ok(Vector {
                data: value.try_into()?
            })
        }
    }
}
impl<T,const N: usize> Index<usize> for Vector<'_,T,N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,const N: usize,const M: usize> Index<usize> for Matrix<'_,T,N,M> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * N)..(index * N + N)]
    }
}
impl<'a,T,const N: usize,const M: usize> From<&'a OwnedMatrix<T,N,M>> for Matrix<'a,T,N,M> {
    fn from(value: &'a OwnedMatrix<T,N,M>) -> Self {
        Matrix {
            data: &value.data
        }
    }
}
impl<'a,T,const N: usize> From<&'a OwnedVector<T,N>> for Vector<'a,T,N> {
    fn from(value: &'a OwnedVector<T,N>) -> Self {
        Vector {
            data: &value.data
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