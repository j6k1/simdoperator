//! Trait and data type features for abstracting SIMD operations

use std::ops::{Add, AddAssign, BitAnd, BitOr, BitXor, Index, IndexMut, Mul, MulAssign, Not, Shl, Shr, Sub, SubAssign};
use crate::backend::autoselect::{AutoSelect, SelectedBackend};
use crate::backend::avx2::Avx2;
use crate::error::{InstantiationError, TryFromSliceError};
use crate::backend::common::{Backend};
use crate::traits::{BitsBitAnd, BitsBitOr, BitsBitXor, Dims, Dot, Product, SimdAddAssignVector, SimdAddVector, SimdBitAndVector, SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdConvertMatrix, SimdConvertVector, SimdDemoteVector, SimdDot, SimdMask, SimdMatMul, SimdMatVec, SimdMulAssignVector, SimdMulVector, SimdOuterProduct, SimdPromoteVector, SimdReg, SimdScalarMulAssignVector, SimdScalarMulVector, SimdShlVector, SimdShrVector, SimdStore, SimdSubAssignVector, SimdSubVector, SimdVMat, ToColumnMajor, Transpose};

pub mod backend;
pub mod traits;
pub mod error;
#[macro_use]
pub mod macros;

pub struct Scalar<T,BE = AutoSelect>
    where T: Copy,
          BE: Backend {
    pub value:T,
    pub backend:BE
}
impl<T,BE> Scalar<T,BE>
    where T: Copy,
          BE: Backend {
    pub fn new(value:T) -> Result<Self,InstantiationError> {
        Ok(Scalar {
            value:value,
            backend:BE::new()?
        })
    }
}
pub struct Vector<'a,T,const N: usize,BE: Backend> {
    data: &'a [T; N],
    backend: BE
}
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a [T]> for Vector<'a,T,N,BE> {
    type Error = InstantiationError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(Vector {
                data: value.try_into()?,
                backend: BE::new()?
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
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a OwnedVector<T,N>> for Vector<'a,T,N,BE> {
    type Error = InstantiationError;
    fn try_from(value: &'a OwnedVector<T,N>) -> Result<Self,Self::Error> {
        Ok(Vector {
            data: &value.data,
            backend: BE::new()?
        })
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a Vector<'a,T,N,BE>> for Box<[T;N]>
    where T: Clone + Copy {

    fn from(value: &'a Vector<'a,T,N,BE>) -> Self {
        Box::new(value.data.clone().into())
    }
}
impl<'a,BE: Backend,T,const N: usize> Vector<'a,T,N,BE> {
    pub fn as_vertical(&self) -> Matrix<'a,T,N,1,BE> {
        Matrix {
            data: self.data,
            backend: BE::new().unwrap()
        }
    }

    pub fn as_horizontal(&self) -> Matrix<'a,T,1,N,BE> {
        Matrix {
            data: self.data,
            backend: BE::new().unwrap()
        }
    }
}
pub struct VectorView<'a,T,const N: usize> {
    data: &'a [T; N]
}
impl<'a,T,const N: usize> TryFrom<&'a [T]> for VectorView<'a,T,N> {
    type Error = InstantiationError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(VectorView {
                data: value.try_into()?
            })
        }
    }
}
impl<T,const N: usize> Index<usize> for VectorView<'_,T,N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<T,const N: usize> AsRef<[T;N]> for VectorView<'_,T,N> {
    fn as_ref(&self) -> &[T;N] {
        &self.data
    }
}
impl<'a,T,const N: usize> From<&'a OwnedVector<T,N>> for VectorView<'a,T,N> {
    fn from(value: &'a OwnedVector<T,N>) -> VectorView<'a,T,N> {
        VectorView {
            data: &value.data
        }
    }
}
impl<'a,T,const N: usize> From<&'a VectorView<'a,T,N>> for Box<[T;N]>
    where T: Clone + Copy {
    fn from(value: &'a VectorView<'a,T,N>) -> Box<[T;N]> {
        value.data.clone().into()
    }
}
impl<'a,T,const N: usize> VectorView<'a,T,N> {
    pub fn as_vertical(&self) -> MatrixView<'a,T,N,1> {
        MatrixView {
            data: self.data
        }
    }

    pub fn as_horizontal(&self) -> MatrixView<'a,T,1,N> {
        MatrixView {
            data: self.data
        }
    }
}
pub struct VectorViewMut<'a,T,const N: usize> {
    data: &'a mut [T;N]
}
impl<'a,T,BE: Backend,const N: usize> From<&'a mut VectorMut<'a,T,N,BE>> for VectorViewMut<'a,T,N> {
    fn from(value: &'a mut VectorMut<'a,T,N,BE>) -> Self {
        VectorViewMut {
            data: &mut value.data
        }
    }
}
impl<'a,T,const N: usize> From<&'a mut OwnedVector<T,N>> for VectorViewMut<'a,T,N> {
     fn from(value: &'a mut OwnedVector<T,N>) -> Self {
        VectorViewMut {
            data: &mut value.data
        }
     }
}
impl<'a,T,const N: usize> Index<usize> for VectorViewMut<'a,T,N> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}
impl<'a,T,const N: usize> IndexMut<usize> for VectorViewMut<'a,T,N> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}
impl<'a,T,const N: usize> AsRef<[T;N]> for VectorViewMut<'a,T,N> {
    fn as_ref(&self) -> &[T; N] {
        &self.data
    }
}
impl<'a,T,const N: usize> AsMut<[T;N]> for VectorViewMut<'a,T,N> {
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }
}
pub struct VectorMut<'a,T,const N: usize,BE> where BE: Backend {
    data: &'a mut [T;N],
    backend: BE
}
impl<'a,BE: Backend,T,const N: usize> TryFrom<&'a mut [T]> for VectorMut<'a,T,N,BE> {
    type Error = InstantiationError;

    #[inline]
    fn try_from(value: &'a mut [T]) -> Result<Self,Self::Error> {
        if value.len() != N {
            Err(InstantiationError::from(TryFromSliceError))
        } else {
            Ok(VectorMut {
                data: value.try_into()?,
                backend: BE::new()?
            })
        }
    }
}
pub struct OwnedVector<T,const N: usize> {
    data: Box<[T; N]>
}
impl<T,const N: usize> From<OwnedVector<T,N>> for Box<[T;N]> {
    fn from(value: OwnedVector<T,N>) -> Self {
        value.data
    }
}
impl<T,const N: usize> From<Box<[T;N]>> for OwnedVector<T,N> {
    fn from(value: Box<[T;N]>) -> Self {
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
    pub fn row(&self,index:usize) -> Vector<'a,T,M,BE> {
        let view = &self.data[index * M..(index + 1) * M];

        Vector {
            data: view.try_into().unwrap(),
            backend: BE::new().unwrap()
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
                backend: BE::new().unwrap()
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
impl<'a,BE: Backend,T,const N: usize,const M: usize> TryFrom<&'a OwnedMatrix<T,N,M>> for Matrix<'a,T,N,M,BE> {
    type Error = InstantiationError;
    fn try_from(value: &'a OwnedMatrix<T,N,M>) -> Result<Self,Self::Error> {
        Ok(Matrix {
            data: &value.data,
            backend: BE::new()?
        })
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a Vector<'a,T,N,BE>> for Matrix<'a,T,N,1,BE> {
    fn from(value: &'a Vector<'a, T, N, BE>) -> Self {
        Matrix {
            data: value.data,
            backend: BE::new().unwrap()
        }
    }
}
pub struct MatrixView<'a,T,const N: usize,const M: usize> {
    data: &'a [T]
}
impl<'a,T,const N: usize,const M: usize> Dims<N,M> for MatrixView<'a,T,N,M> {}
impl<'a,T,const N: usize,const M: usize> MatrixView<'a,T,N,M> {
    #[inline]
    pub fn row(&self,index:usize) -> VectorView<'a,T,M> {
        let view = &self.data[index * M..(index + 1) * M];

        VectorView {
            data: view.try_into().unwrap()
        }
    }
}
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a [T]> for MatrixView<'a,T,M,N> {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(TryFromSliceError)
        } else {
            Ok(MatrixView {
                data: value
            })
        }
    }
}
impl<T,const N: usize,const M: usize> Index<usize> for MatrixView<'_,T,N,M> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * M)..(index * M + M)]
    }
}
impl<'a,T,const N: usize,const M: usize> Transpose<T,N,M> for MatrixView<'a,T,N,M>
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
impl<'a,T,const N: usize,const M: usize> ToColumnMajor<T,N,M> for MatrixView<'a,T,N,M>
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
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a OwnedMatrix<T,N,M>> for MatrixView<'a,T,N,M> {
    type Error = InstantiationError;
    fn try_from(value: &'a OwnedMatrix<T,N,M>) -> Result<Self,Self::Error> {
        Ok(MatrixView {
            data: &value.data
        })
    }
}
impl<'a,BE: Backend,T,const N: usize> From<&'a Vector<'a,T,N,BE>> for MatrixView<'a,T,N,1> {
    fn from(value: &'a Vector<'a, T, N, BE>) -> Self {
        MatrixView {
            data: value.data
        }
    }
}
impl<'a,T,const N: usize> From<&'a VectorView<'a,T,N>> for MatrixView<'a,T,N,1> {
    fn from(value: &'a VectorView<'a, T, N>) -> Self {
        MatrixView {
            data: value.data
        }
    }
}
pub struct MatrixMut<'a,T,const N: usize,const M: usize> {
    data: &'a mut [T]
}
impl<'a,T,const N: usize,const M: usize> Dims<N,M> for MatrixMut<'a,T,N,M> {}
impl<'a,T,const N: usize,const M: usize> Index<(usize,usize)> for MatrixMut<'a,T,N,M> {
    type Output = T;

    fn index(&self, (row,col): (usize,usize)) -> &Self::Output {
        &self.data[(row * M) + col]
    }
}
impl<'a,T,const N: usize,const M: usize> IndexMut<(usize,usize)> for MatrixMut<'a,T,N,M> {
    fn index_mut(&mut self, (row,col): (usize,usize)) -> &mut Self::Output {
        &mut self.data[(row * M) + col]
    }
}
impl<'a,T,const N: usize,const M: usize> From<&'a mut OwnedMatrix<T,N,M>> for MatrixMut<'a,T,N,M> {
    fn from(value: &'a mut OwnedMatrix<T, N, M>) -> Self {
        MatrixMut {
            data: &mut value.data
        }
    }
}
impl<'a,T,const M: usize> From<&'a mut OwnedVector<T,M>> for MatrixMut<'a,T,1,M> {
    fn from(value: &'a mut OwnedVector<T,M>) -> Self {
        MatrixMut {
            data: &mut *value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> AsMut<[T]> for MatrixMut<'a,T,N,M> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
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
impl<'a,T,const N: usize,const M: usize> ColumnMajorMatrix<'a,T,N,M> {
    #[inline]
    pub fn col(&self,index:usize) -> VectorView<'a,T,N> {
        let view = &self.data[index * N..(index + 1) * N];

        VectorView {
            data: view.try_into().unwrap()
        }
    }
}
pub struct ColumnMajorMatrix<'a,T,const N: usize,const M: usize> {
    data: &'a [T]
}
impl<'a,T,const N: usize,const M: usize> Dims<N,M> for ColumnMajorMatrix<'a,T,N,M> {}
impl<'a,T,const N: usize,const M: usize> TryFrom<&'a [T]> for ColumnMajorMatrix<'a,T,M,N> {
    type Error = TryFromSliceError;

    #[inline]
    fn try_from(value: &'a [T]) -> Result<Self,Self::Error> {
        if value.len() != N * M {
            Err(TryFromSliceError)
        } else {
            Ok(ColumnMajorMatrix {
                data: value
            })
        }
    }
}
impl<'a,BE: Backend,T,const M: usize> From<&'a Vector<'a,T,M,BE>> for ColumnMajorMatrix<'a,T,1,M> {
    fn from(value: &'a Vector<'a, T, M, BE>) -> Self {
        ColumnMajorMatrix {
            data: value.data
        }
    }
}
impl<'a,T,const M: usize> From<&'a VectorView<'a,T,M>> for ColumnMajorMatrix<'a,T,1,M> {
    fn from(value: &'a VectorView<'a, T, M>) -> Self {
        ColumnMajorMatrix {
            data: value.data
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
impl<'a,T,const N: usize,const M: usize> From<&'a OwnedColumnMajorMatrix<T,N,M>> for ColumnMajorMatrix<'a,T,N,M> {
    fn from(value: &'a OwnedColumnMajorMatrix<T,N,M>) -> Self {
        ColumnMajorMatrix {
            data: &value.data
        }
    }
}
impl<'a,T,const N: usize,const M: usize> From<&'a ColumnMajorMatrix<'a,T,N,M>>
    for OwnedColumnMajorMatrix<T,M,N>
    where T: Clone + Copy {
    fn from(value: &'a ColumnMajorMatrix<'a,T,N,M>) -> Self {
        OwnedColumnMajorMatrix {
            data: value.data.to_vec().into_boxed_slice()
        }
    }
}
impl<'a,T,const N: usize> Add<&'a Vector<'a,T,N,AutoSelect>> for &'a Vector<'a,T,N,AutoSelect>
    where AutoSelect: Backend,
          Avx2: SimdAddVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = OwnedVector<T,N>;

    fn add(self, rhs: &'a Vector<'a,T,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.add_vector(&self.into(), &rhs.into())
        }
    }
}
impl<'a,BE,T,const N: usize> Add<&'a Vector<'a,T,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdAddVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = OwnedVector<T,N>;

    fn add(self, rhs: &'a Vector<'a,T,N,BE>) -> Self::Output {
        self.backend.add_vector(&self.into(), &rhs.into())
    }
}
impl<'a,T,const N: usize> AddAssign<&'a Vector<'a,T,N,AutoSelect>> for VectorViewMut<'a,T,N>
    where Avx2: Backend + SimdAddAssignVector<T,T>,
          VectorViewMut<'a,T,N>: From<&'a mut VectorMut<'a,T,N,AutoSelect>>,
          VectorView<'a,T,N>: From<&'a Vector<'a,T,N,AutoSelect>> {
    fn add_assign(&mut self, rhs: &'a Vector<'a,T,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.add_assign_vector(self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,T,const N: usize> AddAssign<&'a Vector<'a,T,N,BE>> for VectorViewMut<'a,T,N>
    where BE: Backend + SimdAddAssignVector<T,T>,
          for<'b> VectorViewMut<'b,T,N>: From<&'b mut Vector<'b,T,N,BE>>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    fn add_assign(&mut self, rhs: &'a Vector<'a,T,N,BE>) {
        rhs.backend.add_assign_vector(self, &rhs.into())
    }
}
impl<'a,T,const N: usize> SubAssign<&'a Vector<'a,T,N,AutoSelect>> for VectorViewMut<'a,T,N>
    where Avx2: Backend + SimdSubAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {

    fn sub_assign(&mut self, rhs: &'a Vector<'a,T,N,AutoSelect>) {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.sub_assign_vector(self, &rhs.into()),
        }
    }
}
impl<'a,BE,T,const N: usize> SubAssign<&'a Vector<'a,T,N,BE>> for VectorViewMut<'a,T,N>
    where BE: Backend + SimdSubAssignVector<T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {

    fn sub_assign(&mut self, rhs: &'a Vector<'a,T,N,BE>) {
        rhs.backend.sub_assign_vector(self, &rhs.into())
    }
}
impl<'a,T,const N: usize> Sub<&'a Vector<'a,T,N,AutoSelect>> for &'a Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdSubVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = OwnedVector<T,N>;

    fn sub(self, rhs: &'a Vector<'a,T,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.sub_vector(&self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,T,const N: usize> Sub<&'a Vector<'a,T,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdSubVector<T,T,T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = OwnedVector<T,N>;

    fn sub(self, rhs: &'a Vector<'a,T,N,BE>) -> Self::Output {
        self.backend.sub_vector(&self.into(), &rhs.into())
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,i8,N,AutoSelect>> for &'a Vector<'a,i8,N,AutoSelect>
    where Avx2: Backend + SimdMulVector<i8,i8,i32>,
                for<'b> VectorView<'b,i8,N>: From<&'b Vector<'a,i8,N,AutoSelect>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i8,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.mul_vector(&self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i8,N,BE>> for &'a Vector<'a,i8,N,BE>
    where BE: Backend + SimdMulVector<i8,i8,i32>,
          for<'b> VectorView<'b,i8,N>: From<&'b Vector<'a,i8,N,BE>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i8,N,BE>) -> Self::Output {
        self.backend.mul_vector(&self.into(), &rhs.into())
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,i16,N,AutoSelect>> for &'a Vector<'a,i16,N,AutoSelect>
    where Avx2: Backend + SimdMulVector<i16,i16,i32>,
      for<'b> VectorView<'b,i16,N>: From<&'b Vector<'a,i16,N,AutoSelect>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.mul_vector(&self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i16,N,BE>> for &'a Vector<'a,i16,N,BE>
    where BE: Backend + SimdMulVector<i16,i16,i32>,
          for<'b> VectorView<'b,i16,N>: From<&'b Vector<'a,i16,N,BE>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,BE>) -> Self::Output {
        self.backend.mul_vector(&self.into(), &rhs.into())
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,i32,N,AutoSelect>> for &'a Vector<'a,i32,N,AutoSelect>
    where Avx2: Backend + SimdMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'a,i32,N,AutoSelect>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i32,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.mul_vector(&self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i32,N,BE>> for &'a Vector<'a,i32,N,BE>
    where BE: Backend + SimdMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'a,i32,N,BE>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i32,N,BE>) -> Self::Output {
        self.backend.mul_vector(&self.into(), &rhs.into())
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,i8,N,AutoSelect>> for i8
    where Avx2: Backend + SimdScalarMulVector<i8,i8,i32>,
          for<'b> VectorView<'b,i8,N>: From<&'b Vector<'a,i8,N,AutoSelect>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i8,N,AutoSelect>) -> Self::Output {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.scalarmul_vector(self, &rhs.into()),
        }
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i8,N,BE>> for i8
    where BE: Backend + SimdScalarMulVector<i8,i8,i32>,
          for<'b> VectorView<'b,i8,N>: From<&'b Vector<'a,i8,N,BE>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i8,N,BE>) -> Self::Output {
        rhs.backend.scalarmul_vector(self, &rhs.into())
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,i16,N,AutoSelect>> for i16
    where Avx2: Backend + SimdScalarMulVector<i16,i16,i32>,
          for<'b> VectorView<'b,i16,N>: From<&'b Vector<'a,i16,N,AutoSelect>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i16,N,AutoSelect>) -> Self::Output {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.scalarmul_vector(self, &rhs.into()),
        }
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,i32,N,AutoSelect>> for i32
    where Avx2: Backend + SimdScalarMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'a,i32,N,AutoSelect>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i32,N,AutoSelect>) -> Self::Output {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.scalarmul_vector(self, &rhs.into()),
        }
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,i32,N,BE>> for i32
    where BE: Backend + SimdScalarMulVector<i32,i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'a,i32,N,BE>> {
    type Output = OwnedVector<i32,N>;

    fn mul(self, rhs: &'a Vector<'a,i32,N,BE>) -> Self::Output {
        rhs.backend.scalarmul_vector(self, &rhs.into())
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,f32,N,AutoSelect>> for f32
    where Avx2: Backend + SimdScalarMulVector<f32,f32,f32>,
          for<'b> VectorView<'b,f32,N>: From<&'b Vector<'a,f32,N,AutoSelect>> {
    type Output = OwnedVector<f32,N>;

    fn mul(self, rhs: &'a Vector<'a,f32,N,AutoSelect>) -> Self::Output {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.scalarmul_vector(self, &rhs.into()),
        }
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,f32,N,BE>> for f32
    where BE: Backend + SimdScalarMulVector<f32,f32,f32>,
          for<'b> VectorView<'b,f32,N>: From<&'b Vector<'a,f32,N,BE>> {
    type Output = OwnedVector<f32,N>;

    fn mul(self, rhs: &'a Vector<'a,f32,N,BE>) -> Self::Output {
        rhs.backend.scalarmul_vector(self, &rhs.into())
    }
}
impl<'a,const N: usize> Mul<&'a Vector<'a,f64,N,AutoSelect>> for f64
    where Avx2: Backend + SimdScalarMulVector<f64,f64,f64>,
          for<'b> VectorView<'b,f64,N>: From<&'b Vector<'a,f64,N,AutoSelect>> {
    type Output = OwnedVector<f64,N>;

    fn mul(self, rhs: &'a Vector<'a,f64,N,AutoSelect>) -> Self::Output {
        match rhs.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.scalarmul_vector(self, &rhs.into()),
        }
    }
}
impl<'a,BE,const N: usize> Mul<&'a Vector<'a,f64,N,BE>> for f64
    where BE: Backend + SimdScalarMulVector<f64,f64,f64>,
          for<'b> VectorView<'b,f64,N>: From<&'b Vector<'a,f64,N,BE>> {
    type Output = OwnedVector<f64,N>;

    fn mul(self, rhs: &'a Vector<'a,f64,N,BE>) -> Self::Output {
        rhs.backend.scalarmul_vector(self,&rhs.into())
    }
}
impl<'a,BE,const N: usize> MulAssign<&'a Vector<'a,i32,N,BE>> for VectorViewMut<'a,i32,N>
    where BE: Backend + SimdMulAssignVector<i32,i32>,
          for<'b> VectorView<'b,i32,N>: From<&'b Vector<'a,i32,N,BE>> {
    fn mul_assign(&mut self, rhs: &'a Vector<'a,i32,N,BE>) {
        rhs.backend.mul_assign_vector(self, &rhs.into())
    }
}
impl<'a,BE,const N: usize> MulAssign<Scalar<i32,BE>> for VectorViewMut<'a,i32,N>
    where BE: Backend + SimdScalarMulAssignVector<i32,i32> {
    fn mul_assign(&mut self, rhs: Scalar<i32,BE>) {
        rhs.backend.scalarmul_assign_vector(rhs.value,self)
    }
}
impl<'a,BE,const N: usize> MulAssign<&'a Vector<'a,f32,N,BE>> for VectorViewMut<'a,f32,N>
    where BE: Backend + SimdMulAssignVector<f32,f32>,
          for<'b> VectorView<'b,f32,N>: From<&'b Vector<'a,f32,N,BE>> {
    fn mul_assign(&mut self, rhs: &'a Vector<'a,f32,N,BE>) {
        rhs.backend.mul_assign_vector(self, &rhs.into())
    }
}
impl<'a,BE,const N: usize> MulAssign<Scalar<f32,BE>> for VectorViewMut<'a,f32,N>
    where BE: Backend + SimdScalarMulAssignVector<f32,f32> {
    fn mul_assign(&mut self, rhs: Scalar<f32,BE>) {
        rhs.backend.scalarmul_assign_vector(rhs.value,self)
    }
}
impl<'a,BE,const N: usize> MulAssign<&'a Vector<'a,f64,N,BE>> for VectorViewMut<'a,f64,N>
    where BE: Backend + SimdMulAssignVector<f64,f64>,
          for<'b> VectorView<'b,f64,N>: From<&'b Vector<'a,f64,N,BE>> {
    fn mul_assign(&mut self, rhs: &'a Vector<'a,f64,N,BE>) {
        rhs.backend.mul_assign_vector(self, &rhs.into())
    }
}
impl<'a,BE,const N: usize> MulAssign<Scalar<f64,BE>> for VectorViewMut<'a,f64,N>
    where BE: Backend + SimdScalarMulAssignVector<f64,f64> {
    fn mul_assign(&mut self, rhs: Scalar<f64,BE>) {
        rhs.backend.scalarmul_assign_vector(rhs.value,self)
    }
}
impl<'a,T,const N: usize> BitXor<&'a Vector<'a,<T as BitsBitXor>::Bits,N,AutoSelect>> for &'a Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdBitXorVector<T>,
          T: BitsBitXor,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorView<'b,<T as BitsBitXor>::Bits,N>: From<&'b Vector<'a,<T as BitsBitXor>::Bits,N,AutoSelect>> {
    type Output = OwnedVector<T,N>;

    fn bitxor(self, rhs: &'a Vector<'a,<T as BitsBitXor>::Bits,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.bitxor_vector(&self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,T,const N: usize> BitXor<&'a Vector<'a,<T as BitsBitXor>::Bits,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitXor>::Bits> +
              SimdReg<<T as BitsBitXor>::Bits> +
              SimdBitXorVector<T>,
              T: BitsBitXor,
              for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
              for<'b> VectorView<'b,<T as BitsBitXor>::Bits,N>: From<&'b Vector<'a,<T as BitsBitXor>::Bits,N,BE>> {
    type Output = OwnedVector<T,N>;

    fn bitxor(self, rhs: &'a Vector<'a,<T as BitsBitXor>::Bits,N,BE>) -> Self::Output {
        self.backend.bitxor_vector(&self.into(), &rhs.into())
    }
}
impl<'a,T,const N: usize> BitOr<&'a Vector<'a,<T as BitsBitOr>::Bits,N,AutoSelect>> for &'a Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdBitOrVector<T>,
          T: BitsBitOr,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
          for<'b> VectorView<'b,<T as BitsBitOr>::Bits,N>: From<&'b Vector<'a,<T as BitsBitOr>::Bits,N,AutoSelect>> {
    type Output = OwnedVector<T,N>;

    fn bitor(self, rhs: &'a Vector<'a,<T as BitsBitOr>::Bits,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.bitor_vector(&self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,T,const N: usize> BitOr<&'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitOr>::Bits> +
              SimdReg<<T as BitsBitOr>::Bits> +
              SimdBitOrVector<T>,
              T: BitsBitOr,
              for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
              for<'b> VectorView<'b,<T as BitsBitOr>::Bits,N>: From<&'b Vector<'a,<T as BitsBitOr>::Bits,N,BE>> {
    type Output = OwnedVector<T,N>;

    fn bitor(self, rhs: &'a Vector<'a,<T as BitsBitOr>::Bits,N,BE>) -> Self::Output {
        self.backend.bitor_vector(&self.into(), &rhs.into())
    }
}
impl<'a,T,const N: usize> BitAnd<&'a Vector<'a,<T as BitsBitAnd>::Bits,N,AutoSelect>> for &'a Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdBitAndVector<T>,
      T: BitsBitAnd,
      for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>>,
      for<'b> VectorView<'b,<T as BitsBitAnd>::Bits,N>: From<&'b Vector<'a,<T as BitsBitAnd>::Bits,N,AutoSelect>> {

    type Output = OwnedVector<T,N>;

    fn bitand(self, rhs: &'a Vector<'a,<T as BitsBitAnd>::Bits,N,AutoSelect>) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.bitand_vector(&self.into(), &rhs.into()),
        }
    }
}
impl<'a,BE,T,const N: usize> BitAnd<&'a Vector<'a,<BE as SimdReg<T>>::Bits,N,BE>> for &'a Vector<'a,T,N,BE>
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitAnd>::Bits> +
              SimdReg<<T as BitsBitAnd>::Bits> +
              SimdBitAndVector<T>,
              T: BitsBitAnd,
              for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>>,
              for<'b> VectorView<'b,<T as BitsBitAnd>::Bits,N>: From<&'b Vector<'a,<T as BitsBitAnd>::Bits,N,BE>> {

    type Output = OwnedVector<T,N>;

    fn bitand(self, rhs: &'a Vector<'a,<T as BitsBitAnd>::Bits,N,BE>) -> Self::Output {
        self.backend.bitand_vector(&self.into(), &rhs.into())
    }
}
impl<'a,T,const N: usize> Not for &'a Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdBitNotVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {

    type Output = OwnedVector<T,N>;

    fn not(self) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.bitnot_vector(&self.into()),
        }
    }
}
impl<'a,BE,T,const N: usize> Not for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdBitNotVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {

    type Output = OwnedVector<T,N>;

    fn not(self) -> Self::Output {
        self.backend.bitnot_vector(&self.into())
    }
}
impl<'a,T,const N: usize> Shl<usize> for &'a Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdShlVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = OwnedVector<T,N>;

    fn shl(self, rhs: usize) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.shl_vector(&self.into(), rhs),
        }
    }
}
impl<'a,BE,T,const N: usize> Shl<usize> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdShlVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = OwnedVector<T,N>;

    fn shl(self, rhs: usize) -> Self::Output {
        self.backend.shl_vector(&self.into(), rhs)
    }
}
impl<'a,T,const N: usize> Shr<usize> for &'a Vector<'a,T,N,AutoSelect>
    where Avx2: Backend + SimdShrVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,AutoSelect>> {
    type Output = OwnedVector<T,N>;

    fn shr(self, rhs: usize) -> Self::Output {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.shr_vector(&self.into(), rhs),
        }
    }
}
impl<'a,BE,T,const N: usize> Shr<usize> for &'a Vector<'a,T,N,BE>
    where BE: Backend + SimdShrVector<T>,
          for<'b> VectorView<'b,T,N>: From<&'b Vector<'b,T,N,BE>> {
    type Output = OwnedVector<T,N>;

    fn shr(self, rhs: usize) -> Self::Output {
        self.backend.shr_vector(&self.into(), rhs)
    }
}
impl<'a,SL,SR,const N: usize> From<&'a Vector<'a,SL,N,AutoSelect>> for OwnedVector<SR,N>
    where Avx2: Backend + SimdPromoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>> {
    fn from(s:&'a Vector<'a,SL,N,AutoSelect>) -> OwnedVector<SR,N> {
        match s.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.promotion_vector(&s.into()),
        }
    }
}
impl<'a,BE,SL,SR,const N: usize> From<&'a Vector<'a,SL,N,BE>> for OwnedVector<SR,N>
    where BE: Backend + SimdPromoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>> {
    fn from(s:&'a Vector<'a,SL,N,BE>) -> OwnedVector<SR,N> {
        s.backend.promotion_vector(&s.into())
    }
}
impl<'a,SL,SR,const N: usize> From<&'a Vector<'a,SL,N,AutoSelect>> for OwnedVector<SR,N>
    where Avx2: Backend + SimdDemoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>> {
    fn from(s:&'a Vector<'a,SL,N,AutoSelect>) -> OwnedVector<SR,N> {
        match s.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.demotion_vector(&s.into()),
        }
    }
}
impl<'a,BE,SL,SR,const N: usize> From<&'a Vector<'a,SL,N,BE>> for OwnedVector<SR,N>
    where BE: Backend + SimdDemoteVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>> {
    fn from(s:&'a Vector<'a,SL,N,BE>) -> OwnedVector<SR,N> {
        s.backend.demotion_vector(&s.into())
    }
}
impl<'a,SL,SR,const N: usize> From<&'a Vector<'a,SL,N,AutoSelect>> for OwnedVector<SR,N>
    where Avx2: Backend + SimdConvertVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>> {
    fn from(s:&'a Vector<'a,SL,N,AutoSelect>) -> OwnedVector<SR,N> {
        match s.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.convert_vector(&s.into()),
        }
    }
}
impl<'a,BE,SL,SR,const N: usize> From<&'a Vector<'a,SL,N,BE>> for OwnedVector<SR,N>
    where BE: Backend + SimdConvertVector<SL,SR>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>> {
    fn from(s:&'a Vector<'a,SL,N,BE>) -> OwnedVector<SR,N> {
        s.backend.convert_vector(&s.into())
    }
}
impl<'a,SL,SR,SO,const N: usize> Dot<&Vector<'a,SR,N,AutoSelect>,SO> for Vector<'a,SL,N,AutoSelect>
    where Avx2: Backend + SimdDot<SL,SR,SO>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>>,
          for<'b> VectorView<'b,SR,N>: From<&'b Vector<'b,SR,N,AutoSelect>> {
    fn dot(&self,r:&Vector<'_,SR,N,AutoSelect>) -> SO {
        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => backend.dot(&self.into(),&r.into()),
        }
    }
}
impl<'a,BE,SL,SR,SO,const N: usize> Dot<&Vector<'a,SR,N,BE>,SO> for Vector<'a,SL,N,BE>
    where BE: Backend + SimdDot<SL,SR,SO>,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>>,
          for<'b> VectorView<'b,SR,N>: From<&'b Vector<'b,SR,N,BE>> {
    fn dot(&self,r:&Vector<'_,SR,N,BE>) -> SO {
        self.backend.dot(&self.into(),&r.into())
    }
}
impl<'a,SL,SR,SO,const N: usize,const M: usize> Product<&'a Vector<'a,SR,M,AutoSelect>,OwnedMatrix<SO,N,M>>  for Vector<'a,SL,N,AutoSelect>
        where Avx2: Backend + SimdOuterProduct<SL,SR,SO>,
              SO: Default + Copy + Clone,
                  for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,AutoSelect>>,
                  for<'b> VectorView<'b,SR,M>: From<&'b Vector<'b,SR,M,AutoSelect>> {
    fn product(&self,r:&'a Vector<'a,SR,M,AutoSelect>) -> OwnedMatrix<SO,N,M> {
        let mut o = OwnedMatrix::<SO,N,M>::default();

        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => {
                backend.outer_product(&self.into(),&r.into(),&mut o);
            }
        }

        o
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const M: usize> Product<&'a Vector<'a,SR,M,BE>,OwnedMatrix<SO,N,M>>
    for Vector<'a,SL,N,BE>
    where BE: Backend + SimdOuterProduct<SL,SR,SO>,
          SO: Default + Copy + Clone,
          for<'b> VectorView<'b,SL,N>: From<&'b Vector<'b,SL,N,BE>>,
          for<'b> VectorView<'b,SR,M>: From<&'b Vector<'b,SR,M,BE>> {
    fn product(&self,r:&'a Vector<'a,SR,M,BE>) -> OwnedMatrix<SO,N,M> {
        let mut o = OwnedMatrix::<SO,N,M>::default();
        self.backend.outer_product(&self.into(),&r.into(),&mut o);

        o
    }
}
impl<'a,SL,SR,SO,const M: usize,const K: usize> Product<&'a ColumnMajorMatrix<'a,SR,K,M>,OwnedVector<SO,M>> for Vector<'a,SL,K,AutoSelect>
    where Avx2: Backend + SimdVMat<SL,SR,SO>,
          SO: Default + Copy + Clone ,
          for<'b> VectorView<'b,SL,K>: From<&'b Vector<'b,SL,K,AutoSelect>> {
    fn product(&self,r:&'a ColumnMajorMatrix<'a,SR,K,M>) -> OwnedVector<SO,M> {
        let mut o = OwnedVector::<SO,M>::default();

        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => {
                backend.vmat(&self.into(),r,&mut o);
            }
        }

        o
    }
}
impl<'a,BE,SL,SR,SO,const M: usize,const K: usize> Product<&'a ColumnMajorMatrix<'a,SR,K,M>,OwnedVector<SO,M>>
    for Vector<'a,SL,K,BE>
    where BE: Backend + SimdVMat<SL,SR,SO>,
          SO: Default + Copy + Clone ,
          for<'b> VectorView<'b,SL,K>: From<&'b Vector<'b,SL,K,BE>> {
    fn product(&self,r:&'a ColumnMajorMatrix<'a,SR,K,M>) -> OwnedVector<SO,M> {
        let mut o = OwnedVector::<SO,M>::default();

        self.backend.vmat(&self.into(),r,&mut o);

        o
    }
}
impl<'a,SL,SR,SO,const N: usize,const K: usize> Product<&'a Vector<'a,SR,K,AutoSelect>,OwnedVector<SO,N>> for Matrix<'a,SL,N,K,AutoSelect>
    where Avx2: Backend + SimdMatVec<SL,SR,SO>,
          SO: Default + Copy + Clone,
          for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,AutoSelect>>,
          for<'b> VectorView<'b,SR,K>: From<&'b Vector<'b,SR,K,AutoSelect>> {
    fn product(&self, r: &'a Vector<'a,SR,K,AutoSelect>) -> OwnedVector<SO,N> {
        let mut o = OwnedVector::<SO,N>::default();

        match self.backend.selected {
            SelectedBackend::Avx2(ref backend) => {
                backend.matvec(&self.into(),&r.into(),&mut o);
            }
        }

        o
    }
}
impl<'a,BE,SL,SR,SO,const N: usize,const K: usize> Product<&'a Vector<'a,SR,K,BE>,OwnedVector<SO,N>>
    for Matrix<'a,SL,N,K,BE>
    where BE: Backend + SimdMatVec<SL,SR,SO>,
        SO: Default + Copy + Clone,
        for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,BE>>,
        for<'b> VectorView<'b,SR,K>: From<&'b Vector<'b,SR,K,BE>> {
    fn product(&self, r: &'a Vector<'a,SR,K,BE>) -> OwnedVector<SO,N> {
        let mut o = OwnedVector::<SO,N>::default();

        self.backend.matvec(&self.into(), &r.into(), &mut o);

        o
    }
}
impl<'a,SL,SR,SO,const N: usize,const M: usize,const K: usize> Product<&'a ColumnMajorMatrix<'a,SR,K,M>,OwnedMatrix<SO,N,M>>
    for Matrix<'a,SL,N,K,AutoSelect>
    where Avx2: Backend + SimdMatMul<SL,SR,SO>,
          SO: Default + Copy + Clone,
          for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,AutoSelect>> {
    fn product(&self, r: &'a ColumnMajorMatrix<'a,SR,K,M>) -> OwnedMatrix<SO,N,M> {
        let mut o = OwnedMatrix::<SO,N,M>::default();
        {
            let mut o = (&mut o).into();

            match self.backend.selected {
                SelectedBackend::Avx2(ref backend) => {
                    backend.matmul(&self.into(),r,&mut o);
                }
            }
        }

        o
    }
}impl<'a,BE,SL,SR,SO,const N: usize,const M: usize,const K: usize> Product<&'a ColumnMajorMatrix<'a,SR,K,M>,OwnedMatrix<SO,N,M>>
    for Matrix<'a,SL,N,K,BE>
    where BE: Backend + SimdMatMul<SL,SR,SO>,
          SO: Default + Copy + Clone,
          for<'b> MatrixView<'b,SL,N,K>: From<&'b Matrix<'b,SL,N,K,BE>> {
    fn product(&self, r: &'a ColumnMajorMatrix<'a,SR,K,M>) -> OwnedMatrix<SO,N,M> {
        let mut o = OwnedMatrix::<SO,N,M>::default();
        {
            let mut o = (&mut o).into();
            self.backend.matmul(&self.into(), r, &mut o);
        }

        o
    }
}