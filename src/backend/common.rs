//! Common Backend Implementation

use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Mul, Sub};
use crate::traits::{SimdBitNotVector, SimdBitOrVector, SimdBitXorVector, SimdMulVector, SimdSubVector, SimdMask, SimdScalarMulVector, SimdLoad, SimdStore, SimdReg, SimdLanes, SimdRows, SimdAdd, SimdSub, SimdMul, SimdStoreSeq, SimdSplat, SimdCols, BitsBitAnd, BitsBitOr, BitsBitXor, BitsBitNot, SimdBitAndVector, SimdBitAnd, SimdBitOr, SimdBitXor, SimdBitNot, BitsShl, BitsShr, SimdShlVector, SimdShl, SimdShrVector, SimdShr, SimdPromote, SimdPromoteVector, Assume, SimdDemoteVector, SimdDemote, SimdConvertVector, SimdConvert, SupportMul, FoldRegs, SimdOuterProduct, SimdMulAssignVector, SimdAddAssignVector, SimdSubAssignVector, SimdShiftWidth, SimdScalarMulAssignVector, SimdAddAssignMatrix, SimdScalarMulAssignMatrix, SimdScalarMulMatrix, SimdConvertMatrix, SimdLoadSeq, SimdScalarMulVectorInto, SimdHSum, SimdZero, SimdPartialDot, SimdDotKernel, IsScaler, SimdMatMulKernel, SimdQueryVector, SimdBinaryOperator, NativeBackend, Dim, DimOne, Loader, Mapper, SimdPartialOperator, BinaryOperator, ToSimdBinaryOperator, SimdResolvePatrialBinaryOperator, SimdResolveBinaryOperator, SimdPartialLoadSeq, SimdPartialStoreSeq, SimdPartialLoad, SimdRegBuf, SimdPartialStore, SimdLoadAligned, SimdStoreAligned, Sign, SimdSign, SimdLinearCombinationKernel};
use crate::{ColumnMajorMatrix, MatrixMutView, MatrixView, OwnedVector, VectorMutView, VectorView};
use crate::error::InstantiationError;
use crate::traits::kernel::SimdCombinationQueryVectorKernel;

/// A trait that defines the instantiation of a SIMD arithmetic backend
pub trait Backend: Sized + Send + Sync + 'static {
    fn new() -> Result<Self,InstantiationError>;
}
/// Calculate the number of registers actually used
pub const fn used_regs<BE,T,const REGS: usize, const N: usize>() -> usize where BE: SimdLanes<T> {
    (N % (REGS * <BE as SimdLanes<T>>::LANES)) / <BE as SimdLanes<T>>::LANES
}
/// Calculate a size that does not fit within the SIMD register size
pub const fn remaining<BE,T,const N: usize>() -> usize where BE: SimdLanes<T> {
    N % <BE as SimdLanes<T>>::LANES
}
/// A set of one or more registers
#[derive(Clone,Copy)]
pub struct Regs<R,const N:usize> where R: Copy {
    regs:[R;N]
}
impl<R,const N:usize> Regs<R,N> where R: Copy {
    #[inline(always)]
    pub fn new(regs:[R;N]) -> Self {
        Regs { regs }
    }
}
impl<R,const N:usize> AsRef<[R;N]> for Regs<R,N>
    where R: Copy {
    #[inline(always)]
    fn as_ref(&self) -> &[R;N] {
        &self.regs
    }
}
/// A type representing an operator that performs addition in the SIMD registers
pub struct SimdAddOperator<SL,SR,SO,BE>
    where BE: Backend + NativeBackend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdAdd<SL,SR,SO> {
    sl:PhantomData<SL>,
    sr:PhantomData<SR>,
    so:PhantomData<SO>,
    backend: BE
}
impl<SL,SR,SO,BE> SimdAddOperator<SL,SR,SO,BE>
    where BE: Backend + NativeBackend + Clone +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdAdd<SL,SR,SO> {
    #[inline(always)]
    pub fn new(backend:&BE) -> Self {
        Self {
            sl:PhantomData,
            sr:PhantomData,
            so:PhantomData,
            backend:backend.clone()
        }
    }
}
impl<BE,SL,SR,SO> SimdBinaryOperator<BE,SL,SR> for SimdAddOperator<SL,SR,SO,BE>
    where BE: Backend + NativeBackend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdAdd<SL,SR,SO,Lhs=<BE as SimdReg<SL>>::Reg,Rhs=<BE as SimdReg<SR>>::Reg,Output=<BE as SimdReg<SO>>::Reg>,
          SO: From<SL> + From<SR> + Add<SO,Output=SO> + Copy {
    type Lhs = <BE as SimdReg<SL>>::Reg;
    type Rhs = <BE as SimdReg<SR>>::Reg;
    type Output = <BE as SimdReg<SO>>::Reg;
    type ScalarOutput = SO;

    #[inline(always)]
    fn apply_lanes(&self, l: Self::Lhs, r: Self::Rhs) -> Self::Output {
        unsafe { self.backend.add(l,r) }
    }

    #[inline(always)]
    fn apply_scalar(&self, l: SL, r: SR) -> SO {
        SO::from(l) + SO::from(r)
    }

    #[inline(always)]
    fn compose<F: Fn(Self) -> OP,OP>(self, f: F) -> OP where OP: SimdBinaryOperator<BE,SL,SR> {
        f(self)
    }
}
impl<BE,SL,SR,SO> SimdPartialOperator<BE> for SimdAddOperator<SL,SR,SO,BE>
    where BE: Backend + NativeBackend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdAdd<SL,SR,SO>,
          SO: From<SL> + From<SR> + Add<SO,Output=SO> {
    #[inline(always)]
    fn compose<F: Fn(Self) -> OP,OP>(self, f: F) -> OP {
        f(self)
    }
}
/// A version that combines multiple operators of a type representing operators
/// that perform addition within SIMD registers
pub struct SimdQueryAddOperator<SL,SR,SO,BE>
    where BE: Backend + NativeBackend {
    sl:PhantomData<SL>,
    sr:PhantomData<SR>,
    so:PhantomData<SO>,
    backend: BE
}
impl<SL,SR,SO,BE> SimdQueryAddOperator<SL,SR,SO,BE>
    where BE: Backend + NativeBackend + Clone {
    #[inline(always)]
    pub fn new(backend:&BE) -> Self {
        Self {
            sl:PhantomData,
            sr:PhantomData,
            so:PhantomData,
            backend:backend.clone()
        }
    }
}
impl<BE> SimdBinaryOperator<BE,i32,i8> for SimdQueryAddOperator<i32,i8,i32,BE>
    where BE: Backend + NativeBackend +
              SimdReg<i32> +
              SimdReg<i16> +
              SimdReg<i8> +
              SimdPromote<i8,i16> +
              SimdPromote<i16,i32> +
              SimdPromote<i8,i32,Output=Regs<<BE as SimdReg<i32>>::Reg,4>> +
              SimdAdd<i32,i32,i32,Lhs=<BE as SimdReg<i32>>::Reg,Rhs=<BE as SimdReg<i32>>::Reg,Output=<BE as SimdReg<i32>>::Reg> {
    type Lhs = Regs<<BE as SimdReg<i32>>::Reg,4>;
    type Rhs = Regs<<BE as SimdReg<i8>>::Reg,1>;
    type Output = Regs<<BE as SimdReg<i32>>::Reg,4>;
    type ScalarOutput = i32;

    #[inline(always)]
    fn apply_lanes(&self, l: Self::Lhs, r: Self::Rhs) -> Self::Output {
        unsafe {
            let &[a,b,c,d] = l.as_ref();

            let regs = <BE as SimdPromote<i8,i32>>::promotion(&self.backend,r.as_ref()[0]);

            let &[ra,rb,rc,rd] = regs.as_ref();

            Regs::new([
                <BE as SimdAdd<i32,i32,i32>>::add(&self.backend,a,ra),
                <BE as SimdAdd<i32,i32,i32>>::add(&self.backend,b,rb),
                <BE as SimdAdd<i32,i32,i32>>::add(&self.backend,c,rc),
                <BE as SimdAdd<i32,i32,i32>>::add(&self.backend,d,rd),
            ])
        }
    }

    #[inline(always)]
    fn apply_scalar(&self, l: i32, r: i8) -> i32 {
        l + r as i32
    }

    #[inline(always)]
    fn compose<F: Fn(Self) -> OP,OP>(self, f: F) -> OP where OP: SimdBinaryOperator<BE,i32,i8> {
        f(self)
    }
}
impl<BE,SL,SR,SO> SimdPartialOperator<BE> for SimdQueryAddOperator<SL,SR,SO,BE>
    where BE: Backend + NativeBackend,
          SO: From<SL> + From<SR> + Add<SO,Output=SO> {
    #[inline(always)]
    fn compose<F: Fn(Self) -> OP,OP>(self, f: F) -> OP {
        f(self)
    }
}
/// A version that combines multiple operators of a type representing right-shift operators within SIMD registers
pub struct SimdQueryShrOperator<BE,P,IO,T>
    where BE: Backend + NativeBackend + SimdReg<T>,
          IO: Copy,
          T: Copy {
    backend: BE,
    parent: P,
    input_output: PhantomData<IO>,
    scalar: PhantomData<T>,
    width: <BE as SimdReg<T>>::ShiftWidth,
    sw: usize
}
impl<BE,P,IO,T> SimdQueryShrOperator<BE,P,IO,T>
    where BE: Backend + NativeBackend + SimdReg<T> + SimdShiftWidth<T> + Clone,
          IO: Copy,
          T: Copy {
    #[inline(always)]
    pub fn new(backend: &BE,parent:P,width: usize) -> Self {
        let sw = width;
        let width = backend.shift_width(width);

        Self {
            backend: backend.clone(),
            parent: parent,
            input_output: PhantomData::<IO>,
            scalar: PhantomData::<T>,
            width: width,
            sw: sw
        }
    }
}
impl<BE,P> SimdPartialOperator<BE> for SimdQueryShrOperator<BE,P,Regs<<BE as SimdReg<i32>>::Reg,4>,i32>
    where BE: Backend + NativeBackend +
              SimdReg<i32> +
              SimdShr<i32> {
    #[inline(always)]
    fn compose<F: Fn(Self) -> OP,OP>(self, f: F) -> OP {
        f(self)
    }
}
impl<BE,P,SL,SR> SimdBinaryOperator<BE,SL,SR> for SimdQueryShrOperator<BE,P,Regs<<BE as SimdReg<i32>>::Reg,4>,i32>
    where BE: Backend + NativeBackend +
              SimdReg<i32> +
              SimdShr<i32>,
          P: SimdBinaryOperator<BE,SL,SR,Output=Regs<<BE as SimdReg<i32>>::Reg,4>,ScalarOutput=i32> {
    type Lhs = P::Lhs;
    type Rhs = P::Rhs;
    type Output = Regs<<BE as SimdReg<i32>>::Reg,4>;
    type ScalarOutput = i32;

    #[inline(always)]
    fn apply_lanes(&self, l: Self::Lhs, r: Self::Rhs) -> Self::Output {
        let i = self.parent.apply_lanes(l,r);

        let &[a,b,c,d] = i.as_ref();

        Regs::new([
            self.backend.shr(a,self.width),
            self.backend.shr(b,self.width),
            self.backend.shr(c,self.width),
            self.backend.shr(d,self.width)
        ])
    }

    #[inline(always)]
    fn apply_scalar(&self, l: SL, r: SR) -> i32 {
        let i = self.parent.apply_scalar(l,r);

        i >> self.sw
    }

    #[inline(always)]
    fn compose<F: Fn(Self) -> OP, OP>(self, f: F) -> OP
        where OP: SimdBinaryOperator<BE, SL, SR> {
        f(self)
    }
}
/// A version consisting of multiple operators combined from a type
/// that represents an operator for downcasting data types within SIMD registers
pub struct SimdQueryDemoteOperator<BE,P,I,O,SI,SO>
    where BE: Backend + NativeBackend,
          I: Copy,
          O: Copy,
          SI: Copy,
          SO: Copy {
    backend: BE,
    parent: P,
    input: PhantomData<I>,
    output: PhantomData<O>,
    si: PhantomData<SI>,
    so: PhantomData<SO>,
}
impl<BE,P,I,O,SI,SO> SimdQueryDemoteOperator<BE,P,I,O,SI,SO>
    where BE: Backend + NativeBackend + Clone,
          I: Copy,
          O: Copy,
          SI: Copy,
          SO: Copy {
    #[inline(always)]
    pub fn new(backend: &BE, parent: P) -> SimdQueryDemoteOperator<BE,P,I,O,SI,SO> {
        SimdQueryDemoteOperator {
            backend: backend.clone(),
            parent: parent,
            input: PhantomData::<I>,
            output: PhantomData::<O>,
            si: PhantomData::<SI>,
            so: PhantomData::<SO>,
        }
    }
}
impl<BE,P> SimdPartialOperator<BE>
    for SimdQueryDemoteOperator<BE,P,Regs<<BE as SimdReg<i32>>::Reg,4>,<BE as SimdReg<i8>>::Reg,i32,i8>
    where BE: Backend + NativeBackend +
              SimdReg<i32> +
              SimdReg<i8> +
              SimdDemote<i32,i8,Input=Regs<<BE as SimdReg<i32>>::Reg,4>> {
    #[inline(always)]
    fn compose<F: Fn(Self) -> OP, OP>(self, f: F) -> OP {
        f(self)
    }
}

impl<BE,P,SL,SR> SimdBinaryOperator<BE,SL,SR>
    for SimdQueryDemoteOperator<BE,P,Regs<<BE as SimdReg<i32>>::Reg,4>,Regs<<BE as SimdReg<i8>>::Reg,1>,i32,i8>
    where BE: Backend + NativeBackend +
             SimdReg<i32> +
             SimdReg<i8> +
             SimdDemote<i32,i8,Input=Regs<<BE as SimdReg<i32>>::Reg,4>>,
          SL: Copy,
          SR: Copy,
          P: SimdBinaryOperator<BE,SL,SR,Output=Regs<<BE as SimdReg<i32>>::Reg,4>,ScalarOutput=i32> {
    type Lhs = P::Lhs;
    type Rhs = P::Rhs;
    type Output = Regs<<BE as SimdReg<i8>>::Reg,1>;
    type ScalarOutput = i8;

    #[inline(always)]
    fn apply_lanes(&self, l: Self::Lhs, r: Self::Rhs) -> Self::Output {
        let i = self.parent.apply_lanes(l,r);

        Regs::new([self.backend.demotion(i)])
    }

    #[inline(always)]
    fn apply_scalar(&self, l: SL, r: SR) -> i8 {
        let i = self.parent.apply_scalar(l,r);

        i.assume()
    }

    #[inline(always)]
    fn compose<F: Fn(Self) -> OP, OP>(self, f: F) -> OP
        where OP: SimdBinaryOperator<BE, SL, SR> {
        f(self)
    }
}
impl<S,BE: SimdReg<S> + SimdAdd<S,S,S>> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,1>
    where <BE as SimdReg<S>>::Reg: Copy {
    #[inline(always)]
    fn fold(&self,_: &BE) -> <BE as SimdReg<S>>::Reg {
        self.regs[0]
    }
}
impl<S,BE> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,2>
    where BE: SimdReg<S> +
              SimdAdd<S,S,S,Lhs=<BE as SimdReg<S>>::Reg,Rhs=<BE as SimdReg<S>>::Reg,Output=<BE as SimdReg<S>>::Reg>,
          <BE as SimdReg<S>>::Reg: Copy {
    #[inline(always)]
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg {
        unsafe {
            backend.add(self.regs[0],self.regs[1])
        }
    }
}
impl<S,BE> FoldRegs<S,BE> for Regs<<BE as SimdReg<S>>::Reg,4>
    where BE: SimdReg<S> +
          SimdAdd<S,S,S,Lhs=<BE as SimdReg<S>>::Reg,Rhs=<BE as SimdReg<S>>::Reg,Output=<BE as SimdReg<S>>::Reg>,
          <BE as SimdReg<S>>::Reg: Copy {
    #[inline(always)]
    fn fold(&self,backend: &BE) -> <BE as SimdReg<S>>::Reg {
        unsafe {
            backend.add(backend.add(self.regs[0],self.regs[1]),backend.add(self.regs[2],self.regs[3]))
        }
    }
}
impl IsScaler for i8 {}
impl IsScaler for u8 {}
impl IsScaler for i16 {}
impl IsScaler for u16 {}
impl IsScaler for i32 {}
impl IsScaler for u32 {}
impl IsScaler for i64 {}
impl IsScaler for u64 {}
impl IsScaler for i128 {}
impl IsScaler for u128 {}
impl IsScaler for f32 {}
impl IsScaler for f64 {}
#[inline(always)]
pub const fn mm_shuffle(z: i32, y: i32, x: i32, w: i32) -> i32 {
    (z << 6) | (y << 4) | (x << 2) | w
}
impl<T,BE> SimdPartialLoad<T> for BE
    where Self: SimdLoadAligned<T> + SimdReg<T> + SimdRegBuf {
    #[inline(always)]
    unsafe fn partial_load(&self, input: &[T], rem: usize) -> Self::Reg {
        let tmp = <Self as SimdRegBuf>::initialized(input,rem);

        unsafe { self.load_aligned(tmp.as_ref().as_ptr() as *const u8 as *const T) }
    }
}
impl<T,BE> SimdPartialStore<T> for BE
    where Self: SimdStoreAligned<T> + SimdReg<T> + SimdRegBuf,
          T: Copy {
    #[inline(always)]
    unsafe fn partial_store(&self, output: &mut [T], reg: Self::Reg, rem: usize) {
        let mut tmp = <Self as SimdRegBuf>::uninit();

        unsafe { self.store_aligned(tmp.as_mut().as_mut_ptr() as *mut u8 as *mut T, reg) };

        let src: &[T] = unsafe { std::slice::from_raw_parts(tmp.as_ref().as_ptr() as *const u8 as *const T, rem) };

        output[..rem].copy_from_slice(src);
    }
}
impl<T,BE> SimdLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,1>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> {
    #[inline(always)]
    unsafe fn load_seq(&self, ptr: *const T) -> Regs<<Self as SimdReg<T>>::Reg,1> {
        let reg = unsafe { self.load(ptr) };
        Regs::new([reg])
    }
}
impl<T,BE> SimdLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,2>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> {
    #[inline(always)]
    unsafe fn load_seq(&self, ptr: *const T) -> Regs<<Self as SimdReg<T>>::Reg,2> {
        let a = unsafe { self.load(ptr) };
        let b = unsafe { self.load(ptr.add(<Self as SimdLanes<T>>::LANES)) };
        Regs::new([a,b])
    }
}
impl<T,BE> SimdLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,4>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> {
    #[inline(always)]
    unsafe fn load_seq(&self, ptr: *const T) -> Regs<<Self as SimdReg<T>>::Reg,4> {
        let a = unsafe { self.load(ptr) };
        let b = unsafe { self.load(ptr.add(<Self as SimdLanes<T>>::LANES)) };
        let c = unsafe { self.load(ptr.add(<Self as SimdLanes<T>>::LANES * 2)) };
        let d = unsafe { self.load(ptr.add(<Self as SimdLanes<T>>::LANES * 3)) };
        Regs::new([a,b,c,d])
    }
}
impl<T,BE> SimdPartialLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,1>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> +
              SimdPartialLoad<T>,
          T: Default + Copy {
    #[inline(always)]
    unsafe fn partial_load_seq<const N: usize>(&self, input: &[T]) -> Regs<<Self as SimdReg<T>>::Reg,1> {
        let rem = remaining::<Self,T,N>();

        unsafe {
            Regs::new([self.partial_load(input, rem)])
        }
    }
}
impl<T,BE> SimdPartialLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,2>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> +
              SimdZero<T> +
              SimdPartialLoad<T>,
          T: Default + Copy {
    #[inline(always)]
    unsafe fn partial_load_seq<const N: usize>(&self, input: &[T]) -> Regs<<Self as SimdReg<T>>::Reg,2> {
        let rem = remaining::<Self,T,N>();
        let used_regs = used_regs::<Self,T,2,N>();

        let preg = if rem == 0 {
            <BE as SimdZero<T>>::zero()
        } else {
            let offset = used_regs * <Self as SimdLanes<T>>::LANES;

            unsafe {
                self.partial_load(&input[offset..],rem)
            }
        };

        match used_regs {
            0 => {
                Regs::new([preg, <BE as SimdZero<T>>::zero()])
            },
            1 => {
                unsafe {
                    Regs::new([self.load(input.as_ptr()), preg])
                }
            },
            _ => {
                Regs::new([<BE as SimdZero<T>>::zero(), <BE as SimdZero<T>>::zero()])
            }
        }
    }
}
impl<T,BE> SimdPartialLoadSeq<T,Regs<<Self as SimdReg<T>>::Reg,4>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdLoad<T> +
              SimdZero<T> +
              SimdPartialLoad<T>,
          T: Default + Copy {
    #[inline(always)]
    unsafe fn partial_load_seq<const N: usize>(&self, input: &[T]) -> Regs<<Self as SimdReg<T>>::Reg,4> {
        let rem = remaining::<Self,T,N>();
        let used_regs = used_regs::<Self,T,4,N>();

        let preg = if rem == 0 {
            <BE as SimdZero<T>>::zero()
        } else {
            let offset = used_regs * <Self as SimdLanes<T>>::LANES;

            unsafe {
                self.partial_load(&input[offset..],rem)
            }
        };

        match used_regs {
            0 => {
                Regs::new([
                    preg,
                    <BE as SimdZero<T>>::zero(),
                    <BE as SimdZero<T>>::zero(),
                    <BE as SimdZero<T>>::zero()])
            },
            1 => {
                unsafe {
                    Regs::new([
                        self.load(input.as_ptr()),
                        preg,
                        <BE as SimdZero<T>>::zero(),
                        <BE as SimdZero<T>>::zero()])
                }
            },
            2 => {
                unsafe {
                    Regs::new([
                        self.load(input.as_ptr()),
                        self.load(input[<Self as SimdLanes<T>>::LANES..].as_ptr()),
                        preg,
                        <BE as SimdZero<T>>::zero()])
                }
            },
            3 => {
                unsafe {
                    Regs::new([
                        self.load(input.as_ptr()),
                        self.load(input[<Self as SimdLanes<T>>::LANES..].as_ptr()),
                        self.load(input[(<Self as SimdLanes<T>>::LANES * 2)..].as_ptr()),
                        preg
                    ])
                }
            },
            _ => {
                Regs::new([
                    <BE as SimdZero<T>>::zero(),
                    <BE as SimdZero<T>>::zero(),
                    <BE as SimdZero<T>>::zero(),
                    <BE as SimdZero<T>>::zero()
                ])
            }
        }
    }
}
impl<T,BE> SimdStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,1>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> {
    #[inline(always)]
    unsafe fn store_seq(&self, ptr: *mut T, regs: Regs<<Self as SimdReg<T>>::Reg,1>) {
        let &[reg] = regs.as_ref();

        unsafe {
            self.store(ptr, reg);
        }
    }
}
impl<T,BE> SimdStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,2>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> {
    #[inline(always)]
    unsafe fn store_seq(&self, ptr: *mut T, regs: Regs<<Self as SimdReg<T>>::Reg,2>) {
        let &[a,b] = regs.as_ref();

        unsafe {
            self.store(ptr, a);
            self.store(ptr.add(1 * <Self as SimdLanes<T>>::LANES),b);
        }
    }
}
impl<T,BE> SimdStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,4>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> {
    #[inline(always)]
    unsafe fn store_seq(&self, ptr: *mut T, regs: Regs<<Self as SimdReg<T>>::Reg,4>) {
        let &[a,b,c,d] = regs.as_ref();

        unsafe {
            self.store(ptr, a);
            self.store(ptr.add(1 * <Self as SimdLanes<T>>::LANES),b);
            self.store(ptr.add(2 * <Self as SimdLanes<T>>::LANES), c);
            self.store(ptr.add(3 * <Self as SimdLanes<T>>::LANES),d);
        }
    }
}
impl<T,BE> SimdPartialStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,1>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdPartialStore<T> +
              SimdRegBuf,
          T: Default + Copy  {
    #[inline(always)]
    unsafe fn partial_store_seq<const N: usize>(&self, output: &mut [T], regs: Regs<<Self as SimdReg<T>>::Reg,1>) {
        let rem = remaining::<Self,T,N>();

        let &[reg] = regs.as_ref();

        unsafe {
            self.partial_store(output,reg,rem)
        }
    }
}
impl<T,BE> SimdPartialStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,2>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdZero<T> +
              SimdPartialStore<T> +
              SimdRegBuf,
          T: Default + Copy  {
    #[inline(always)]
    unsafe fn partial_store_seq<const N: usize>(&self, output: &mut [T], regs: Regs<<Self as SimdReg<T>>::Reg,2>) {
        let used_regs = used_regs::<Self,T,2,N>();
        let rem = remaining::<Self,T,N>();

        let &[a,b] = regs.as_ref();

        match used_regs {
            1 => unsafe { self.store(output.as_mut_ptr(),a) },
            _ => ()
        }

        if rem > 0 {
            let r = match used_regs {
                0 => a,
                1 => b,
                _ => <BE as SimdZero<T>>::zero(),
            };

            let offset = used_regs * <Self as SimdLanes<T>>::LANES;

            unsafe {
                self.partial_store(&mut output[offset..],r,rem)
            }
        }
    }
}
impl<T,BE> SimdPartialStoreSeq<T,Regs<<Self as SimdReg<T>>::Reg,4>> for BE
    where BE: Backend +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdReg<T> +
              SimdStore<T> +
              SimdZero<T> +
              SimdPartialStore<T> +
              SimdRegBuf,
          T: Default + Copy  {
    #[inline(always)]
    unsafe fn partial_store_seq<const N: usize>(&self, output: &mut [T], regs: Regs<<Self as SimdReg<T>>::Reg,4>) {
        let used_regs = used_regs::<Self,T,4,N>();
        let rem = remaining::<Self,T,N>();

        let &[a,b,c,d] = regs.as_ref();

        match used_regs {
            1 => {
                unsafe {
                    self.store(output.as_mut_ptr(),a);
                }
            },
            2 => {
                unsafe {
                    self.store(output.as_mut_ptr(),a);
                    self.store(output[<Self as SimdLanes<T>>::LANES..].as_mut_ptr(),b);
                }
            },
            3 => {
                unsafe {
                    self.store(output.as_mut_ptr(), a);
                    self.store(output[<Self as SimdLanes<T>>::LANES..].as_mut_ptr(), b);
                    self.store(output[(<Self as SimdLanes<T>>::LANES * 2)..].as_mut_ptr(), c)
                }
            },
            _ => ()
        }

        if rem > 0 {
            let r = match used_regs {
                0 => a,
                1 => b,
                2 => c,
                3 => d,
                _ => <BE as SimdZero<T>>::zero(),
            };

            let offset = used_regs * <Self as SimdLanes<T>>::LANES;

            unsafe {
                self.partial_store(&mut output[offset..],r,rem)
            }
        }
    }
}
impl<BE> SimdSign<i8> for BE
    where BE: Backend + NativeBackend +
              SimdReg<i8> +
              SimdBitXor<i8> +
              SimdSub<i8,i8,i8> {
    type Input = <Self as SimdReg<i8>>::Reg;
    type Output = <Self as SimdReg<i8>>::Reg;
    fn sign(&self, l: <Self as SimdReg<i8>>::Reg, mask: <Self as SimdReg<i8>>::Reg) -> <Self as SimdReg<i8>>::Reg {
        self.sub(self.bitxor(l,mask),mask)
    }
}
impl<BE> SimdSign<i16> for BE
    where BE: Backend + NativeBackend +
              SimdReg<i16> +
              SimdBitXor<i16> +
              SimdSub<i16,i16,i16> {
    type Input = <Self as SimdReg<i16>>::Reg;
    type Output = <Self as SimdReg<i16>>::Reg;
    fn sign(&self, l: <Self as SimdReg<i16>>::Reg, mask: <Self as SimdReg<i16>>::Reg) -> <Self as SimdReg<i16>>::Reg {
        self.sub(self.bitxor(l,mask),mask)
    }
}
impl<BE> SimdSign<i32> for BE
    where BE: Backend + NativeBackend +
              SimdReg<i32> +
              SimdBitXor<i32> +
              SimdSub<i32,i32,i32> {
    type Input = <Self as SimdReg<i32>>::Reg;
    type Output = <Self as SimdReg<i32>>::Reg;
    fn sign(&self, l: <Self as SimdReg<i32>>::Reg, mask: <Self as SimdReg<i32>>::Reg) -> <Self as SimdReg<i32>>::Reg {
        self.sub(self.bitxor(l,mask),mask)
    }
}
impl<BE> SimdSign<i64> for BE
    where BE: Backend + NativeBackend +
              SimdReg<i64> +
              SimdBitXor<i64> +
              SimdSub<i64,i64,i64> {
    type Input = <Self as SimdReg<i64>>::Reg;
    type Output = <Self as SimdReg<i64>>::Reg;
    fn sign(&self, l: <Self as SimdReg<i64>>::Reg, mask: <Self as SimdReg<i64>>::Reg) -> <Self as SimdReg<i64>>::Reg {
        self.sub(self.bitxor(l,mask),mask)
    }
}
impl<BE> SimdSign<f32> for BE
    where BE: Backend + NativeBackend +
              SimdReg<f32> +
              SimdBitXor<f32> +
              SimdSub<f32,f32,f32>,
          f32: BitsBitXor<Bits=u32> {
    type Input = <Self as SimdReg<f32>>::Reg;
    type Output = <Self as SimdReg<f32>>::Reg;
    fn sign(&self, l: <Self as SimdReg<f32>>::Reg, mask: <Self as SimdReg<u32>>::Reg) -> <Self as SimdReg<f32>>::Reg {
        self.bitxor(l,mask)
    }
}
impl<BE> SimdSign<f64> for BE
    where BE: Backend + NativeBackend +
              SimdReg<f64> +
              SimdBitXor<f64> +
              SimdSub<f64,f64,f64>,
              f64: BitsBitXor<Bits=u64> {
    type Input = <Self as SimdReg<f64>>::Reg;
    type Output = <Self as SimdReg<f64>>::Reg;
    fn sign(&self, l: <Self as SimdReg<f64>>::Reg, mask: <Self as SimdReg<u64>>::Reg) -> <Self as SimdReg<f64>>::Reg {
        self.bitxor(l,mask)
    }
}
impl<T,BE,OP> SimdQueryVector<T,OP> for BE
    where BE: Backend + NativeBackend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdLoad<T> +
              SimdStore<T>,
          T: Default + Copy,
          OP: SimdBinaryOperator<BE,T,T,Lhs=<BE as SimdReg<T>>::Reg, Rhs=<BE as SimdReg<T>>::Reg, Output=<BE as SimdReg<T>>::Reg, ScalarOutput = T> {
    #[target_feature(enable = "avx2")]
    unsafe fn query_vector<'a,'b,const N: usize>(&self, l: VectorView<'a,T,N>, r: VectorView<'b,T,N>,operator: &OP) -> OwnedVector<T,N> {
        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        unsafe {
            let ref_l = l.as_ref();
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_l = ref_l.chunks_exact(<Self as SimdLanes<T>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);

            for (co,(cl,cr)) in chunks_o.zip(chunks_l.zip(chunks_r)) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let rr = operator.apply_lanes(ra,rb);

                self.store(co.as_mut_ptr(),rr);
            }

            let chuks_l_remainder = ref_l.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();
            let chuks_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();
            let chuks_o_remainder = ref_o.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for (o,(l,r)) in chuks_o_remainder.iter_mut().zip(chuks_l_remainder.iter().zip(chuks_r_remainder.iter())) {
                    *o = operator.apply_scalar(*l,*r);
                }
            }

            rs
        }
    }
}
impl<BE,OP,SL,SR,SO> SimdCombinationQueryVectorKernel<OP,SL,SR,SO> for BE
    where BE: Backend + NativeBackend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLoadSeq<SL,<OP as SimdBinaryOperator<Self,SL,SR>>::Lhs> +
              SimdLoadSeq<SR,<OP as SimdBinaryOperator<Self,SL,SR>>::Rhs> +
              SimdStoreSeq<SO,<OP as SimdBinaryOperator<Self,SL,SR>>::Output> +
              SimdPartialLoadSeq<SL,<OP as SimdBinaryOperator<Self,SL,SR>>::Lhs> +
              SimdPartialLoadSeq<SR,<OP as SimdBinaryOperator<Self,SL,SR>>::Rhs> +
              SimdPartialStoreSeq<SO,<OP as SimdBinaryOperator<Self,SL,SR>>::Output>,
          OP: SimdBinaryOperator<Self,SL,SR, ScalarOutput = SO>,
          SL: Copy,
          SR: Copy,
          SO: Default + Copy {
    #[inline(always)]
    fn combination_query_vector<'a, 'b, const N: usize, const LANES: usize>(
        &self, l: VectorView<'a, SL, N>, r: VectorView<'b, SR, N>, operator: OP
    ) -> OwnedVector<SO, N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let ref_l = l.as_ref();
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_l = ref_l.chunks_exact(LANES);
            let chunks_r = ref_r.chunks_exact(LANES);
            let chunks_o = ref_o.chunks_exact_mut(LANES);

            for (co,(cl,cr)) in chunks_o.zip(chunks_l.zip(chunks_r)) {
                let ra = self.load_seq(cl.as_ptr());
                let rb = self.load_seq(cr.as_ptr());

                let rr = operator.apply_lanes(ra,rb);

                self.store_seq(co.as_mut_ptr(),rr);
            }

            let chuks_l_remainder = ref_l.chunks_exact(LANES).remainder();
            let chuks_r_remainder = ref_r.chunks_exact(LANES).remainder();
            let chuks_o_remainder = ref_o.chunks_exact_mut(LANES).into_remainder();

            if N % LANES != 0 {
                let ra = self.partial_load_seq::<N>(chuks_l_remainder);
                let rb = self.partial_load_seq::<N>(chuks_r_remainder);

                let rr = operator.apply_lanes(ra,rb);

                self.partial_store_seq::<N>(chuks_o_remainder,rr);
            }

            rs
        }
    }
}
impl<T,BE> SimdAddAssignVector<T,T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdAdd<T,T,T,Lhs=<Self as SimdReg<T>>::Reg,Rhs=<Self as SimdReg<T>>::Reg,Output=<Self as SimdReg<T>>::Reg> +
              SimdLoad<T> +
              SimdStore<T>,
          T: Default + Add<Output=T> + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn add_assign_vector<'a,'b,const N: usize>(&self, l: VectorMutView<'a,T,N>, r: VectorView<'b,T,N>) {
        unsafe {
            let mut l = l;
            let ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let chunks_l = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES);

            for (cl,cr) in chunks_l.zip(chunks_r) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let rr = self.add(ra,rb);

                self.store(cl.as_mut_ptr(),rr);
            }

            let chuks_l_remainder = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();
            let chuks_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for (l,r) in chuks_l_remainder.iter_mut().zip(chuks_r_remainder.iter()) {
                    *l = *l + *r;
                }
            }
        }
    }
}
/*
impl<T,BE> SimdAddVector<T,T,T> for BE
    where BE: Backend +
              SimdAddAssignVector<T,T>,
              T: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn add_vector<'a,'b,const N: usize>(&self, l: VectorView<'a,T,N>, r: VectorView<'b,T,N>)
        -> OwnedVector<T,N> {
        let mut acc = OwnedVector::from(Box::<[T;N]>::from(&l));

        let l = VectorMutView::<T,N>::from(&mut acc);

        unsafe { <Self as SimdAddAssignVector<T,T>>::add_assign_vector(self,l,r) };

        acc
    }
}

 */
impl<T,BE> SimdSubAssignVector<T,T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdSub<T,T,T> +
              SimdLoad<T> +
              SimdStore<T>,
          T: Default + Sub<Output=T> + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn sub_assign_vector<'a,'b,const N: usize>(&self, l: VectorMutView<'a,T,N>, r: VectorView<'b,T,N>) {
        unsafe {
            let mut l = l;
            let ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let chunks_l = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES);

            for (cl,cr) in chunks_l.zip(chunks_r) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let rr = self.sub(ra,rb);

                self.store(cl.as_mut_ptr(),rr);
            }

            let chuks_l_remainder = ref_l.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();
            let chuks_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for (l,r) in chuks_l_remainder.iter_mut().zip(chuks_r_remainder.iter()) {
                    *l = *l - *r;
                }
            }
        }
    }
}
impl<T,BE> SimdSubVector<T,T,T> for BE
    where BE: Backend +
              SimdSubAssignVector<T,T>,
      T: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn sub_vector<'a,'b,const N: usize>(&self, l: VectorView<'a,T,N>, r: VectorView<'b,T,N>)
        -> OwnedVector<T,N> {
        let mut acc = OwnedVector::from(Box::<[T;N]>::from(&l));

        let l = VectorMutView::<T,N>::from(&mut acc);

        unsafe { <Self as SimdSubAssignVector<T,T>>::sub_assign_vector(self,l,r) };

        acc
    }
}
impl<SL,SR,SO,BE> SimdMulVector<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdRows<SL> +
              SimdMul<SL,SR,SO> +
              SimdLoad<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
              SO: Default + From<SL> + From<SR> + Mul<SO,Output=SO> + Copy,
              SL: Copy,
              SR: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Heterogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn mul_vector<'a,const N: usize>(&self, l: VectorView<'a,SL,N>, r: VectorView<'a,SR,N>)
        -> OwnedVector<SO,N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let ref_l = l.as_ref();
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_l = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);

            for (co,(cl,cr)) in chunks_o.zip(chunks_l.zip(chunks_r)) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let prod = self.mul(ra,rb);

                self.store_seq(co.as_mut_ptr(),prod);
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunck_l_remainder = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunck_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunck_o_remainder = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,(&l,&r)) in chunck_o_remainder.iter_mut().zip(chunck_l_remainder.iter().zip(chunck_r_remainder.iter())) {
                    *o = SO::from(l) * SO::from(r);
                }
            }
        }

        rs
    }
}
impl<SL,SR,BE> SimdMulAssignVector<SL,SR> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdLanes<SL> +
              SimdRows<SL> +
              SimdMul<SL,SR,SL> +
              SimdLoad<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SL,<Self as SimdMul<SL,SR,SL>>::Output>,
          SL: Mul<SR,Output=SL> + Copy,
          SR: Copy,
          <Self as SimdMul<SL,SR,SL>>::Output: Copy,
          (SL,SL): SupportMul<Homogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn mul_assign_vector<'a,'b,const N: usize>(&self, l: VectorMutView<'a,SL,N>, r: VectorView<'b,SR,N>) {
        unsafe {
            let mut l = l;
            let ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let chunks_l = ref_l.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES);

            for (cl,cr) in chunks_l.zip(chunks_r) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let prod = self.mul(ra,rb);

                self.store_seq(cl.as_mut_ptr(),prod);
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunck_l_remainder = ref_l.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();
                let chunck_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();

                for (l,r) in chunck_l_remainder.iter_mut().zip(chunck_r_remainder.iter()) {
                    *l = *l * *r;
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdMulVector<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdRows<SL> +
              SimdMul<SL,SR,SO> +
              SimdLoad<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
              SO: Default + Copy,
              SL: Mul<SR,Output=SO> + Copy,
              SR: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Homogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn mul_vector<'a,'b,const N: usize>(&self, l: VectorView<'a,SL,N>, r: VectorView<'b,SR,N>)
        -> OwnedVector<SO,N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let ref_l = l.as_ref();
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_l = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);

            for (co,(cl,cr)) in chunks_o.zip(chunks_l.zip(chunks_r)) {
                let ra = self.load(cl.as_ptr());
                let rb = self.load(cr.as_ptr());

                let prod = self.mul(ra,rb);

                self.store_seq(co.as_mut_ptr(),prod);
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunck_l_remainder = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunck_r_remainder = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunck_o_remainder = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,(&l,&r)) in chunck_o_remainder.iter_mut().zip(chunck_l_remainder.iter().zip(chunck_r_remainder.iter())) {
                    *o = l * r;
                }
            }
        }

        rs
    }
}
impl<SL,SR,SO,BE> SimdScalarMulVector<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SO> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output> +
              SimdStore<SO>,
              SO: Default + From<SL> + From<SR> + Mul<SO,Output=SO> + Copy,
              SL: Copy,
              SR: Copy,
              <Self as SimdReg<SL>>::Reg: Copy,
              <Self as SimdMul<SL,SR,SO>>::Output: Copy,
              (SL,SO): SupportMul<Heterogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector<'a,'b,const N: usize>(&self, l:SL, r: VectorView<'b, SR, N>) -> OwnedVector<SO, N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let s = self.splat(l);
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            for (co,cr) in chunks_o.zip(chunks_r) {
                for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (o,r) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(r.as_ptr());

                    let rs = self.mul(s,rr);

                    self.store_seq(o.as_mut_ptr(),rs);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,r) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *o = SO::from(l) * SO::from(*r);
                }
            }
        }

        rs
    }
}
impl<SL,SR,BE> SimdScalarMulAssignVector<SL,SR> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SR> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SR,<Self as SimdMul<SL,SR,SR>>::Output>,
      SL: Copy,
      SR: From<SL> + Mul<Output=SR> + Copy,
      SR: Copy,
      <Self as SimdReg<SR>>::Reg: Copy,
      <Self as SimdMul<SL,SR,SR>>::Output: Copy,
      (SL,SR): SupportMul<Homogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_assign_vector<'a,'b,const N: usize>(&self, l:SL, r: VectorMutView<'b, SR, N>) {
        unsafe {
            let mut r = r;
            let s = self.splat(l);
            let ref_r = r.as_mut();

            let chunks_r = ref_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            for cr in chunks_r {
                for cr in cr.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES) {
                    let rr = self.load(cr.as_ptr());

                    let rs = self.mul(s,rr);

                    self.store_seq(cr.as_mut_ptr(),rs);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for cr in chunks_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(cr.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for r in chunks_r.iter_mut() {
                    *r = SR::from(l) * *r;
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdScalarMulVector<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SO> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
          SO: Default + Copy,
          SL: Mul<SR,Output=SO> + Copy,
          SR: Copy,
          <Self as SimdReg<SL>>::Reg: Copy,
          <Self as SimdMul<SL,SR,SO>>::Output: Copy,
          (SL,SO): SupportMul<Homogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector<'a, const N: usize>(&self, l:SL, r: VectorView<'a, SR, N>) -> OwnedVector<SO, N> {
        let mut rs = OwnedVector::from(Box::new([SO::default(); N]));

        unsafe {
            let ref_r = r.as_ref();
            let ref_o = rs.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            let s = self.splat(l);

            for (co,cr) in chunks_o.zip(chunks_r) {
                for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (co,cr) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (co,cr) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *co = l * *cr;
                }
            }
        }

        rs
    }
}
impl<SL,SR,SO,BE> SimdScalarMulVectorInto<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SO> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output> +
              SimdStore<SO>,
          SO: Default + From<SL> + From<SR> + Mul<SO,Output=SO> + Copy,
          SL: Copy,
          SR: Copy,
          <Self as SimdReg<SL>>::Reg: Copy,
          <Self as SimdMul<SL,SR,SO>>::Output: Copy,
          (SL,SO): SupportMul<Heterogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector_into<'a, const N: usize>(&self, l:SL, r: &VectorView<'a, SR, N>, o: VectorMutView<'a, SO, N>) {
        unsafe {
            let s = self.splat(l);
            let ref_r = r.as_ref();

            let mut o = o;
            let ref_o = o.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            for (co,cr) in chunks_o.zip(chunks_r) {
                for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (co,cr) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,r) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *o = SO::from(l) * SO::from(*r);
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdScalarMulVectorInto<SL,SR,SO> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SO> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
          SO: Default + Copy,
          SL: Mul<SR,Output=SO> + Copy,
          SR: Copy,
          <Self as SimdReg<SL>>::Reg: Copy,
          <Self as SimdMul<SL,SR,SO>>::Output: Copy,
          (SL,SO): SupportMul<Homogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalarmul_vector_into<'a, const N: usize>(&self, l:SL, r: &VectorView<'a, SR, N>, o: VectorMutView<'a, SO, N>) {
        unsafe {
            let s = self.splat(l);
            let ref_r = r.as_ref();

            let mut o = o;
            let ref_o = o.as_mut();

            let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS);

            for (co,cr) in chunks_o.zip(chunks_r) {
                 for (co,cr) in co.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(cr.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS) != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES * <Self as SimdCols<SL>>::COLS).into_remainder();

                for (co,cr) in chunks_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).zip(chunks_r.chunks_exact(<Self as SimdLanes<SL>>::LANES)) {
                    let rr = self.load(cr.as_ptr());

                    let o = self.mul(s,rr);

                    self.store_seq(co.as_mut_ptr(),o);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                for (o,r) in chunks_o.iter_mut().zip(chunks_r.iter()) {
                    *o = l * *r;
                }
            }
        }
    }
}
impl<T,BE> SimdBitAndVector<T> for BE
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitAnd>::Bits> +
              SimdReg<<T as BitsBitAnd>::Bits> +
              SimdLoad<T> +
              SimdLoad<<T as BitsBitAnd>::Bits> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitAnd<T>,
              T: BitsBitAnd + Default + Copy,
              <T as BitsBitAnd>::Bits: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn bitand_vector<'a,const N: usize>(&self, l: VectorView<'a,T,N>, r: VectorView<'a,<T as BitsBitAnd>::Bits,N>)
        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.bitand(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitand(r[j]);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdBitOrVector<T> for BE
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitOr>::Bits> +
              SimdReg<<T as BitsBitOr>::Bits,Reg=<Self as SimdReg<<T as BitsBitOr>::Bits>>::Mask> +
              SimdLoad<T> +
              SimdLoad<<Self as SimdReg<T>>::Bits> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitOr<T>,
              T: BitsBitOr + Default + Copy,
              <T as BitsBitOr>::Bits: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn bitor_vector<'a,const N: usize>(&self, l: VectorView<'a,T,N>, r: VectorView<'a,<T as BitsBitOr>::Bits,N>)
                                        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.bitor(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitor(r[j]);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdBitXorVector<T> for BE
    where BE: Backend +
              SimdReg<T,Bits=<T as BitsBitXor>::Bits> +
              SimdReg<<T as BitsBitXor>::Bits,Reg=<Self as SimdReg<<T as BitsBitXor>::Bits>>::Mask> +
              SimdLoad<T> +
              SimdLoad<<Self as SimdReg<T>>::Bits> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitXor<T>,
              T: BitsBitXor + Default + Copy,
              <T as BitsBitXor>::Bits: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn bitxor_vector<'a,const N: usize>(&self, l: VectorView<'a,T,N>, r: VectorView<'a,<T as BitsBitXor>::Bits,N>)
        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        unsafe {
            let pa = l.as_ref().as_ptr();
            let pb = r.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));
                let rb = self.load(pb.add(i));

                let rr = self.bitxor(ra,rb);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitxor(r[j]);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdBitNotVector<T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLoad<T> +
              SimdStore<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdBitNot<T>,
              T: BitsBitNot + Default + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn bitnot_vector<'a,const N: usize>(&self, l: VectorView<'a,T,N>)
        -> OwnedVector<T,N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        unsafe {
            let pa = l.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<T>>::LANES <= N {
                let ra = self.load(pa.add(i));

                let rr = self.bitnot(ra);

                self.store(po.add(i),rr);

                i += <Self as SimdLanes<T>>::LANES;
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                for j in i..N {
                    rs[j] = l[j].bits_bitnot();
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdShlVector<T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLoad<T> +
              SimdStore<T> +
              SimdShl<T> +
              SimdLanes<T> +
              SimdShiftWidth<T>,
              T: BitsShl + Default + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn shl_vector<'a,const N: usize>(&self, v: VectorView<'a,T,N>, w:usize)
        -> OwnedVector<T,N> {
        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        let rw = <BE as SimdShiftWidth<T>>::shift_width(self,w);

        unsafe {
            let ref_v = v.as_ref();
            let ref_o = rs.as_mut();

            let chunks_v = ref_v.chunks_exact(<Self as SimdLanes<T>>::LANES);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);

            for (o,v) in chunks_o.zip(chunks_v) {
                let vr = self.load(v.as_ptr());
                let rr = self.shl(vr,rw);

                self.store(o.as_mut_ptr(),rr);
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                let chunk_v = ref_v.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();
                let chunk_o = ref_o.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();

                for (o,v) in chunk_o.iter_mut().zip(chunk_v.iter()) {
                    *o = v.bits_shl(w);
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdShrVector<T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLoad<T> +
              SimdStore<T> +
              SimdShr<T> +
              SimdLanes<T> +
              SimdShiftWidth<T>,
              T: BitsShr + Default + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn shr_vector<'a,const N: usize>(&self, v: VectorView<'a,T,N>, w:usize)
        -> OwnedVector<T,N> {
        let mut rs = OwnedVector::from(Box::new([T::default(); N]));

        let rw = <BE as SimdShiftWidth<T>>::shift_width(self,w);

        unsafe {
            let ref_v = v.as_ref();
            let ref_o = rs.as_mut();

            let chunks_v = ref_v.chunks_exact(<Self as SimdLanes<T>>::LANES);
            let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);

            for (o,v) in chunks_o.zip(chunks_v) {
                let vr = self.load(v.as_ptr());
                let rr = self.shr(vr,rw);

                self.store(o.as_mut_ptr(),rr);
            }

            if N % <Self as SimdLanes<T>>::LANES != 0 {
                let chunk_v = ref_v.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();
                let chunk_o = ref_o.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();

                for (o,v) in chunk_o.iter_mut().zip(chunk_v.iter()) {
                    *o = v.bits_shr(w);
                }
            }
        }

        rs
    }
}
impl<SS,SD,BE> SimdPromoteVector<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SS> +
              SimdLanes<SS> +
              SimdPromote<SS,SD> +
              SimdLoad<SS> +
              SimdStoreSeq<SD,<Self as SimdPromote<SS,SD>>::Output> +
              SimdStore<SD>,
          SS: Copy,
          SD: Default + Copy + From<SS>,
          <Self as SimdPromote<SS,SD>>::Output: Copy,
          <Self as SimdReg<SS>>::Reg: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn promotion_vector<'a, const N: usize>(&self, s: VectorView<'a, SS, N>) -> OwnedVector<SD, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SD::default(); N]));

        unsafe {
            let pb = s.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SS>>::LANES <= N {
                let sr = self.load(pb.add(i));

                let o = self.promotion(sr);

                self.store_seq(po.add(i),o);

                i += <Self as SimdLanes<SS>>::LANES;
            }

            if N % <Self as SimdLanes<SS>>::LANES != 0 {
                for j in i..N {
                    rs[j] = s[j].into();
                }
            }
        }

        rs
    }
}
impl<SS,SD,BE> SimdDemoteVector<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SD> +
              SimdLanes<SD> +
              SimdDemote<SS,SD> +
              SimdLoadSeq<SS,<Self as SimdDemote<SS,SD>>::Input> +
              SimdStore<SD>,
          SS: Assume<SD> + Copy,
          SD: Default + Copy,
          <Self as SimdDemote<SS,SD>>::Input: Copy,
          <Self as SimdReg<SD>>::Reg: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn demotion_vector<'a, const N: usize>(&self, s: VectorView<'a, SS, N>) -> OwnedVector<SD, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SD::default(); N]));

        unsafe {
            let pb = s.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SD>>::LANES <= N {
                let sr = self.load_seq(pb.add(i));

                let o = self.demotion(sr);

                self.store(po.add(i),o);

                i += <Self as SimdLanes<SD>>::LANES;
            }

            if N % <Self as SimdLanes<SD>>::LANES != 0 {
                for j in i..N {
                    rs[j] = s[j].assume();
                }
            }
        }

        rs
    }
}
impl<SS,SD,BE> SimdConvertVector<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SS> +
              SimdLanes<SS> +
              SimdConvert<SS,SD> +
              SimdLoadSeq<SS,<Self as SimdConvert<SS,SD>>::Input> +
              SimdStoreSeq<SD,<Self as SimdConvert<SS,SD>>::Output> +
              SimdStore<SD>,
          SS: Assume<SD> + Copy,
          SD: Default + Copy,
          <Self as SimdConvert<SS,SD>>::Output: Copy,
          <Self as SimdReg<SS>>::Reg: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn convert_vector<'a, const N: usize>(&self, s: VectorView<'a, SS, N>) -> OwnedVector<SD, N> {
        let mut i = 0;

        let mut rs = OwnedVector::from(Box::new([SD::default(); N]));

        unsafe {
            let pb = s.as_ref().as_ptr();
            let po = rs.as_mut().as_mut_ptr();

            while i + <Self as SimdLanes<SS>>::LANES <= N {
                let sr = self.load_seq(pb.add(i));

                let o = self.convert(sr);

                self.store_seq(po.add(i),o);

                i += <Self as SimdLanes<SS>>::LANES;
            }

            if N % <Self as SimdLanes<SS>>::LANES != 0 {
                for j in i..N {
                    rs[j] = s[j].assume();
                }
            }
        }

        rs
    }
}
impl<T,BE> SimdAddAssignMatrix<T,T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdAdd<T,T,T,Lhs=<Self as SimdReg<T>>::Reg,Rhs=<Self as SimdReg<T>>::Reg,Output=<Self as SimdReg<T>>::Reg> +
              SimdLoad<T> +
              SimdStore<T>,
          T: Default + Add<Output=T> + Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn add_assign_matrix<'a,'b,const N: usize,const M: usize>(&self, l: MatrixMutView<'a,T,N,M>, r: MatrixView<'b,T,N,M>) {
        unsafe {
            let mut l = l;
            let ref_l = l.as_mut();
            let ref_r = r.as_ref();

            let chunks_l = ref_l.chunks_exact_mut(M);
            let chunks_r = ref_r.chunks_exact(M);

            for (cl,cr) in chunks_l.zip(chunks_r) {
                let chunks_l = cl.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);
                let chunks_r = cr.chunks_exact(<Self as SimdLanes<T>>::LANES);

                for (cl,cr) in chunks_l.zip(chunks_r) {
                    let ra = self.load(cl.as_ptr());
                    let rb = self.load(cr.as_ptr());

                    let rr = self.add(ra,rb);

                    self.store(cl.as_mut_ptr(),rr);
                }

                if M % <Self as SimdLanes<T>>::LANES != 0 {
                    let chunks_l = cl.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();
                    let chunks_r = cr.chunks_exact(<Self as SimdLanes<T>>::LANES).remainder();

                    for (l,r) in chunks_l.iter_mut().zip(chunks_r.iter()) {
                        *l = *l + *r;
                    }
                }
            }
        }
    }
}
impl<T,BE> SimdScalarMulAssignMatrix<T,T> for BE
    where BE: Backend +
              SimdReg<T> +
              SimdLanes<T> +
              SimdRows<T> +
              SimdMask<T> +
              SimdMul<T,T,T> +
              SimdSplat<T> +
              SimdLoad<T> +
              SimdStoreSeq<T,<Self as SimdMul<T,T,T>>::Output>,
          T: Default + Mul<Output=T> + Copy,
          <Self as SimdMul<T,T,T>>::Output: Copy,
          (T,T): SupportMul<Homogeneous> {
    #[target_feature(enable = "avx2")]
    unsafe fn scalar_mul_assign_matrix<'a,const N: usize,const M: usize>(&self, l: T, r: MatrixMutView<'a,T,N,M>) {
        unsafe {
            let rl = <Self as SimdSplat<T>>::splat(self,l);

            let mut r = r;
            let r_ref = r.as_mut();

            let chunks_r = r_ref.chunks_exact_mut(M);

            for cr in chunks_r {
                let chunks_r = cr.chunks_exact_mut(<Self as SimdLanes<T>>::LANES);

                for cr in chunks_r {
                    let rb = self.load(cr.as_ptr());

                    let rr = self.mul(rl,rb);

                    self.store_seq(cr.as_mut_ptr(),rr);
                }

                if M % <Self as SimdLanes<T>>::LANES != 0 {
                    let chunks_r = cr.chunks_exact_mut(<Self as SimdLanes<T>>::LANES).into_remainder();

                    for r in chunks_r {
                        *r = l * *r;
                    }
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdScalarMulMatrix<SL,SR> for BE
    where BE: Backend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdReg<SO> +
              SimdLanes<SL> +
              SimdCols<SL> +
              SimdMul<SL,SR,SO> +
              SimdSplat<SL> +
              SimdLoad<SR> +
              SimdLoad<SO> +
              SimdStoreSeq<SO,<Self as SimdMul<SL,SR,SO>>::Output>,
          SO: Default + Copy,
          SL: Mul<SR,Output=SO> + Copy,
          SR: Copy,
          <Self as SimdReg<SL>>::Reg: Copy,
          <Self as SimdMul<SL,SR,SO>>::Output: Copy,
          (SL,SO): SupportMul<Homogeneous> {
    type OutputScalar = SO;
    #[target_feature(enable = "avx2")]
    unsafe fn scalar_mul_matrix<'a,const N: usize,const M: usize>(&self, l: SL, r: MatrixView<'a,SR,N,M>, acc:MatrixMutView<'a,SO,N,M>) {
        unsafe {
            let rl = <Self as SimdSplat<SL>>::splat(self,l);

            let r_ref = r.as_ref();

            let mut acc = acc;
            let ref_acc = acc.as_mut();

            let chunks_r = r_ref.chunks_exact(M);
            let acc_chunks_r = ref_acc.chunks_exact_mut(M);

            for (cr,acc_cr) in chunks_r.zip(acc_chunks_r) {
                let chunks_cr = cr.chunks_exact(<Self as SimdLanes<SL>>::LANES);
                let acc_chunks_cr = acc_cr.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES);

                for (cr,acc_cr) in chunks_cr.zip(acc_chunks_cr) {
                    let rb = self.load(cr.as_ptr());

                    let rr = self.mul(rl,rb);

                    self.store_seq(acc_cr.as_mut_ptr(),rr);
                }

                if M % <Self as SimdLanes<SL>>::LANES != 0 {
                    let chunks_r = cr.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                    let chunks_acc = acc_cr.chunks_exact_mut(<Self as SimdLanes<SL>>::LANES).into_remainder();

                    for (acc,r) in chunks_acc.iter_mut().zip(chunks_r.iter()) {
                        *acc = l * *r;
                    }
                }
            }
        }
    }
}
impl<SS,SD,BE> SimdConvertMatrix<SS,SD> for BE
    where BE: Backend +
              SimdReg<SS> +
              SimdReg<SS> +
              SimdLanes<SS> +
              SimdConvert<SS,SD> +
              SimdLoadSeq<SS,<Self as SimdConvert<SS,SD>>::Input> +
              SimdStoreSeq<SD,<Self as SimdConvert<SS,SD>>::Output> +
              SimdStore<SD>,
          SS: Assume<SD> + Copy,
          SD: Default + Copy,
          <Self as SimdConvert<SS,SD>>::Output: Copy,
          <Self as SimdReg<SS>>::Reg: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn convert_matrix<'a, const N: usize,const M: usize>(&self, s: MatrixView<'a, SS, N, M>, acc:MatrixMutView<'a,SD,N,M>) {
        unsafe {
            let mut acc = acc;

            for i in 0..N {
                let lr = s.row(i);
                let pa = lr.as_ref().as_ptr();

                let po = acc.as_mut().as_mut_ptr().add(i * M);

                let mut j = 0;

                while j + <Self as SimdLanes<SS>>::LANES <= M {
                    let sr = self.load_seq(pa.add(j));

                    let o = self.convert(sr);

                    self.store_seq(po.add(j),o);

                    j += <Self as SimdLanes<SS>>::LANES;
                }

                if M % <Self as SimdLanes<SS>>::LANES != 0 {
                    for k in j..M {
                        acc[(i,k)] = lr[k].assume();
                    }
                }
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdDotKernel<SL,SR,SO> for BE
    where Self: SimdReg<SL> + SimdReg<SR> + SimdReg<SO> + SimdAdd<SO,SO,SO> + SimdHSum<SO> +
                SimdLoad<SL> + SimdLoad<SR> +
                SimdPartialLoad<SL> + SimdPartialLoad<SR> + SimdZero<SO> +
                SimdLanes<SL> + SimdLanes<SR> + SimdLanes<SO> +
                SimdPartialDot<SL,SR,SO>,
          SL: Clone + Copy,
          SR: Clone + Copy,
          SO: From<SL> + From<SR> + Default + Mul<SO,Output=SO> + AddAssign,
          <Self as SimdPartialDot<SL,SR,SO>>::Output: Copy {
    #[inline(always)]
    unsafe fn dot<'a, 'b, const N: usize, const COLS: usize>(&self, l: &VectorView<'a, SL, N>, r: VectorView<'b, SR, N>) -> SO {
        let mut acc = [<Self as SimdPartialDot<SL,SR,SO>>::zero_acc(self);COLS];

        let ref_l = l.as_ref();
        let ref_r = r.as_ref();

        let l_chuncks = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES * COLS);
        let r_chuncks = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * COLS);

        unsafe {
            for (lc, rc) in l_chuncks.zip(r_chuncks) {
                let lc = lc.chunks_exact(<Self as SimdLanes<SL>>::LANES);
                let rc = rc.chunks_exact(<Self as SimdLanes<SL>>::LANES);

                for ((lc, rc), acc) in lc.zip(rc).zip(acc.iter_mut()) {
                    let lr = self.load(lc.as_ptr());
                    let rr = self.load(rc.as_ptr());

                    *acc = self.partial_dot(lr, rr, *acc);
                }
            }

            if N % (<Self as SimdLanes<SL>>::LANES * COLS) != 0 {
                let lc = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES * COLS).remainder().chunks_exact(<Self as SimdLanes<SL>>::LANES);
                let rc = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES * COLS).remainder().chunks_exact(<Self as SimdLanes<SL>>::LANES);

                for ((lc, rc), acc) in lc.zip(rc).zip(acc.iter_mut()) {
                    let lr = self.load(lc.as_ptr());
                    let rr = self.load(rc.as_ptr());

                    *acc = self.partial_dot(lr, rr, *acc);
                }
            }

            if N % <Self as SimdLanes<SL>>::LANES != 0 {
                let lc = ref_l.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();
                let rc = ref_r.chunks_exact(<Self as SimdLanes<SL>>::LANES).remainder();

                let lr = self.partial_load(lc, N % <Self as SimdLanes<SL>>::LANES);
                let rr = self.partial_load(rc, N % <Self as SimdLanes<SL>>::LANES);

                acc[0] = self.partial_dot(lr, rr, acc[0]);
            }

            let mut sum = SO::default();

            for acc in acc.iter() {
                sum += <Self as SimdHSum<SO>>::hsum(self, acc.fold(self));
            }

            sum
        }
    }
}
impl<SL, SR, SO,BE> SimdOuterProduct<SL,SR,SO> for BE
    where BE: Backend +
              SimdScalarMulVectorInto<SL,SR,SO>,
          SL: Copy {
    #[target_feature(enable = "avx2")]
    unsafe fn outer_product<'a, const N: usize, const M: usize>(&self, l: VectorView<'a, SL, N>,
                                                         r: VectorView<'a, SR, M>,
                                                         o: MatrixMutView<'a, SO, N, M>) {
        let mut o = o;

        unsafe {
            for (row,&l) in o.as_mut().chunks_exact_mut(M).zip(l.as_ref().iter()) {
                <Self as SimdScalarMulVectorInto<SL,SR,SO>>::scalarmul_vector_into(self,l,&r,row.try_into().unwrap());
            }
        }
    }
}
impl<SL,SR,SO,BE> SimdMatMulKernel<SL,SR,SO> for BE
    where Self: SimdPartialDot<SL,SR,SO> +
                SimdLanes<SL> + SimdRows<SL> + SimdZero<SL> + SimdReg<SL> + SimdLoad<SL> +
                SimdCols<SR> + SimdZero<SR> + SimdReg<SR> + SimdLoad<SR> +
                SimdZero<SO> + SimdHSum<SO> +
                SimdLanes<SR>,
          <Self as SimdReg<SL>>::Reg: Clone + Copy,
          <Self as SimdReg<SR>>::Reg: Clone + Copy,
          <Self as SimdReg<SO>>::Reg: Clone + Copy,
          SL: Copy,
          SR: Copy,
          SO: Default + AddAssign + Mul<SO,Output=SO> + From<SL> + From<SR> + Copy,
          <BE as SimdPartialDot<SL,SR,SO>>::Output: Copy {
    #[inline(always)]
    unsafe fn matmul<'a, const N: usize, const M: usize, const K: usize,const ROWS: usize,const COLS: usize>(
        &self, l: MatrixView<'a, SL, N, K>, r: ColumnMajorMatrix<'a, SR, K, M>, acc: MatrixMutView<'a,SO, N, M>) {

        let mut acc = acc;

        for i in (0..(N - N % ROWS)).step_by(ROWS) {
            for j in (0..(M - M % COLS)).step_by(COLS) {
                unsafe { self.matmul_tile::<N,M,K,{ ROWS }, { COLS }>(
                    &l, &r, ROWS, COLS, i, j, &mut acc
                ) };
            }
        }

        if M % ROWS != 0 {
            for j in (0..(M - M % COLS)).step_by(COLS) {
                unsafe { self.matmul_tile::<N,M,K,{ ROWS }, { COLS }>(
                    &l, &r, N % ROWS, COLS, N - N % ROWS, j,
                    &mut acc
                ) };
            }
        }

        if N % COLS != 0 {
            for i in (0..(N - N % ROWS)).step_by(ROWS) {
                unsafe { self.matmul_tile::<N,M,K,{ ROWS },{ COLS }>(
                    &l, &r, ROWS, M % COLS, i, M - M % COLS,
                    &mut acc
                ) };
            }
        }

        if N % ROWS != 0 && M % COLS != 0 {
            unsafe { self.matmul_tile::<N,M,K,{ ROWS }, { COLS }>(
                &l,&r,
                N % ROWS,
                M % COLS,
                N - N % ROWS,
                M - M % COLS,
                &mut acc
            ) };
        }
    }

    #[inline(always)]
    unsafe fn matmul_tile<'a,const N: usize,const M: usize,const K: usize,const ROWS: usize,const COLS: usize>(
        &self,
        l:&MatrixView<'a,SL,N,K>,
        r:&ColumnMajorMatrix<'a,SR,K,M>,
        rows:usize,
        cols:usize,
        i:usize,
        j:usize,
        acc:&mut MatrixMutView<'a,SO,N,M>
    ) {
        let mut acc_tile = [[<Self as SimdPartialDot<SL,SR,SO>>::zero_acc(self);COLS];ROWS];

        let ref_l = l.as_ref();
        let ref_r = r.as_ref();

        unsafe {
            let mut rr: [<Self as SimdReg<SL>>::Reg; ROWS] = std::mem::MaybeUninit::uninit().assume_init();
            let mut cr: [<Self as SimdReg<SR>>::Reg; COLS] = std::mem::MaybeUninit::uninit().assume_init();

            for k in (0..(K - K % <Self as SimdLanes<SL>>::LANES)).step_by(<Self as SimdLanes<SL>>::LANES) {
                for r in 0..rows {
                    rr[r] = <Self as SimdLoad<SL>>::load(self, ref_l[((i + r) * K + k)..].as_ptr());
                }

                for c in 0..cols {
                    cr[c] = <Self as SimdLoad<SR>>::load(self, ref_r[((j + c) * K + k)..].as_ptr());
                }

                for r in 0..rows {
                    for c in 0..cols {
                        acc_tile[r][c] = <Self as SimdPartialDot<SL, SR, SO>>::partial_dot(self, rr[r], cr[c], acc_tile[r][c]);
                    }
                }
            }

            for r in 0..rows {
                for c in 0..cols {
                    acc[(i + r,j + c)] = <Self as SimdHSum<SO>>::hsum(self,acc_tile[r][c].fold(self));
                }
            }

            for k in (K - K % <Self as SimdLanes<SL>>::LANES)..K {
                for row in 0..rows {
                    for c in 0..cols {
                        acc[(i + row,j + c)] += SO::from(l[i + row][k]) * SO::from(r.col(j + c)[k]);
                    }
                }
            }
        }
    }
}
impl<BE,SL,SR,S> SimdLinearCombinationKernel<SL,SR,S> for BE
    where BE: Backend + NativeBackend +
              SimdReg<SL> +
              SimdReg<SR> +
              SimdAdd<SL,SR,SL,Rhs=Regs<<Self as SimdReg<SR>>::Reg,1>> +
              SimdSign<SR,Input=<Self as SimdReg<SR>>::Reg,Output=<Self as SimdReg<SR>>::Reg> +
              SimdSplat<<S as Sign>::Mask> +
              SimdLoadSeq<SL,<Self as SimdAdd<SL,SR,SL>>::Lhs> +
              SimdLoad<SR> +
              SimdPartialLoadSeq<SL,<Self as SimdAdd<SL,SR,SL>>::Lhs> +
              SimdPartialLoad<SR> +
              SimdStoreSeq<SL,<Self as SimdAdd<SL,SR,SL>>::Output> +
              SimdPartialStoreSeq<SL,<Self as SimdAdd<SL,SR,SL>>::Output> +
              SimdLanes<SR>,
          SR: BitsBitXor + Copy,
          SL: Copy,
          S: Sign<Mask=<SR as BitsBitXor>::Bits> {
    #[inline(always)]
    unsafe fn linear_combination<'a, 'b, const N: usize>(&self,
                                                         mut acc: VectorMutView<'b,SL,N>,
                                                         lines: &[(VectorView<'a, SR, N>, S)],
    ) {
        unsafe {
            let ref_o = acc.as_mut();

            for (line,sign) in lines {
                let mask = self.splat(sign.mask());
                let ref_r = line.as_ref();

                {
                    let chunks_r = ref_r.chunks_exact(<Self as SimdLanes<SR>>::LANES);
                    let chunks_o = ref_o.chunks_exact_mut(<Self as SimdLanes<SR>>::LANES);

                    for (co, cr) in chunks_o.zip(chunks_r) {
                        let ra = self.load_seq(co.as_ptr());
                        let rb = self.load(cr.as_ptr());

                        let rr = Regs::new([self.sign(rb, mask)]);

                        let rr = self.add(ra,rr);

                        self.store_seq(co.as_mut_ptr(), rr);
                    }
                }

                if N % <Self as SimdLanes<SR>>::LANES != 0 {
                    let rem = N % <Self as SimdLanes<SR>>::LANES;

                    let cr = ref_r.chunks_exact(<Self as SimdLanes<SR>>::LANES).remainder();
                    let co = ref_o.chunks_exact_mut(<Self as SimdLanes<SR>>::LANES).into_remainder();

                    let ra = self.partial_load_seq::<N>(co);
                    let rb = self.partial_load(cr,rem);

                    let rr = Regs::new([self.sign(rb, mask)]);

                    let rr = self.add(ra,rr);
                    self.partial_store_seq::<N>(co, rr);
                }
            }
        }
    }
}
impl BitsBitAnd for i8 {
    type Bits = i8;

    #[inline(always)]
    fn bits_bitand(self,m:i8) -> Self {
        self & m
    }
}
impl BitsBitAnd for i16 {
    type Bits = i16;
    #[inline(always)]
    fn bits_bitand(self,m:i16) -> Self {
        self & m
    }
}
impl BitsBitAnd for i32 {
    type Bits = i32;
    #[inline(always)]
    fn bits_bitand(self,m:i32) -> Self {
        self & m
    }
}
impl BitsBitAnd for i64 {
    type Bits = i64;
    #[inline(always)]
    fn bits_bitand(self,m:i64) -> Self {
        self & m
    }
}
impl BitsBitAnd for f32 {
    type Bits = u32;

    #[inline(always)]
    fn bits_bitand(self, m: Self::Bits) -> Self {
        f32::from_bits(self.to_bits() & m)
    }
}
impl BitsBitAnd for f64 {
    type Bits = u64;
    #[inline(always)]
    fn bits_bitand(self, m: Self::Bits) -> Self {
        f64::from_bits(self.to_bits() & m)
    }
}
impl BitsBitOr for i8 {
    type Bits = i8;

    #[inline(always)]
    fn bits_bitor(self,m:i8) -> Self {
        self | m
    }
}
impl BitsBitOr for i16 {
    type Bits = i16;
    fn bits_bitor(self,m:i16) -> Self {
        self | m
    }
}
impl BitsBitOr for i32 {
    type Bits = i32;
    #[inline(always)]
    fn bits_bitor(self,m:i32) -> Self {
        self | m
    }
}
impl BitsBitOr for i64 {
    type Bits = i64;
    #[inline(always)]
    fn bits_bitor(self,m:i64) -> Self {
        self | m
    }
}
impl BitsBitOr for f32 {
    type Bits = u32;

    #[inline(always)]
    fn bits_bitor(self, m: Self::Bits) -> Self {
        f32::from_bits(self.to_bits() | m)
    }
}
impl BitsBitOr for f64 {
    type Bits = u64;
    #[inline(always)]
    fn bits_bitor(self, m: Self::Bits) -> Self {
        f64::from_bits(self.to_bits() | m)
    }
}
impl BitsBitXor for i8 {
    type Bits = i8;
    #[inline(always)]
    fn bits_bitxor(self,m:i8) -> Self {
        self ^ m
    }
}
impl BitsBitXor for i16 {
    type Bits = i16;
    #[inline(always)]
    fn bits_bitxor(self,m:i16) -> Self {
        self ^ m
    }
}
impl BitsBitXor for i32 {
    type Bits = i32;
    #[inline(always)]
    fn bits_bitxor(self,m:i32) -> Self {
        self ^ m
    }
}
impl BitsBitXor for i64 {
    type Bits = i64;
    #[inline(always)]
    fn bits_bitxor(self,m:i64) -> Self {
        self ^ m
    }
}
impl BitsBitXor for f32 {
    type Bits = u32;
    #[inline(always)]
    fn bits_bitxor(self, m: Self::Bits) -> Self {
        f32::from_bits(self.to_bits() ^ m)
    }
}
impl BitsBitXor for f64 {
    type Bits = u64;
    #[inline(always)]
    fn bits_bitxor(self, m: Self::Bits) -> Self {
        f64::from_bits(self.to_bits() ^ m)
    }
}
impl BitsBitNot for i8 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for i16 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for i32 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for i64 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        !self
    }
}
impl BitsBitNot for f32 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        f32::from_bits(!self.to_bits())
    }
}
impl BitsBitNot for f64 {
    #[inline(always)]
    fn bits_bitnot(self) -> Self {
        f64::from_bits(!self.to_bits())
    }
}
impl BitsShl for i16 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        self << w
    }
}
impl BitsShl for i32 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        self << w
    }
}
impl BitsShl for f32 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        f32::from_bits(self.to_bits() << w)
    }
}
impl BitsShl for f64 {
    #[inline(always)]
    fn bits_shl(self,w:usize) -> Self {
        f64::from_bits(self.to_bits() << w)
    }
}
impl BitsShr for i16 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        self >> w
    }
}
impl BitsShr for i32 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        self >> w
    }
}
impl BitsShr for f32 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        f32::from_bits(self.to_bits() >> w)
    }
}
impl BitsShr for f64 {
    #[inline(always)]
    fn bits_shr(self,w:usize) -> Self {
        f64::from_bits(self.to_bits() >> w)
    }
}
impl<T> Assume<T> for T {
    #[inline(always)]
    fn assume(self) -> T {
        self
    }
}
impl<T> Assume<T> for &T where T: Assume<T> + Copy {
    #[inline(always)]
    fn assume(self) -> T {
        *self
    }
}
impl Assume<f32> for f64 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i8> for f64 {
    #[inline(always)]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<i16> for f64 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<f64> for f32 {
    #[inline(always)]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<i16> for f32 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<i8> for f32 {
    #[inline(always)]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<f64> for i8 {
    #[inline(always)]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<f32> for i8 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i16> for i8 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<i32> for i8 {
    #[inline(always)]
    fn assume(self) -> i32 {
        self as i32
    }
}
impl Assume<f32> for i16 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<f64> for i16 {
    #[inline(always)]
    fn assume(self) -> f64 {
        self as f64
    }
}
impl Assume<i32> for i16 {
    #[inline(always)]
    fn assume(self) -> i32 {
        self as i32
    }
}
impl Assume<i8> for i16 {
    #[inline(always)]
    fn assume(self) -> i8 {
        self as i8
    }
}
impl Assume<f32> for i32 {
    #[inline(always)]
    fn assume(self) -> f32 {
        self as f32
    }
}
impl Assume<i16> for i32 {
    #[inline(always)]
    fn assume(self) -> i16 {
        self as i16
    }
}
impl Assume<i8> for i32 {
    #[inline(always)]
    fn assume(self) -> i8 {
        self as i8
    }
}
#[derive(Debug,Clone,Copy)]
pub struct SignInt<T> where T: Debug + Clone + Copy {
    mask: T
}
impl<T> SignInt<T> where T: Debug + Clone + Copy {
    pub fn mask(&self) -> T {
        self.mask
    }
}
impl Sign for SignInt<i64> {
    type Mask = i64;
    fn plus() -> Self {
        Self {
            mask: 0
        }
    }

    fn minus() -> Self {
        Self {
            mask: !0
        }
    }

    fn mask(&self) -> Self::Mask {
        self.mask
    }
}
impl Sign for SignInt<i32> {
    type Mask = i32;
    fn plus() -> Self {
        Self {
            mask: 0
        }
    }

    fn minus() -> Self {
        Self {
            mask: !0
        }
    }

    fn mask(&self) -> Self::Mask {
        self.mask
    }
}
impl Sign for SignInt<i16> {
    type Mask = i16;
    fn plus() -> Self {
        Self {
            mask: 0
        }
    }

    fn minus() -> Self {
        Self {
            mask: !0
        }
    }

    fn mask(&self) -> Self::Mask {
        self.mask
    }
}
impl Sign for SignInt<i8> {
    type Mask = i8;
    fn plus() -> Self {
        Self {
            mask: 0
        }
    }

    fn minus() -> Self {
        Self {
            mask: !0
        }
    }

    fn mask(&self) -> Self::Mask {
        self.mask
    }
}
#[derive(Debug,Clone,Copy)]
pub struct SignFloat<T>
    where T: BitsBitXor + Debug + Clone + Copy,
          <T as BitsBitXor>::Bits: Debug + Clone + Copy {
    mask: <T as BitsBitXor>::Bits
}
impl<T> SignFloat<T>
    where T: BitsBitXor + Debug + Clone + Copy,
          <T as BitsBitXor>::Bits: Debug + Clone + Copy {
    pub fn mask(&self) -> <T as BitsBitXor>::Bits {
        self.mask
    }
}
impl Sign for SignFloat<f64> {
    type Mask = u64;
    fn plus() -> Self {
        Self {
            mask: 0
        }
    }

    fn minus() -> Self {
        Self {
            mask: 1 << 63
        }
    }

    fn mask(&self) -> Self::Mask {
        self.mask
    }
}
impl Sign for SignFloat<f32> {
    type Mask = u32;
    fn plus() -> Self {
        Self {
            mask: 0
        }
    }

    fn minus() -> Self {
        Self {
            mask: 1 << 31
        }
    }

    fn mask(&self) -> Self::Mask {
        self.mask
    }
}
/// Operations Between Elements of the Same Type
pub enum Homogeneous {}
/// Operations Between Different Types
pub enum Heterogeneous {}
impl SupportMul<Heterogeneous> for (i8,i32) {}
impl SupportMul<Heterogeneous> for (i16,i32) {}
impl<T> SupportMul<Homogeneous> for (T,T) {}
/// Number of elements in a vector
pub enum Extent<const N: usize> {}
/// The size of each dimension of a multidimensional array
pub struct Dims<E: DimOne,D: Dim> {
    extent:PhantomData<E>,
    dim:PhantomData<D>
}
impl<const N:usize> Dim for Extent<N> {}
impl<const N:usize> DimOne for Extent<N> {}
impl<E:DimOne,D:Dim> Dim for Dims<E,D> {}
/// Implementation of a mapper that maps inputs using data loaded from the data loader and returns the results
pub struct LoaderMapper<D,SI,SO,L,F: FnMut(SI,<L as Loader<D>>::Output) -> SO>
    where D: Dim, L: Loader<D> {
    dim:PhantomData<D>,
    si:PhantomData<SI>,
    so:PhantomData<SO>,
    loader:L,
    f:F
}
impl<D,L,F,SI,SO> Mapper<D,SI,SO> for LoaderMapper<D,SI,SO,L,F>
    where D: Dim,
          L: Loader<D>,
          F: FnMut(SI,<L as Loader<D>>::Output) -> SO {
    #[inline(always)]
    fn map(&mut self, i: SI) -> SO {
        (self.f)(i,self.loader.load())
    }
}
/// A builder for generating mappers that use data sources
pub struct LoaderMappperBuilder {
}
impl LoaderMappperBuilder {
    /// Create a LoaderMappperBuilder
    pub fn new() -> LoaderMappperBuilder {
        LoaderMappperBuilder {
        }
    }

    /// Build a LoaderMapper from a loader and a mapping function
    pub fn build<D,SI,SO,L,F: FnMut(SI,<L as Loader<D>>::Output) -> SO>(self, loader:L,f:F) -> LoaderMapper<D,SI,SO,L,F>
        where D: Dim,
              L: Loader<D> {
        LoaderMapper {
            dim:PhantomData::<D>,
            si:PhantomData::<SI>,
            so:PhantomData::<SO>,
            loader:loader,
            f
        }
    }
}
/// A loader for loading Vector
pub struct VectorLoader<'a,T,const N:usize> where T: Copy + 'static {
    it:<VectorView<'a,T,N> as IntoIterator>::IntoIter
}
impl<'a,T,const N: usize> VectorLoader<'a,T,N> where T: Copy + 'static {
    /// Create a VectorLoader from a VectorView
    pub fn new(v:VectorView<'a,T,N>) -> VectorLoader<'a,T,N> {
        VectorLoader {
            it:v.into_iter()
        }
    }
}
impl<'a,T,const N: usize> Loader<Extent<N>> for VectorLoader<'a,T,N>
    where T: Copy + 'static {
    type Output = T;

    #[inline(always)]
    fn load(&mut self) -> Self::Output {
        *self.it.next().expect("Unable to read an item from the iterator.")
    }
}
/// A type that abstracts operators that perform addition
pub struct AddOperator<SL,SR,SO> {
    sl:PhantomData<SL>,
    sr:PhantomData<SR>,
    so:PhantomData<SO>
}
impl<SL,SR,SO> AddOperator<SL,SR,SO> {
    #[inline(always)]
    pub fn new() -> AddOperator<SL,SR,SO> {
        AddOperator {
            sl:PhantomData::<SL>,
            sr:PhantomData::<SR>,
            so:PhantomData::<SO>
        }
    }
}
impl<SL,SR,SO> BinaryOperator for AddOperator<SL,SR,SO>
    where SL: Copy + 'static,
          SR: Copy + 'static,
          SO: Copy + 'static {
    type Lhs = SL;
    type Rhs = SR;
    type Output = SO;
    #[inline(always)]
    fn compose<F: Fn(Self) -> OP, OP>(self, f: F) -> OP
        where OP: BinaryOperator {
        f(self)
    }
}
impl<BE,SL,SR,SO> ToSimdBinaryOperator<BE> for AddOperator<SL,SR,SO>
    where BE: Backend + NativeBackend + Clone +
              SimdResolveBinaryOperator<AddOperator<SL,SR,SO>,SL,SR,SO>,
          SL: Copy,
          SR: Copy,
          SO: Copy,
          Self: BinaryOperator<Lhs=SL,Rhs=SR,Output=SO> {
    type NativeOperator = <BE as SimdResolveBinaryOperator<AddOperator<SL,SR,SO>,SL,SR,SO>>::NativeOperator;
    #[inline(always)]
    fn to_simd_operator(&self,backend:&BE) -> Self::NativeOperator {
        backend.resolve_operator(self)
    }
}
/// A type that abstracts operators that perform right shift.
pub struct ShrOperator<P,T> {
    parent:P,
    s:PhantomData<T>,
    pub width:usize
}
impl<P,T> ShrOperator<P,T> {
    #[inline(always)]
    pub fn new(parent:P,width:usize) -> ShrOperator<P,T> {
        ShrOperator {
            parent:parent,
            s:PhantomData::<T>,
            width:width
        }
    }
}
impl<P,T> BinaryOperator for ShrOperator<P,T>
    where P: BinaryOperator,
          T: Copy + 'static {
    type Lhs = P::Lhs;
    type Rhs = P::Rhs;
    type Output = T;
    #[inline(always)]
    fn compose<F: Fn(Self) -> OP, OP>(self, f: F) -> OP
        where OP: BinaryOperator {
        f(self)
    }
}
impl<BE,P,T> ToSimdBinaryOperator<BE> for ShrOperator<P,T>
    where BE: Backend + NativeBackend +
              SimdResolvePatrialBinaryOperator<P,ShrOperator<P,T>,Self::Lhs,Self::Rhs,T> +
              SimdReg<T> +
              SimdShiftWidth<T> +
              SimdShr<T> +
              Clone,
          P: BinaryOperator + ToSimdBinaryOperator<BE>,
          T: Copy + 'static {
    type NativeOperator = <BE as SimdResolvePatrialBinaryOperator<P,ShrOperator<P,T>,Self::Lhs,Self::Rhs,T>>::NativeOperator;
    #[inline(always)]
    fn to_simd_operator(&self,backend:&BE) -> Self::NativeOperator {
        self.parent.to_simd_operator(backend).compose(|p| {
            backend.resolve_operator(p,self)
        })
    }
}
/// A type that abstracts operators that perform demotion.
pub struct DemoteOperator<P,T> {
    parent:P,
    s:PhantomData<T>,
}
impl<P,T> DemoteOperator<P,T> {
    #[inline(always)]
    pub fn new(parent:P) -> DemoteOperator<P,T> {
        DemoteOperator {
            parent:parent,
            s:PhantomData::<T>,
        }
    }
}
impl<P,T> BinaryOperator for DemoteOperator<P,T>
    where P: BinaryOperator,
          T: Copy + 'static {
    type Lhs = P::Lhs;
    type Rhs = P::Rhs;
    type Output = T;
    #[inline(always)]
    fn compose<F: Fn(Self) -> OP, OP>(self, f: F) -> OP
        where OP: BinaryOperator {
        f(self)
    }
}
impl<BE,P,T> ToSimdBinaryOperator<BE> for DemoteOperator<P,T>
    where BE: Backend + NativeBackend +
              SimdResolvePatrialBinaryOperator<P,DemoteOperator<P,T>,Self::Lhs,Self::Rhs,T> +
              SimdReg<T> +
              Clone,
      P: BinaryOperator + ToSimdBinaryOperator<BE>,
      T: Copy + 'static {
    type NativeOperator = <BE as SimdResolvePatrialBinaryOperator<P,DemoteOperator<P,T>,Self::Lhs,Self::Rhs,T>>::NativeOperator;
    #[inline(always)]
    fn to_simd_operator(&self,backend:&BE) -> Self::NativeOperator {
        self.parent.to_simd_operator(backend).compose(|p| {
            backend.resolve_operator(p,self)
        })
    }
}
