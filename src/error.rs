//! SIMD Error Types

use std::{error, fmt};

#[derive(Debug)]
pub enum InstantiationError {
    TryFromSliceError(TryFromSliceError),
    NotSupportingError
}

impl fmt::Display for InstantiationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InstantiationError::TryFromSliceError(e) => write!(f, "{}", e),
            InstantiationError::NotSupportingError => write!(f, "This environment is not supported."),
        }
    }
}
impl error::Error for InstantiationError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
    fn description(&self) -> &str {
        match self {
            InstantiationError::TryFromSliceError(_) => "Could not convert slice to SIMD vector",
            InstantiationError::NotSupportingError => "This environment is not supported.",
        }
    }
}
impl From<TryFromSliceError> for InstantiationError {
    fn from(err: TryFromSliceError) -> InstantiationError {
        InstantiationError::TryFromSliceError(err)
    }
}
impl From<std::array::TryFromSliceError> for InstantiationError {
    fn from(_: std::array::TryFromSliceError) -> InstantiationError {
        InstantiationError::TryFromSliceError(TryFromSliceError)
    }
}

#[derive(Debug)]
pub struct TryFromSliceError;

impl fmt::Display for TryFromSliceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TryFromSliceError => write!(f, "Could not convert slice to SIMD vector"),
        }
    }
}
impl error::Error for TryFromSliceError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        None
    }
    fn description(&self) -> &str {
        match self {
            TryFromSliceError => "Could not convert slice to SIMD vector",
        }
    }

}
impl From<std::array::TryFromSliceError> for TryFromSliceError {
    fn from(_: std::array::TryFromSliceError) -> TryFromSliceError {
        TryFromSliceError
    }
}



