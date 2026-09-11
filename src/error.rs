//! SIMD Error Types

use std::{error, fmt};

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



