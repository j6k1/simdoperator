//! Automatically select the best backend for the current CPU.

use crate::backend::avx2::Avx2;
use crate::backend::common::Backend;
use crate::error::InstantiationError;

pub enum SelectedBackend {
    Avx2(Avx2)
}
pub struct AutoSelect {
    backend: SelectedBackend
}
impl Backend for AutoSelect {
    fn new() -> Result<Self, InstantiationError> {
        let backend = Avx2::new()?;
        
        Ok(AutoSelect {
            backend: SelectedBackend::Avx2(backend)
        })
    }
}