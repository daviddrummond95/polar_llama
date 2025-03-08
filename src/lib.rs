mod expressions;
pub mod utils;
pub mod model_client;

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

use pyo3::prelude::*;
use model_client::Provider;

// Make PyProvider available at module level for Python
#[pyclass]
#[derive(Clone)]
pub struct PyProvider(Provider);

#[pymethods]
impl PyProvider {
    #[classattr]
    const OPENAI: Self = PyProvider(Provider::OpenAI);
    
    #[classattr]
    const ANTHROPIC: Self = PyProvider(Provider::Anthropic);
    
    #[classattr]
    const GEMINI: Self = PyProvider(Provider::Gemini);
    
    #[classattr]
    const GROQ: Self = PyProvider(Provider::Groq);
    
    #[new]
    fn new(provider_str: &str) -> PyResult<Self> {
        match Provider::from_str(provider_str) {
            Some(provider) => Ok(PyProvider(provider)),
            None => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid provider: {}", provider_str)
            )),
        }
    }
    
    fn __str__(&self) -> String {
        self.0.as_str().to_string()
    }
}

#[pymodule]
fn polar_llama(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add the PyProvider class to the module
    m.add_class::<PyProvider>()?;
    
    Ok(())
}
