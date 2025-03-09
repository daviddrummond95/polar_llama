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
use std::str::FromStr;

// Make PyProvider available at module level for Python
#[pyclass(name = "Provider")]
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
            Ok(provider) => Ok(PyProvider(provider)),
            Err(err) => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid provider: {}", err)
            )),
        }
    }
    
    fn __str__(&self) -> String {
        self.0.as_str().to_string()
    }
}

// The crucial fix: We need to add this attribute to register the polars expressions
// This will ensure that pyo3_polars properly registers the expression functions defined in expressions.rs
#[pyo3_polars::pyo3_register_polars_expressions]

#[pymodule]
fn polar_llama(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add the PyProvider class to the module
    m.add_class::<PyProvider>()?;
    
    // Register the Polars expressions - this line is optional since the #[pyo3_register_polars_expressions]
    // attribute above should handle this, but including it for extra clarity
    pyo3_polars::register(_py, m)?;
    
    Ok(())
}
