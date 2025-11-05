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
    
    #[classattr]
    const BEDROCK: Self = PyProvider(Provider::Bedrock);
    
    #[new]
    fn new(provider_str: &str) -> PyResult<Self> {
        match Provider::from_str(provider_str) {
            Ok(provider) => Ok(PyProvider(provider)),
            Err(err) => Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid provider: {err}")
            )),
        }
    }
    
    fn __str__(&self) -> String {
        self.0.as_str().to_string()
    }
}

// Register expression functions with the Python module
#[pyfunction]
fn register_expressions(_py: Python<'_>) -> PyResult<&'static str> {
    // We don't need to do anything here since the expressions are registered by the polars_expr macro
    // This function exists just to make it explicit in the code that the expressions are registered
    Ok("Expressions registered successfully")
}

#[pymodule]
fn polar_llama(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.setattr("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add the PyProvider class to the module
    m.add_class::<PyProvider>()?;
    
    // Add the register_expressions function to the module
    m.add_function(wrap_pyfunction!(register_expressions, m)?)?;
    
    // Display a message in Python's stdout when the module is loaded
    py.import("builtins")?.getattr("print")?.call1((
        "polar_llama module loaded successfully. Polars expressions should be available.",
    ))?;
    
    Ok(())
}
