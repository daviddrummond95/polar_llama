#[allow(clippy::single_component_path_imports)]
use pyo3_build_config;

fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
