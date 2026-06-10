"""
Helper module for working with Polars expressions in polar_llama.
"""
import os
from pathlib import Path
import polars as pl

# Import the register_expressions function to ensure it gets called
try:
    from polar_llama.polar_llama import register_expressions
    # Call it to make sure expressions are registered
    register_expressions()
except ImportError:
    import warnings
    warnings.warn(
        "Could not import register_expressions from polar_llama.polar_llama",
        RuntimeWarning,
        stacklevel=2,
    )
except Exception as e:  # noqa: BLE001
    import warnings
    warnings.warn(f"Error calling register_expressions: {e}", RuntimeWarning, stacklevel=2)

def get_lib_path():
    """Get the path to the native library."""
    # Find the shared library
    lib_dir = Path(__file__).parent
    
    # Look for any .so or .dll files in the directory
    potential_libs = list(lib_dir.glob("*.so")) + list(lib_dir.glob("*.abi3.so")) + list(lib_dir.glob("*.dll"))
    
    if potential_libs:
        # Return the first one found
        return str(potential_libs[0])
    else:
        # As a fallback, guess the name based on the module name
        if os.name == 'posix':
            return str(lib_dir / "polar_llama.so")
        return str(lib_dir / "polar_llama.pyd")

def ensure_expressions_registered():
    """Ensure all expressions are registered with Polars."""
    lib_path = get_lib_path()
    if not os.path.exists(lib_path):
        import warnings
        warnings.warn(
            f"polar_llama native library not found at {lib_path}; "
            "expressions will not be available. Did the build succeed?",
            RuntimeWarning,
            stacklevel=2,
        )
    return True 