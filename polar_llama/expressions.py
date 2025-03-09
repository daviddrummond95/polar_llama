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
    print("Warning: Could not import register_expressions from polar_llama.polar_llama")
except Exception as e:
    print(f"Warning: Error calling register_expressions: {e}")

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
        else:
            return str(lib_dir / "polar_llama.pyd")

def ensure_expressions_registered():
    """Ensure all expressions are registered with Polars."""
    # This is mainly for debugging
    print(f"Using library at: {get_lib_path()}")
    
    # Check that the expressions are available
    expressions = ["inference", "inference_async", "string_to_message"]
    for expr in expressions:
        try:
            # Try to directly access the expression via the plugin registry
            registered = hasattr(pl.plugin_registry, expr)
            print(f"Expression {expr} registered: {registered}")
        except Exception as e:
            print(f"Error checking expression {expr}: {e}")
    
    return True 