"""
Helper module for working with Polars expressions in polar_llama.
"""
import os
from pathlib import Path
import polars as pl

# Import the register_expressions function to ensure it gets called
try:
    from polar_llama import register_expressions
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
        lib_path = str(potential_libs[0])
        print(f"Found library at: {lib_path}")
        return lib_path
    else:
        # As a fallback, guess the name based on the module name
        if os.name == 'posix':
            fallback_path = str(lib_dir / "polar_llama.so")
        else:
            fallback_path = str(lib_dir / "polar_llama.pyd")
        print(f"No library found, using fallback: {fallback_path}")
        return fallback_path

def ensure_expressions_registered():
    """Ensure all expressions are registered with Polars."""
    # This is mainly for debugging
    lib_path = get_lib_path()
    print(f"Using library at: {lib_path}")
    
    # Check if the library file actually exists
    if os.path.exists(lib_path):
        print(f"✓ Library file exists: {lib_path}")
    else:
        print(f"✗ Library file not found: {lib_path}")
        # List what files are actually in the directory
        lib_dir = Path(lib_path).parent
        print(f"Files in {lib_dir}:")
        for file in lib_dir.iterdir():
            print(f"  - {file.name}")
    
    return True 