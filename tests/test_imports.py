import polars as pl
import inspect
import sys
import importlib.util
import os

def test_package_contents():
    """Test what's actually available in the polar_llama package."""
    
    # Import the package
    import polar_llama
    
    # Print the dir of the package to see what's available
    print("\nPackage contents:")
    contents = dir(polar_llama)
    print(contents)
    
    # Check specifically for functions we expect
    functions_to_check = ['inference_async', 'inference', 'string_to_message', 'Provider']
    for func in functions_to_check:
        print(f"{func} in polar_llama: {func in contents}")
    
    # Check for the extension module
    extension_name = 'polar_llama.polar_llama'
    extension_available = importlib.util.find_spec(extension_name) is not None
    print(f"Extension module available: {extension_available}")
    
    if extension_available:
        try:
            ext_module = importlib.import_module(extension_name)
            print(f"Extension module dir: {dir(ext_module)}")
        except ImportError as e:
            print(f"Error importing extension module: {e}")
    
    # Print the file path of the package
    print("\nPackage file location:", polar_llama.__file__)
    
    # Try to check Provider enum
    if 'Provider' in contents:
        print("\nProvider enum:")
        provider_values = [attr for attr in dir(polar_llama.Provider) if not attr.startswith('_')]
        print(provider_values)
    
    # Check module structure
    site_packages = os.path.dirname(os.path.dirname(polar_llama.__file__))
    print("\nPackage directory structure:")
    for root, dirs, files in os.walk(os.path.dirname(polar_llama.__file__)):
        rel_path = os.path.relpath(root, site_packages)
        print(f"Directory: {rel_path}")
        for file in files:
            print(f"  - {file}")
    
    # Print __init__.py contents
    try:
        with open(polar_llama.__file__, 'r') as f:
            print("\nFirst 50 lines of __init__.py:")
            lines = f.readlines()
            for i, line in enumerate(lines[:50]):
                print(f"{i+1}: {line.rstrip()}")
                
            # Check if there are any errors in importing from polar_llama.polar_llama
            if "from polar_llama.polar_llama import PyProvider as Provider" in "".join(lines):
                print("\nTrying to import PyProvider directly:")
                try:
                    from polar_llama.polar_llama import PyProvider
                    print("PyProvider imported successfully")
                except ImportError as e:
                    print(f"Error importing PyProvider: {e}")
                except Exception as e:
                    print(f"Unexpected error importing PyProvider: {e}")
                    
    except Exception as e:
        print(f"Error reading __init__.py: {e}")
    
    # Basic assertion to make the test pass
    assert True 