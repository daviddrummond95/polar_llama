import pytest
import polars as pl
from pydantic import BaseModel, Field
from typing import Optional
import warnings
from polar_llama import Provider, template

# Mock the register_plugin to avoid actual Rust calls if possible, 
# or just rely on the fact that we are testing the Python API surface.
# However, since we don't have the Rust extension built in this environment (maybe?),
# we might hit issues if we try to actually execute the expressions.
# But wait, the user has the extension installed or built? 
# The file list showed `polar_llama.abi3.so`, so it should be importable.
# But `inference` calls into Rust.
# I can test the expression construction without evaluation for some parts,
# or use a mock provider if available (the library doesn't seem to have a mock provider easily exposed).
# Actually, `string_to_message` and `template` are pure python or simple wrappers.
# `inference` creates an expression. We can check the expression structure or try to run it if we have a key (which we might not).
# Let's focus on API surface and expression generation.

def test_namespace_accessor_to_message():
    df = pl.DataFrame({"text": ["hello"]})
    
    # Test .llama.to_message
    expr = pl.col("text").llama.to_message(role="user")
    
    # We can't easily check the internal expression structure without running it,
    # but we can check if it returns a pl.Expr
    assert isinstance(expr, pl.Expr)
    
    # If we can run it, let's try (might fail if extension not loaded properly or no runtime)
    # But `string_to_message` is a plugin call.
    # Let's assume the environment is set up correctly as per previous context.

def test_namespace_accessor_inference():
    # Test .llama.inference
    expr = pl.col("text").llama.inference(model="gpt-4")
    assert isinstance(expr, pl.Expr)

def test_namespace_accessor_inference_async():
    # Test .llama.inference_async
    expr = pl.col("text").llama.inference_async(model="gpt-4")
    assert isinstance(expr, pl.Expr)

def test_response_format_alias():
    class User(BaseModel):
        name: str
        age: int

    # Test inference with response_format
    expr = pl.col("text").llama.inference(response_format=User)
    assert isinstance(expr, pl.Expr)
    
    # Test inference_async with response_format
    expr = pl.col("text").llama.inference_async(response_format=User)
    assert isinstance(expr, pl.Expr)

def test_strict_mode_validation_warning():
    class Currency(BaseModel):
        code: str = "USD"  # Default value
        amount: float

    # Should warn when used
    with pytest.warns(UserWarning, match="has a default value"):
        pl.col("text").llama.inference(response_model=Currency)

def test_strict_mode_validation_no_warning():
    class Currency(BaseModel):
        code: str
        amount: float

    # Should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pl.col("text").llama.inference(response_model=Currency)
        # Filter out unrelated warnings if any
        relevant_warnings = [x for x in w if "has a default value" in str(x.message)]
        assert len(relevant_warnings) == 0

def test_template_helper():
    df = pl.DataFrame({"name": ["World"]})
    
    # Test positional args
    expr_pos = template("Hello {}", pl.col("name"))
    assert isinstance(expr_pos, pl.Expr)
    result_pos = df.select(expr_pos)
    assert result_pos.item(0, 0) == "Hello World"
    
    # Test kwargs (might not be supported in all Polars versions)
    # expr_kwargs = template("Hello {name}", name=pl.col("name"))
    # assert isinstance(expr_kwargs, pl.Expr)
    # result_kwargs = df.select(expr_kwargs)
    # assert result_kwargs.item(0, 0) == "Hello World"
