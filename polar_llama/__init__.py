from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import polars as pl

from polar_llama.utils import parse_into_expr, register_plugin, parse_version

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr

if parse_version(pl.__version__) < parse_version("0.20.16"):
    from polars.utils.udfs import _get_shared_lib_location

    lib: str | Path = _get_shared_lib_location(__file__)
else:
    lib = Path(__file__).parent

# Import Provider enum directly from the extension module
try:
    # First try relative import from the extension module in current directory
    from .polar_llama import Provider
except ImportError:
    # Fallback to try absolute import
    try:
        from polar_llama.polar_llama import Provider
    except ImportError:
        # Define a basic Provider class as fallback if neither import works
        class Provider:
            OPENAI = "openai"
            ANTHROPIC = "anthropic"
            GEMINI = "gemini"
            GROQ = "groq"
            
            def __init__(self, provider_str):
                self.value = provider_str
                
            def __str__(self):
                return self.value

# Import and initialize the expressions helper to ensure expressions are registered
from polar_llama.expressions import ensure_expressions_registered, get_lib_path

# Ensure the expressions are registered
ensure_expressions_registered()
# Update the lib path to make sure we're using the actual library
lib = get_lib_path()

def inference_async(
    expr: IntoExpr, 
    *, 
    provider: Optional[Union[str, Provider]] = None, 
    model: Optional[str] = None
) -> pl.Expr:
    """
    Asynchronously infer completions for the given text expressions using an LLM.
    
    Parameters
    ----------
    expr : polars.Expr
        The text expression to use for inference
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq)
    model : str, optional
        The model name to use
        
    Returns
    -------
    polars.Expr
        Expression with inferred completions
    """
    expr = parse_into_expr(expr)
    kwargs = {}
    
    if provider is not None:
        # Convert Provider to string to make it picklable
        if isinstance(provider, Provider):
            provider = str(provider)
        kwargs["provider"] = provider
        
    if model is not None:
        kwargs["model"] = model
        
    return register_plugin(
        args=[expr],
        symbol="inference_async",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )

def inference(
    expr: IntoExpr, 
    *, 
    provider: Optional[Union[str, Provider]] = None, 
    model: Optional[str] = None
) -> pl.Expr:
    """
    Synchronously infer completions for the given text expressions using an LLM.
    
    Parameters
    ----------
    expr : polars.Expr
        The text expression to use for inference
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq)
    model : str, optional
        The model name to use
        
    Returns
    -------
    polars.Expr
        Expression with inferred completions
    """
    expr = parse_into_expr(expr)
    kwargs = {}
    
    if provider is not None:
        # Convert Provider to string to make it picklable
        if isinstance(provider, Provider):
            provider = str(provider)
        kwargs["provider"] = provider
        
    if model is not None:
        kwargs["model"] = model
        
    return register_plugin(
        args=[expr],
        symbol="inference",
        is_elementwise=True,
        lib=lib,
        kwargs=kwargs,
    )

def string_to_message(expr: IntoExpr, *, message_type: str) -> pl.Expr:
    """
    Convert a string to a message with the specified type.
    
    Parameters
    ----------
    expr : polars.Expr
        The text expression to convert
    message_type : str
        The type of message to create ("user", "system", "assistant")
        
    Returns
    -------
    polars.Expr
        Expression with formatted messages
    """
    expr = parse_into_expr(expr)
    return register_plugin(
        args=[expr],
        symbol="string_to_message",
        is_elementwise=True,
        lib=lib,
        kwargs={"message_type": message_type},
    )