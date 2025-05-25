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
            BEDROCK = "bedrock"
            
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
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
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
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
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

def inference_messages(
    expr: IntoExpr, 
    *, 
    provider: Optional[Union[str, Provider]] = None, 
    model: Optional[str] = None
) -> pl.Expr:
    """
    Process message arrays (conversations) for inference using LLMs.
    
    This function accepts properly formatted JSON message arrays and sends them
    to the LLM for inference while preserving conversation context.
    
    Parameters
    ----------
    expr : polars.Expr
        The expression containing JSON message arrays
    provider : str or Provider, optional
        The provider to use (OpenAI, Anthropic, Gemini, Groq, Bedrock)
    model : str, optional
        The model name to use
        
    Returns
    -------
    polars.Expr
        Expression with inferred completions
    """
    import inspect
    print(f"Type of provider: {type(provider)}")
    if provider is not None:
        print(f"Provider value: {provider}")
        print(f"Provider attributes: {inspect.getmembers(provider)}")
    
    expr = parse_into_expr(expr)
    kwargs = {}
    
    if provider is not None:
        # Convert Provider to string to make it picklable
        if hasattr(provider, 'as_str'):
            provider_str = provider.as_str()
        elif hasattr(provider, '__str__'):
            provider_str = str(provider)
        else:
            provider_str = provider
        
        print(f"Provider string: {provider_str}")
        kwargs["provider"] = provider_str
        
    if model is not None:
        kwargs["model"] = model
    
    print(f"Final kwargs: {kwargs}")
    
    # Don't pass empty kwargs dictionary    
    if not kwargs:
        return register_plugin(
            args=[expr],
            symbol="inference_messages",
            is_elementwise=True,
            lib=lib,
        )
    
    return register_plugin(
        args=[expr],
        symbol="inference_messages",
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

def combine_messages(*exprs: IntoExpr) -> pl.Expr:
    """
    Combine multiple message expressions into a single message array.
    
    This function takes multiple message expressions (strings containing JSON formatted messages)
    and combines them into a single JSON array of messages, preserving the order.
    
    Parameters
    ----------
    *exprs : polars.Expr
        One or more expressions containing messages to combine
        
    Returns
    -------
    polars.Expr
        Expression with combined message arrays
    """
    args = [parse_into_expr(expr) for expr in exprs]
    
    return register_plugin(
        args=args,
        symbol="combine_messages",
        is_elementwise=True,
        lib=lib,
    )