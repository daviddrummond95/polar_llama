#!/usr/bin/env python3
"""
Test script to verify Bedrock provider integration.
"""

import polars as pl
from polar_llama import Provider, string_to_message, inference_async

def test_bedrock_provider():
    """Test that the Bedrock provider is available and can be used."""
    
    # Test 1: Check that Provider.BEDROCK exists
    print("Test 1: Checking Provider.BEDROCK availability...")
    try:
        bedrock_provider = Provider.BEDROCK
        print(f"✓ Provider.BEDROCK is available: {bedrock_provider}")
    except AttributeError as e:
        print(f"✗ Provider.BEDROCK not available: {e}")
        return False
    
    # Test 2: Check that we can convert it to string
    print("\nTest 2: Converting Provider.BEDROCK to string...")
    try:
        provider_str = str(bedrock_provider)
        print(f"✓ Provider.BEDROCK as string: '{provider_str}'")
        assert provider_str == "bedrock", f"Expected 'bedrock', got '{provider_str}'"
    except Exception as e:
        print(f"✗ Failed to convert Provider.BEDROCK to string: {e}")
        return False
    
    # Test 3: Create a simple dataframe and test the function signature
    print("\nTest 3: Testing function signature with Bedrock provider...")
    try:
        # Create test data
        questions = [
            'What is the capital of France?',
            'Explain quantum computing in simple terms.'
        ]
        
        df = pl.DataFrame({'Questions': questions})
        
        # Add prompts to the dataframe
        df = df.with_columns(
            prompt=string_to_message("Questions", message_type='user')
        )
        
        print("✓ Created dataframe with prompts")
        print(df)
        
        # Test that we can create the expression (but don't execute it)
        # This tests that the provider parameter is accepted
        expr = inference_async(
            'prompt', 
            provider='bedrock', 
            model='anthropic.claude-3-haiku-20240307-v1:0'
        )
        print("✓ Successfully created inference_async expression with bedrock provider")
        
        # Also test with Provider enum
        expr2 = inference_async(
            'prompt', 
            provider=Provider.BEDROCK, 
            model='anthropic.claude-3-haiku-20240307-v1:0'
        )
        print("✓ Successfully created inference_async expression with Provider.BEDROCK")
        
    except Exception as e:
        print(f"✗ Failed to create expression with Bedrock provider: {e}")
        return False
    
    print("\n🎉 All tests passed! Bedrock provider is properly integrated.")
    print("\nNote: To actually use Bedrock, you'll need:")
    print("1. AWS credentials configured (AWS CLI, environment variables, or IAM roles)")
    print("2. Access to Amazon Bedrock service")
    print("3. The specific model enabled in your AWS account")
    
    return True

if __name__ == "__main__":
    test_bedrock_provider() 