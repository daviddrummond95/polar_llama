import polars as pl
from polar_llama import inference_async, Provider
from pydantic import BaseModel
from typing import List
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MovieRecommendation(BaseModel):
    """A movie recommendation with a title and reason."""
    title: str
    genre: str
    year: int
    reason: str


def test_structured_output_basic():
    """Test basic structured output with Groq API."""
    # Check if API key is available
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping test: GROQ_API_KEY not set")
        return

    # Create a test dataframe
    df = pl.DataFrame({
        "prompt": ["Recommend a sci-fi movie from the 2010s"]
    })

    # Run inference with structured output
    result_df = df.with_columns(
        recommendation=inference_async(
            pl.col("prompt"),
            provider=Provider.GROQ,
            model="openai/gpt-oss-120b",
            response_model=MovieRecommendation
        )
    )

    print("\nTest Results (Struct):")
    print(result_df)

    # Access struct fields directly!
    print("\n✨ Direct struct field access:")
    try:
        # First check the raw response to see what we got
        raw = result_df['recommendation'].struct.field('_raw')[0]
        print(f"  Raw response: {raw}")

        print(f"  Title: {result_df['recommendation'].struct.field('title')[0]}")
        print(f"  Genre: {result_df['recommendation'].struct.field('genre')[0]}")
        print(f"  Year: {result_df['recommendation'].struct.field('year')[0]}")
        print(f"  Reason: {result_df['recommendation'].struct.field('reason')[0]}")

        # Check for errors
        error = result_df['recommendation'].struct.field('_error')[0]
        print(f"  Error field: {error}")
        if error:
            print(f"\n⚠️  Error occurred: {error}")
            details = result_df['recommendation'].struct.field('_details')[0]
            print(f"  Details: {details}")
        else:
            print(f"\n✓ Successfully returned structured output!")

    except Exception as e:
        print(f"\n✗ Error accessing struct fields: {e}")
        print(f"  Result type: {type(result_df['recommendation'])}")
        print(f"  Result dtype: {result_df['recommendation'].dtype}")


class PersonInfo(BaseModel):
    """Information about a person."""
    name: str
    age: int
    occupation: str
    hobbies: List[str]


def test_structured_output_multiple_rows():
    """Test structured output with multiple rows."""
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping test: GROQ_API_KEY not set")
        return

    # Create a test dataframe with multiple rows
    df = pl.DataFrame({
        "prompt": [
            "Generate info for a fictional character named Alice, a 28-year-old software engineer who likes hiking and reading",
            "Generate info for a fictional character named Bob, a 45-year-old chef who enjoys cooking and gardening",
        ]
    })

    # Run inference with structured output
    result_df = df.with_columns(
        person_info=inference_async(
            pl.col("prompt"),
            provider=Provider.GROQ,
            model="openai/gpt-oss-120b",
            response_model=PersonInfo
        )
    )

    print("\nMultiple Rows Test Results (Struct):")
    print(result_df)

    # Access struct fields directly for all rows!
    print("\n✨ Direct struct field access for all rows:")
    try:
        names = result_df['person_info'].struct.field('name')
        ages = result_df['person_info'].struct.field('age')
        occupations = result_df['person_info'].struct.field('occupation')
        hobbies = result_df['person_info'].struct.field('hobbies')

        for idx in range(len(result_df)):
            print(f"\n--- Row {idx} ---")
            print(f"  Name: {names[idx]}")
            print(f"  Age: {ages[idx]}")
            print(f"  Occupation: {occupations[idx]}")
            print(f"  Hobbies: {hobbies[idx]}")

            # Check for errors
            error = result_df['person_info'].struct.field('_error')[idx]
            if error:
                print(f"  ⚠️  Error: {error}")
            else:
                print(f"  ✓ Valid")

    except Exception as e:
        print(f"\n✗ Error accessing struct fields: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Structured Outputs with Pydantic")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("Test 1: Basic Structured Output")
    print("=" * 60)
    test_structured_output_basic()

    print("\n" + "=" * 60)
    print("Test 2: Multiple Rows")
    print("=" * 60)
    test_structured_output_multiple_rows()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
