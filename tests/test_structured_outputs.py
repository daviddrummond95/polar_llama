import polars as pl
from polar_llama import inference_async, Provider
from pydantic import BaseModel
import os
import json


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
            model="llama-3.3-70b-versatile",
            response_model=MovieRecommendation
        )
    )

    print("\nTest Results:")
    print(result_df)

    # Get the result
    recommendation_json = result_df["recommendation"][0]
    print(f"\nRaw response:\n{recommendation_json}")

    # Try to parse as JSON
    try:
        parsed = json.loads(recommendation_json)
        print(f"\nParsed recommendation:")
        print(f"  Title: {parsed.get('title')}")
        print(f"  Genre: {parsed.get('genre')}")
        print(f"  Year: {parsed.get('year')}")
        print(f"  Reason: {parsed.get('reason')}")

        # Check if it's a valid response or an error
        if "error" in parsed:
            print(f"\n⚠️  Error occurred: {parsed['error']}")
            print(f"  Details: {parsed.get('details')}")
            if "raw" in parsed:
                print(f"  Raw response: {parsed['raw']}")
        else:
            # Validate with Pydantic
            movie = MovieRecommendation(**parsed)
            print(f"\n✓ Successfully validated with Pydantic!")
            print(f"  Movie: {movie.title} ({movie.year})")
            print(f"  Genre: {movie.genre}")
            print(f"  Reason: {movie.reason}")
    except json.JSONDecodeError as e:
        print(f"\n✗ Failed to parse as JSON: {e}")
        print(f"  Response: {recommendation_json}")
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")


class PersonInfo(BaseModel):
    """Information about a person."""
    name: str
    age: int
    occupation: str
    hobbies: list[str]


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
            model="llama-3.3-70b-versatile",
            response_model=PersonInfo
        )
    )

    print("\nMultiple Rows Test Results:")
    print(result_df)

    # Process each result
    for idx, person_json in enumerate(result_df["person_info"]):
        print(f"\n--- Row {idx} ---")
        print(f"Raw response: {person_json}")

        try:
            parsed = json.loads(person_json)

            if "error" in parsed:
                print(f"⚠️  Error: {parsed['error']}")
                if "details" in parsed:
                    print(f"   Details: {parsed['details']}")
            else:
                person = PersonInfo(**parsed)
                print(f"✓ Valid: {person.name}, {person.age}, {person.occupation}")
                print(f"  Hobbies: {', '.join(person.hobbies)}")
        except Exception as e:
            print(f"✗ Failed: {e}")


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
