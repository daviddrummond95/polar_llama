import polars as pl
from polar_llama import tag_taxonomy, Provider
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def test_taxonomy_tagging_basic():
    """Test basic taxonomy tagging with a simple two-field taxonomy."""
    # Check if API key is available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping test: ANTHROPIC_API_KEY not set")
        return

    # Define a simple taxonomy
    taxonomy = {
        "sentiment": {
            "description": "The emotional tone of the text",
            "values": {
                "positive": "Text expresses positive emotions, optimism, or favorable opinions",
                "negative": "Text expresses negative emotions, pessimism, or unfavorable opinions",
                "neutral": "Text is factual and objective without clear emotional content"
            }
        },
        "urgency": {
            "description": "How urgent or time-sensitive the content is",
            "values": {
                "high": "Requires immediate attention or action",
                "medium": "Should be addressed soon but not immediately critical",
                "low": "Can be addressed at any convenient time"
            }
        }
    }

    # Create a test dataframe
    df = pl.DataFrame({
        "id": [1],
        "document": ["URGENT: The server is down and customers can't access the site!"]
    })

    # Apply taxonomy tagging
    result_df = df.with_columns(
        tags=tag_taxonomy(
            pl.col("document"),
            taxonomy,
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4-5-20250929"
        )
    )

    print("\nTest Results:")
    print(result_df)

    # Access nested struct fields
    print("\n✨ Accessing taxonomy tag fields:")
    try:
        # Access sentiment field
        print("\n--- Sentiment ---")
        sentiment_thinking = result_df['tags'].struct.field('sentiment').struct.field('thinking')[0]
        sentiment_reflection = result_df['tags'].struct.field('sentiment').struct.field('reflection')[0]
        sentiment_value = result_df['tags'].struct.field('sentiment').struct.field('value')[0]
        sentiment_confidence = result_df['tags'].struct.field('sentiment').struct.field('confidence')[0]

        print(f"  Thinking: {sentiment_thinking}")
        print(f"  Reflection: {sentiment_reflection}")
        print(f"  Value: {sentiment_value}")
        print(f"  Confidence: {sentiment_confidence}")

        # Access urgency field
        print("\n--- Urgency ---")
        urgency_thinking = result_df['tags'].struct.field('urgency').struct.field('thinking')[0]
        urgency_reflection = result_df['tags'].struct.field('urgency').struct.field('reflection')[0]
        urgency_value = result_df['tags'].struct.field('urgency').struct.field('value')[0]
        urgency_confidence = result_df['tags'].struct.field('urgency').struct.field('confidence')[0]

        print(f"  Thinking: {urgency_thinking}")
        print(f"  Reflection: {urgency_reflection}")
        print(f"  Value: {urgency_value}")
        print(f"  Confidence: {urgency_confidence}")

        # Check for errors
        error = result_df['tags'].struct.field('_error')[0]
        if error:
            print(f"\n⚠️  Error occurred: {error}")
            details = result_df['tags'].struct.field('_details')[0]
            print(f"  Details: {details}")
        else:
            print(f"\n✓ Successfully tagged document with taxonomy!")

    except Exception as e:
        print(f"\n✗ Error accessing struct fields: {e}")
        print(f"  Result type: {type(result_df['tags'])}")
        print(f"  Result dtype: {result_df['tags'].dtype}")


def test_taxonomy_tagging_multiple_rows():
    """Test taxonomy tagging with multiple documents in parallel."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping test: ANTHROPIC_API_KEY not set")
        return

    # Define taxonomy
    taxonomy = {
        "sentiment": {
            "description": "The emotional tone of the text",
            "values": {
                "positive": "Text expresses positive emotions, optimism, or favorable opinions",
                "negative": "Text expresses negative emotions, pessimism, or unfavorable opinions",
                "neutral": "Text is factual and objective without clear emotional content"
            }
        },
        "topic": {
            "description": "The main subject matter of the text",
            "values": {
                "technology": "Discusses technology, software, hardware, or digital topics",
                "business": "Discusses business, finance, economics, or corporate matters",
                "personal": "Discusses personal experiences, emotions, or individual matters"
            }
        }
    }

    # Create a test dataframe with multiple rows
    df = pl.DataFrame({
        "id": [1, 2, 3],
        "document": [
            "Our quarterly revenue exceeded expectations by 25%! The team did an amazing job.",
            "The new software update introduces significant performance improvements and bug fixes.",
            "I'm feeling really grateful for all the support I've received from my friends and family."
        ]
    })

    # Apply taxonomy tagging (processes rows in parallel)
    result_df = df.with_columns(
        tags=tag_taxonomy(
            pl.col("document"),
            taxonomy,
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4-5-20250929"
        )
    )

    print("\nMultiple Rows Test Results:")
    print(result_df)

    # Create a clean view with just the selected values and confidence
    print("\n✨ Simplified view of tagged results:")
    try:
        view_df = result_df.select([
            "id",
            "document",
            pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
            pl.col("tags").struct.field("sentiment").struct.field("confidence").alias("sentiment_conf"),
            pl.col("tags").struct.field("topic").struct.field("value").alias("topic"),
            pl.col("tags").struct.field("topic").struct.field("confidence").alias("topic_conf")
        ])

        print(view_df)

        # Show detailed thinking for first row
        print("\n--- Detailed thinking for first row ---")
        sentiment_thinking = result_df['tags'].struct.field('sentiment').struct.field('thinking')[0]
        print(f"Sentiment thinking: {sentiment_thinking}")

        topic_thinking = result_df['tags'].struct.field('topic').struct.field('thinking')[0]
        print(f"Topic thinking: {topic_thinking}")

        print(f"\n✓ Successfully tagged {len(result_df)} documents in parallel!")

    except Exception as e:
        print(f"\n✗ Error processing results: {e}")


def test_taxonomy_tagging_complex():
    """Test taxonomy tagging with a more complex multi-value taxonomy."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping test: ANTHROPIC_API_KEY not set")
        return

    # Define a more complex taxonomy with more fields and values
    taxonomy = {
        "tone": {
            "description": "The overall tone and style of communication",
            "values": {
                "formal": "Professional, business-like, and official tone",
                "casual": "Informal, conversational, and relaxed tone",
                "technical": "Specialized, jargon-heavy, and precise tone",
                "friendly": "Warm, approachable, and personable tone"
            }
        },
        "intent": {
            "description": "The primary purpose or goal of the text",
            "values": {
                "inform": "Primarily aims to share information or educate",
                "persuade": "Aims to convince or change opinions",
                "request": "Asks for something or seeks assistance",
                "express": "Shares feelings, opinions, or personal thoughts"
            }
        },
        "audience": {
            "description": "The intended target audience",
            "values": {
                "general": "Broad audience with no specialized knowledge",
                "technical": "Audience with technical expertise",
                "business": "Business professionals or stakeholders",
                "academic": "Scholars, researchers, or students"
            }
        }
    }

    # Create a test dataframe
    df = pl.DataFrame({
        "id": [1],
        "document": [
            "Dear Team, I wanted to share some exciting news about our recent performance metrics. "
            "Our system throughput has increased by 40% following the implementation of the new caching "
            "layer and optimized query execution pipeline. This improvement is particularly significant "
            "given the concurrent user load we've been experiencing."
        ]
    })

    # Apply taxonomy tagging
    result_df = df.with_columns(
        tags=tag_taxonomy(
            pl.col("document"),
            taxonomy,
            provider=Provider.ANTHROPIC,
            model="claude-sonnet-4-5-20250929"
        )
    )

    print("\nComplex Taxonomy Test Results:")

    # Create a clean view
    try:
        view_df = result_df.select([
            pl.col("tags").struct.field("tone").struct.field("value").alias("tone"),
            pl.col("tags").struct.field("tone").struct.field("confidence").alias("tone_conf"),
            pl.col("tags").struct.field("intent").struct.field("value").alias("intent"),
            pl.col("tags").struct.field("intent").struct.field("confidence").alias("intent_conf"),
            pl.col("tags").struct.field("audience").struct.field("value").alias("audience"),
            pl.col("tags").struct.field("audience").struct.field("confidence").alias("audience_conf")
        ])

        print(view_df)

        # Show reflections for each field
        print("\n--- Field Reflections ---")
        tone_reflection = result_df['tags'].struct.field('tone').struct.field('reflection')[0]
        print(f"Tone: {tone_reflection}")

        intent_reflection = result_df['tags'].struct.field('intent').struct.field('reflection')[0]
        print(f"Intent: {intent_reflection}")

        audience_reflection = result_df['tags'].struct.field('audience').struct.field('reflection')[0]
        print(f"Audience: {audience_reflection}")

        print(f"\n✓ Successfully tagged with complex taxonomy!")

    except Exception as e:
        print(f"\n✗ Error processing results: {e}")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Taxonomy-based Tagging")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("Test 1: Basic Taxonomy Tagging")
    print("=" * 80)
    test_taxonomy_tagging_basic()

    print("\n" + "=" * 80)
    print("Test 2: Multiple Rows (Parallel Processing)")
    print("=" * 80)
    test_taxonomy_tagging_multiple_rows()

    print("\n" + "=" * 80)
    print("Test 3: Complex Multi-field Taxonomy")
    print("=" * 80)
    test_taxonomy_tagging_complex()

    print("\n" + "=" * 80)
    print("All taxonomy tagging tests completed!")
    print("=" * 80)
