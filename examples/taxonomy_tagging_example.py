"""
Taxonomy-based Tagging Example
==============================

This example demonstrates how to use polar_llama's taxonomy-based tagging feature
to classify documents with detailed reasoning and confidence scores.

The taxonomy tagging system allows you to:
1. Define custom taxonomies with fields and values
2. Get detailed reasoning for each classification decision
3. Process multiple documents in parallel
4. Extract specific tags and confidence scores for analysis
"""

import polars as pl
from polar_llama import tag_taxonomy, Provider
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_basic_taxonomy():
    """
    Example 1: Basic taxonomy with sentiment and urgency classification.

    This shows how to define a simple taxonomy and apply it to documents.
    """
    print("\n" + "=" * 80)
    print("Example 1: Basic Taxonomy Tagging")
    print("=" * 80)

    # Define a taxonomy for customer support tickets
    taxonomy = {
        "sentiment": {
            "description": "The emotional tone of the customer message",
            "values": {
                "positive": "Customer is satisfied, happy, or expressing gratitude",
                "negative": "Customer is frustrated, angry, or disappointed",
                "neutral": "Customer is simply stating facts without clear emotion"
            }
        },
        "urgency": {
            "description": "How urgently the issue needs to be addressed",
            "values": {
                "high": "Critical issue requiring immediate attention",
                "medium": "Important issue that should be addressed soon",
                "low": "Minor issue or general inquiry that can wait"
            }
        }
    }

    # Sample customer support tickets
    df = pl.DataFrame({
        "ticket_id": [1, 2, 3],
        "message": [
            "URGENT: Our production server has been down for 2 hours! We're losing revenue!",
            "Thanks for the quick response yesterday. Just following up on the timeline.",
            "I noticed a small typo in the documentation. Not urgent, but wanted to let you know."
        ]
    })

    print("\nOriginal Data:")
    print(df)

    # Apply taxonomy tagging
    print("\n📊 Applying taxonomy tagging...")
    result_df = df.with_columns(
        tags=tag_taxonomy(
            pl.col("message"),
            taxonomy,
            provider=Provider.GROQ,
            model="openai/gpt-oss-120b"
        )
    )

    # Extract just the values and confidence for easy viewing
    analysis_df = result_df.select([
        "ticket_id",
        "message",
        pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
        pl.col("tags").struct.field("sentiment").struct.field("confidence").alias("sentiment_confidence"),
        pl.col("tags").struct.field("urgency").struct.field("value").alias("urgency"),
        pl.col("tags").struct.field("urgency").struct.field("confidence").alias("urgency_confidence"),
    ])

    print("\n✨ Classification Results:")
    print(analysis_df)

    # Show detailed reasoning for the first ticket
    print("\n🔍 Detailed Analysis for Ticket 1:")
    print(f"Message: {df['message'][0]}")
    print(f"\nSentiment thinking:")
    print(result_df['tags'].struct.field('sentiment').struct.field('thinking')[0])
    print(f"\nSentiment reflection:")
    print(result_df['tags'].struct.field('sentiment').struct.field('reflection')[0])
    print(f"\nUrgency thinking:")
    print(result_df['tags'].struct.field('urgency').struct.field('thinking')[0])
    print(f"\nUrgency reflection:")
    print(result_df['tags'].struct.field('urgency').struct.field('reflection')[0])


def example_content_classification():
    """
    Example 2: Content classification taxonomy.

    This shows a more detailed taxonomy for classifying blog posts or articles.
    """
    print("\n" + "=" * 80)
    print("Example 2: Content Classification")
    print("=" * 80)

    # Define a taxonomy for content classification
    taxonomy = {
        "category": {
            "description": "The primary topic or category of the content",
            "values": {
                "technology": "Content about software, hardware, programming, or digital innovation",
                "business": "Content about entrepreneurship, management, finance, or corporate strategy",
                "lifestyle": "Content about health, wellness, personal development, or daily life",
                "science": "Content about scientific research, discoveries, or methodology"
            }
        },
        "content_type": {
            "description": "The format and purpose of the content",
            "values": {
                "tutorial": "Step-by-step instructional content teaching how to do something",
                "analysis": "In-depth examination or critical thinking about a topic",
                "news": "Timely reporting of recent events or developments",
                "opinion": "Personal viewpoint or commentary on a subject"
            }
        },
        "expertise_level": {
            "description": "The level of prior knowledge assumed of the reader",
            "values": {
                "beginner": "Assumes little to no prior knowledge, explains basic concepts",
                "intermediate": "Assumes some familiarity with the topic",
                "advanced": "Assumes significant expertise and uses specialized terminology"
            }
        }
    }

    # Sample blog post excerpts
    df = pl.DataFrame({
        "post_id": [1, 2],
        "content": [
            "In this guide, we'll walk through setting up your first Python environment. "
            "Don't worry if you've never programmed before - we'll start from the very beginning. "
            "First, let's download Python from the official website...",

            "The recent advances in transformer architecture have fundamentally changed how we "
            "approach sequence-to-sequence tasks. By leveraging self-attention mechanisms, "
            "these models can capture long-range dependencies more effectively than traditional RNNs..."
        ]
    })

    print("\nOriginal Data:")
    print(df)

    # Apply taxonomy tagging
    print("\n📊 Applying taxonomy tagging...")
    result_df = df.with_columns(
        tags=tag_taxonomy(
            pl.col("content"),
            taxonomy,
            provider=Provider.GROQ,
            model="openai/gpt-oss-120b"
        )
    )

    # Extract classification results
    analysis_df = result_df.select([
        "post_id",
        pl.col("tags").struct.field("category").struct.field("value").alias("category"),
        pl.col("tags").struct.field("content_type").struct.field("value").alias("type"),
        pl.col("tags").struct.field("expertise_level").struct.field("value").alias("level"),
        # Include confidence scores
        pl.col("tags").struct.field("category").struct.field("confidence").alias("category_conf"),
        pl.col("tags").struct.field("content_type").struct.field("confidence").alias("type_conf"),
        pl.col("tags").struct.field("expertise_level").struct.field("confidence").alias("level_conf"),
    ])

    print("\n✨ Classification Results:")
    print(analysis_df)


def example_email_routing():
    """
    Example 3: Email routing and prioritization.

    This shows how to use taxonomy tagging for automated email triage.
    """
    print("\n" + "=" * 80)
    print("Example 3: Email Routing and Prioritization")
    print("=" * 80)

    # Define a taxonomy for email routing
    taxonomy = {
        "department": {
            "description": "Which department should handle this email",
            "values": {
                "sales": "Inquiries about products, pricing, or purchasing",
                "support": "Technical issues, bug reports, or help requests",
                "billing": "Questions about invoices, payments, or accounts",
                "general": "General information requests or other inquiries"
            }
        },
        "priority": {
            "description": "How quickly this email should be addressed",
            "values": {
                "urgent": "Requires immediate response (service down, payment issues, etc.)",
                "high": "Should be responded to within a few hours",
                "normal": "Can be responded to within 24 hours",
                "low": "Non-time-sensitive, can be handled when convenient"
            }
        },
        "intent": {
            "description": "What the sender wants to accomplish",
            "values": {
                "question": "Asking for information or clarification",
                "complaint": "Expressing dissatisfaction or reporting a problem",
                "request": "Asking for a specific action to be taken",
                "feedback": "Providing suggestions or general feedback"
            }
        }
    }

    # Sample emails
    df = pl.DataFrame({
        "email_id": [1, 2, 3, 4],
        "subject": [
            "Payment failed - cannot access account",
            "Feature request: dark mode",
            "How much does the enterprise plan cost?",
            "Login not working - URGENT"
        ],
        "body": [
            "Hi, my payment was declined and now I can't access my account. Please help!",
            "Love your product! Would be great to have a dark mode option for the interface.",
            "We're interested in the enterprise plan for our team of 50. What's the pricing?",
            "I can't log in to my account and I have a presentation in 30 minutes!"
        ]
    })

    print("\nOriginal Emails:")
    print(df.select(["email_id", "subject"]))

    # Combine subject and body for better classification
    df = df.with_columns(
        full_message=pl.concat_str([
            pl.lit("Subject: "), pl.col("subject"),
            pl.lit("\n\nBody: "), pl.col("body")
        ])
    )

    # Apply taxonomy tagging
    print("\n📊 Applying taxonomy tagging...")
    result_df = df.with_columns(
        tags=tag_taxonomy(
            pl.col("full_message"),
            taxonomy,
            provider=Provider.GROQ,
            model="openai/gpt-oss-120b"
        )
    )

    # Create routing table
    routing_df = result_df.select([
        "email_id",
        "subject",
        pl.col("tags").struct.field("department").struct.field("value").alias("route_to"),
        pl.col("tags").struct.field("priority").struct.field("value").alias("priority"),
        pl.col("tags").struct.field("intent").struct.field("value").alias("intent"),
        pl.col("tags").struct.field("department").struct.field("confidence").alias("confidence"),
    ])

    print("\n✨ Email Routing Table:")
    print(routing_df)

    # Filter high-priority items
    urgent_emails = routing_df.filter(
        (pl.col("priority") == "urgent") | (pl.col("priority") == "high")
    )

    print("\n🚨 High-Priority Emails:")
    print(urgent_emails)


def example_advanced_filtering():
    """
    Example 4: Advanced filtering and analysis.

    This shows how to use confidence scores and combine multiple taxonomy fields
    for sophisticated filtering.
    """
    print("\n" + "=" * 80)
    print("Example 4: Advanced Filtering with Confidence Scores")
    print("=" * 80)

    taxonomy = {
        "sentiment": {
            "description": "Overall sentiment",
            "values": {
                "positive": "Positive sentiment",
                "negative": "Negative sentiment",
                "neutral": "Neutral sentiment"
            }
        },
        "actionable": {
            "description": "Whether the item requires action",
            "values": {
                "yes": "Requires follow-up or action",
                "no": "Informational only, no action needed"
            }
        }
    }

    # Sample feedback messages
    df = pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "feedback": [
            "The new feature is amazing! Works perfectly.",
            "I think there might be a bug, but I'm not sure. Sometimes it works, sometimes it doesn't.",
            "Critical bug: data loss when clicking save multiple times!",
            "Just wanted to say thanks for the great support.",
            "The UI is confusing and I can't find the export button."
        ]
    })

    # Apply taxonomy tagging
    result_df = df.with_columns(
        tags=tag_taxonomy(
            pl.col("feedback"),
            taxonomy,
            provider=Provider.GROQ,
            model="openai/gpt-oss-120b"
        )
    )

    # Extract tags and confidence
    analysis_df = result_df.select([
        "id",
        "feedback",
        pl.col("tags").struct.field("sentiment").struct.field("value").alias("sentiment"),
        pl.col("tags").struct.field("sentiment").struct.field("confidence").alias("sent_conf"),
        pl.col("tags").struct.field("actionable").struct.field("value").alias("actionable"),
        pl.col("tags").struct.field("actionable").struct.field("confidence").alias("action_conf"),
    ])

    print("\n✨ All Feedback with Tags:")
    print(analysis_df)

    # Filter for negative feedback that requires action with high confidence
    critical_items = analysis_df.filter(
        (pl.col("sentiment") == "negative") &
        (pl.col("actionable") == "yes") &
        (pl.col("action_conf") > 0.8)
    )

    print("\n🚨 Critical Issues (Negative + Actionable + High Confidence):")
    print(critical_items)

    # Filter for items where the model is uncertain (low confidence)
    uncertain_items = analysis_df.filter(
        (pl.col("sent_conf") < 0.7) | (pl.col("action_conf") < 0.7)
    )

    print("\n❓ Uncertain Classifications (May Need Human Review):")
    print(uncertain_items)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("⚠️  Please set GROQ_API_KEY environment variable to run these examples.")
        print("   You can create a .env file with: GROQ_API_KEY=your_key_here")
        exit(1)

    print("=" * 80)
    print("Polar Llama - Taxonomy-based Tagging Examples")
    print("=" * 80)
    print("\nThese examples demonstrate various use cases for taxonomy-based document tagging.")
    print("The tagging system processes documents in parallel and provides detailed reasoning,")
    print("reflections, selected values, and confidence scores for each classification.")

    # Run all examples
    example_basic_taxonomy()
    example_content_classification()
    example_email_routing()
    example_advanced_filtering()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\n📚 Key Takeaways:")
    print("  • Define custom taxonomies with fields and values")
    print("  • Get detailed reasoning for each classification decision")
    print("  • Process multiple documents in parallel automatically")
    print("  • Use confidence scores to filter uncertain classifications")
    print("  • Combine multiple taxonomy fields for sophisticated filtering")
    print("  • Extract nested struct fields using .struct.field() syntax")
