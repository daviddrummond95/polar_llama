"""
Tool use, unrolled into the dataframe: the calorie-tracker example from
docs/design/MCP_TOOL_INTEGRATION.md.

Each meal log implies N food items that must be searched in a nutrition
database and reconciled. The pipeline is three explicit turns, each a column:

    meal text -> emit searches -> execute in parallel -> synthesize summary

Run with an OpenAI key in the environment. The "database" here is a local
Python executor so the example needs no MCP server; swap `executor=` for
`transport="http://localhost:8811/mcp"` to target a real one.
"""
import json

import dotenv
import polars as pl
from pydantic import BaseModel

from polar_llama import (
    Provider,
    combine_messages,
    execute_tool_calls,
    inference_messages,
    tool_results_to_message,
    tools_to_response_model,
)

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# 1. Tool definitions (could equally come from mcp_tools("http://...")).
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "search_food_db",
        "description": "Search a nutrition database for one food item and portion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The food item, singular"},
                "portion": {"type": "string", "description": "Portion as stated, e.g. 'two eggs'"},
            },
            "required": ["query", "portion"],
        },
    }
]

# A stand-in nutrition "database" the executor targets.
FAKE_DB = {
    "egg": 78, "sourdough toast": 120, "butter": 102, "black coffee": 2,
    "chicken bowl": 510, "guacamole": 230, "rice": 200,
}


def nutrition_executor(tool_name: str, arguments: dict):
    query = arguments["query"].lower()
    for food, calories in FAKE_DB.items():
        if food in query or query in food:
            return json.dumps({"food": food, "calories_per_serving": calories,
                               "portion": arguments["portion"]})
    return json.dumps({"food": query, "error": "not found"}), True


# ---------------------------------------------------------------------------
# 2. The pipeline.
# ---------------------------------------------------------------------------
class NutritionSummary(BaseModel):
    total_calories: int
    items: list[str]
    notes: str


meals = pl.DataFrame({"meal": [
    "two eggs, sourdough toast with butter, black coffee",
    "chipotle chicken bowl, no rice, extra guac",
]})

FoodSearches = tools_to_response_model(TOOLS)

result = (
    meals
    # Turn 1a: the LLM parameterizes N searches per row (structured output).
    .with_columns(
        calls=pl.col("meal").llama.inference_async(
            provider=Provider.OPENAI, model="gpt-4o-mini",
            response_model=FoodSearches,
        )
    )
    # Turn 1b: all searches across all rows execute concurrently.
    .with_columns(
        results=execute_tool_calls(pl.col("calls"), executor=nutrition_executor, tools=TOOLS)
    )
    # Turn 2: reconcile results into a typed summary.
    .with_columns(
        nutrition=inference_messages(
            combine_messages(
                pl.col("meal").llama.to_message(role="user"),
                tool_results_to_message(pl.col("results")),
            ),
            provider=Provider.OPENAI, model="gpt-4o-mini",
            response_model=NutritionSummary,
        )
    )
)

print(result.select("meal", "nutrition"))
print("\nEvery intermediate is a column:")
print(result.select(
    searches=pl.col("calls").struct.field("calls").list.len(),
    failed_lookups=pl.col("results").list.eval(
        pl.element().struct.field("is_error")
    ).list.sum(),
    total_calories=pl.col("nutrition").struct.field("total_calories"),
))
