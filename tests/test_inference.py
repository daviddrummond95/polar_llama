import polars as pl
from polar_llama import inference_async, string_to_message, Provider
import dotenv

dotenv.load_dotenv()

def test_inference():
 
# Example questions
    questions = [
        'What is the capital of France? Respond with only the city name and no other text.',
        'What is the capital of India? Respond with only the city name and no other text.'
    ]

    # Creating a dataframe with questions
    df = pl.DataFrame({'Questions': questions})

    # Adding prompts to the dataframe
    df = df.with_columns(
        prompt=string_to_message("Questions", message_type='user')
    )

    # Sending parallel inference requests
    result = df.with_columns(
        answer=inference_async('prompt', provider = Provider.OPENAI, model = 'gpt-4o-mini')
    )
    expected_df = pl.DataFrame(
        {
            "Questions": ["What is the capital of France?", "What is the capital of India?"],
            "answer": ["Paris", "New Delhi"],
        }
    )

    assert result.equals(expected_df)
