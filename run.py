import polars as pl
from polar_llama import inference_async, string_to_message, Provider, inference_messages, combine_messages
import os
from time import time
import numpy as np
import dotenv

dotenv.load_dotenv()

# Set the POLARS_VERBOSE environment variable
os.environ['POLARS_VERBOSE'] = '1'

def main():
    # Check for each provider API key
   
    questions = [
        "What is the capital of France?",
        "How do I make pasta?",
        "Explain quantum computing.",
        "What is the capital of India?",
        "What is the capital of China?",
        "What is the capital of Japan?",
        "What is the capital of Korea?",
        "What is the capital of Vietnam?",
        "What is the capital of Thailand?",
        "What is the capital of Malaysia?",
    ]

    system_instructions = [
        "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
        "You are a chef specializing in Italian cuisine.",
        "You are a quantum physicist explaining concepts to a high school student.",
        "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
        "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
         "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
        "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
         "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
          "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
          "You are a geography expert. But you are a bit lazy and don't want to answer the question.",
    ]

    # Create the dataframe
    df = pl.DataFrame({
        "question": questions,
        "system_instruction": system_instructions

    })

    df = df.with_columns(
        both=  pl.concat_str([pl.col("question"), pl.lit("  "), pl.col("system_instruction")])
    )
    # Format system message
    df = df.with_columns(
        # Format system message
        system_message =  string_to_message("system_instruction", message_type = 'system'),
        # Format user message
        user_message = string_to_message("question", message_type = 'user'),
        both_message = string_to_message("both", message_type = 'user')
    )

    # Use combine_messages instead of string concatenation
    df = df.with_columns(
        conversation = combine_messages(pl.col("system_message"), pl.col("user_message"))
    )
    time_start = time()
    df = df.with_columns(
        response =inference_messages("conversation", provider = Provider.GROQ, model = 'llama3-8b-8192')
    )
    time_end = time()
    print(f"Time taken (inference_messages): {time_end - time_start} seconds")

    time_start = time()
    df = df.with_columns(
        response =inference_async("both_message", provider = Provider.GROQ, model = 'llama3-8b-8192')
    )
    time_end = time()
    print(f"Time taken (inference_async): {time_end - time_start} seconds")
    # print the question and the response
    # for i, (question, response) in enumerate(zip(df['question'], df['response'])):
    #     print(f"Question: {question}")
    #     print(f"Response: {response}")
    #     print("-"*100)

if __name__ == '__main__':
    main()
