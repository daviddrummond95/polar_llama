import polars as pl
from polar_llama import inference_async, string_to_message, Provider
import os
from time import time
import numpy as np
import dotenv

dotenv.load_dotenv()

# Set the POLARS_VERBOSE environment variable
os.environ['POLARS_VERBOSE'] = '1'

def main():
    # Check for each provider API key
    for provider in ['OPENAI', 'GROQ', 'ANTHROPIC', 'GEMINI']:
        api_key = os.environ.get(provider+"_API_KEY")
        if not api_key:
            print(f"Error: {provider} API key is not set.")
            return

    print("PolarLlama - OpenAI Integration for Polars")
    print("------------------------------------------")
    
    # Sample questions for demonstration
    questions = [
        'What is the capital of France?',
        'Who are the founders of OpenAI?',
        'What are the advantages of polars vs pandas?',
        'Write a simple script in python that takes a list of numbers and returns the sum of the numbers.',
        'What is the capital of India?',
        'How many continents are there in the world?',
        "How old was George Washington when he died?",
        "What is the capital of the United States?",
        "What is the capital of the United Kingdom?",
        "What is the capital of the United Arab Emirates?",
        "Who was the the winner of the first American Idol?",
        "When did the Great British Bakeoff first air?",
        "Who was the first prime minister of the UK?",
        "What year did Canada become a country?",
        "What is the capital of Australia?",
        "What is the capital of New Zealand?",
        'What is the deepest part of the ocean?',
        'Who won the Nobel Prize in Physics in 2020?',
        'Where is the tallest building in the world located?',
        'What are the health benefits of eating apples?',
        'Write a Python script that reverses a string.',
        'What is the smallest planet in our solar system?',
        'How many states are there in the United States?',
        'Who was the first woman to fly solo across the Atlantic Ocean?',
        'What is the capital of Sweden?',
        'What are the main ingredients in a margarita cocktail?',
        'What year was the United Nations founded?',
        'Who is the CEO of Tesla?',
        'What is the main export of Brazil?',
        'What are the symptoms of vitamin D deficiency?',
        'When was the first computer virus discovered?',
        'What is the melting point of gold?',
        'What is the capital of Brazil?',
        'How does quantum computing work?',
        'What is the main function of the kidneys?',
        'Write a Python function that checks if a number is a palindrome.',
        'What is the birthplace of Shakespeare?',
        'Who discovered penicillin?',
        'When was the Louvre Museum established?',
        'What are the three branches of government in the United States?',
        'How long is the Great Wall of China?',
        'Who directed the movie "Inception"?',
        'What are the benefits of meditation?',
        'Who invented the light bulb?',
        'What causes earthquakes?',
        'How many Oscars did the movie "Titanic" win?',
        'What is the largest animal in the world?',
        'How do solar panels work?',
        'What is the official language of Brazil?',
        'What is the life expectancy in Canada?',
        'Describe the process of evaporation.',
        'Who wrote the musical "Hamilton"?',
        'What are the primary colors?',
        'What is the deadliest animal in the world?',
        'What year was the first email sent?',
        'What is the capital of Egypt?',
        'Who won the NBA championship in 2021?',
        'Where is the oldest university in the world?',
        'What are the main uses of silicon in technology?',
        'Write a Python script that finds the factorial of a number.',
        'What is the diameter of Earth?',
        'How many languages are spoken in India?',
        'Who was the first president of the United States?',
        'What is the capital of Thailand?',
        'What is the main ingredient in sushi rice?',
        'What year did the Berlin Wall fall?',
        'Who is the author of "Pride and Prejudice"?',
        'What is the largest desert on Earth?',
        'What are the symptoms of dehydration?',
        'When was the camera invented?',
        'What is the freezing point of mercury?',
        'What is the capital of Colombia?',
        'What principles govern blockchain technology?',
        'What is the primary source of energy for the Earth?',
        'Write a Python function that detects if a word is an anagram.',
        'Where was Leonardo da Vinci born?',
        'Who discovered the structure of DNA?',
        'When was the Eiffel Tower completed?',
        'What are the major exports of Germany?',
        'How far is Mars from Earth?',
        'Who directed the movie "Titanic"?',
        'What are the pros and cons of intermittent fasting?',
        'Who invented the first car?',
        'What causes the northern lights?',
        'How many Grammy Awards has Beyoncé won?',
        'What is the tallest mountain in North America?',
        'How does wind energy work?',
        'What is the national sport of Japan?',
        'What is the typical lifespan of a house cat?',
        'Describe the process of nuclear fission.',
        'Who composed the music for "Star Wars"?',
        'What is the hardest natural substance on Earth?',
        'What is the most spoken language in the world?',
        'What year was the first smartphone released?'
    ]

    # Create a DataFrame with the questions
    df = pl.DataFrame({'Questions': questions})
    
    print(f"\nProcessing {len(questions)} questions...")
    
    # Convert questions to messages
    df = df.with_columns(
        prompt = string_to_message("Questions", message_type = 'user'),
    )
    start_time = time()
    # Run inference
    df = df.with_columns(
        openai_answer = inference_async('prompt', provider = Provider.OPENAI, model = 'gpt-4o-mini'),
        groq_answer = inference_async('prompt', provider = Provider.GROQ, model = 'llama3-8b-8192'),
        anthropic_answer = inference_async('prompt', provider = Provider.ANTHROPIC, model = 'claude-3-5-sonnet-20240620'),
        gemini_answer = inference_async('prompt', provider = Provider.GEMINI, model = 'gemini-2.0-flash-lite')
    )
    end_time = time()
    
    
    # Display results
    print("\nResults:")
    print("--------")
    for i, (question, openai_answer, groq_answer, anthropic_answer, gemini_answer) in enumerate(zip(df['Questions'][0:2], df['openai_answer'][0:2], df['groq_answer'][0:2], df['anthropic_answer'][0:2], df['gemini_answer'][0:2])):
        print(f"\nQ{i+1}: {question}")
        print(f"OpenAI Answer: {openai_answer}")
        print(f"Groq Answer: {groq_answer}")
        print(f"Anthropic Answer: {anthropic_answer}")
        print(f"Gemini Answer: {gemini_answer}")
    print(f"Time taken: {end_time - start_time} seconds")
if __name__ == '__main__':
    main()
