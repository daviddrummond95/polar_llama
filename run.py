from matplotlib.cbook import print_cycles
import polars as pl
from polar_llama import inference_async, string_to_message
import os
from time import time
import numpy as np
import matplotlib.pyplot as plt


# Set the POLARS_VERBOSE environment variable
os.environ['POLARS_VERBOSE'] = '1'


def time_function(func, dataframe, column_name, print_output=False):
    """Measure the time taken by a function to execute."""
    start = time()
    result = dataframe.with_columns(
        pig_latin = func(column_name)
    )
    if print_output:
        print('Result:', result.to_pandas()['pig_latin'][0])
    end = time()
    return end - start, result

def run_experiments(questions, num_runs=1):
    """Run synchronous and asynchronous inference and collect timing data."""
    # Prepare data structures to store times
    sync_times = []
    async_times = []
    question_counts = list(range(5, len(questions), 10))

    # Loop over different question set sizes
    for count in question_counts:
        print(f'Running experiments for {count} questions')
        sync_run_times = []
        async_run_times = []


        # Perform multiple runs to average the results
        for run in range(num_runs):
            df = pl.DataFrame({'Questions': questions[:count]})

            # # Time synchronous function
            # time_taken, _ = time_function(inference, df, 'Questions', print_output)
            # sync_run_times.append(time_taken)
            # print(f'Synchronous run time: {time_taken:.2f} seconds')

            # Time asynchronous function
            time_taken, _ = time_function(inference_async, df, 'Questions')
            async_run_times.append(time_taken)
            print(f'Asynchronous run time: {time_taken:.2f} seconds')

        # Compute average time for current number of questions
        # sync_times.append(np.mean(sync_run_times))
        # async_times.append(np.mean(async_run_times))
        # # print(f'Average synchronous time: {sync_times[-1]:.2f} seconds')
        # print(f'Average asynchronous time: {async_times[-1]:.2f} seconds')

    return question_counts, sync_times, async_times

def plot_results(question_counts, sync_times, async_times):
    """Plot the results of the experiments."""
    plt.figure(figsize=(10, 5))
    # plt.plot(question_counts, sync_times, label='Synchronous', marker='o')
    plt.plot(question_counts, async_times, label='Asynchronous', marker='o')
    plt.xlabel('Number of Questions')
    plt.ylabel('Average Time Taken (seconds)')
    plt.title('Performance of Asynchronous Inference Based on Number of Questions')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    questions = ['What is the capital of France?',
        'Who are the founders of OpenAI?',
        'Where is the global headquarters of Fractal Analytics?',
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

    df = pl.DataFrame({'Questions': questions[0:10]})

    df = df.with_columns(
        prompt = string_to_message("Questions", message_type = 'user'),
    )
    df = df.with_columns(
        answer = inference_async('prompt')
    )

    print(df.to_pandas()['answer'][0])

if __name__ == '__main__':
    main()
