### README for Polar Llama

![Logo](PolarLlama.webp)

#### Overview

Polar Llama is a Python library designed to enhance the efficiency of making parallel inference calls to the ChatGPT API using the Polars dataframe tool. This library enables users to manage multiple API requests simultaneously, significantly speeding up the process compared to serial request handling.

#### Key Features

- **Parallel Inference**: Send multiple inference requests in parallel to the ChatGPT API without waiting for each individual request to complete.
- **Integration with Polars**: Utilizes the Polars dataframe for organizing and handling requests, leveraging its efficient data processing capabilities.
- **Easy to Use**: Simplifies the process of sending queries and retrieving responses from the ChatGPT API through a clean and straightforward interface.

#### Installation

To install Polar Llama, use will need to execute the folloiwng bash command:

```bash
maturin develop
```

I will be making the package available on PyPI soon.

#### Example Usage

Here’s how you can use Polar Llama to send multiple inference requests in parallel:

```python
import polars as pl
from polar_llama import string_to_message, inference_async

# Example questions
questions = [
    'What is the capital of France?',
    'What is the difference between polars and pandas?'
]

# Creating a dataframe with questions
df = pl.DataFrame({'Questions': questions})

# Adding prompts to the dataframe
df = df.with_columns(
    prompt=string_to_message("Questions", message_type='user')
)

# Sending parallel inference requests
df = df.with_columns(
    answer=inference_async('prompt')
)
```

#### Benefits

- **Speed**: Processes multiple queries in parallel, drastically reducing the time required for bulk query handling.
- **Scalability**: Scales efficiently with the increase in number of queries, ideal for high-demand applications.
- **Ease of Integration**: Integrates seamlessly into existing Python projects that utilize Polars, making it easy to add parallel processing capabilities.

#### Contributing

We welcome contributions to Polar Llama! If you're interested in improving the library or adding new features, please feel free to fork the repository and submit a pull request.

#### License

Polar Llama is released under the MIT license. For more details, see the LICENSE file in the repository.

---

This README provides a basic guide to getting started with Polar Llama, highlighting its key features and advantages for users needing high-performance inference capabilities.
