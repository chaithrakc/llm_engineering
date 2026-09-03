
# LLM Engineering Course - Key Concepts & Code Syntax

## DAY 1: First Frontier LLM Project (Web Summarizer)

### Core Concept

Build a web scraper that summarizes websites using OpenAI's Chat Completions API.

### Key Imports

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from scripts.scraper import fetch_website_contents
from IPython.display import Markdown, display
```

### Essential Setup Pattern

```python
# Load environment variables
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Validate API key
if not api_key.startswith("sk-proj-"):
    print("Invalid API key format")

# Create OpenAI client
client = OpenAI()  # Uses OPENAI_API_KEY from .env by default
```

### Chat Completions API Structure

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "system_prompt_here"},
        {"role": "user", "content": "user_message_here"}
    ]
)

# Extract response
result = response.choices[0].message.content
```

### Display in Jupyter

```python
from IPython.display import Markdown, display

# Display markdown formatted text
display(Markdown(result))
```

### Common Pattern: Summarization

```python
def display_summary(url):
    # Step 1: Fetch website content
    website_content = fetch_website_contents(url)
  
    # Step 2: Create messages
    messages = [
        {"role": "system", "content": "Summarize the following webpage in 3 bullet points"},
        {"role": "user", "content": website_content}
    ]
  
    # Step 3: Call API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
  
    # Step 4: Display result
    display(Markdown(response.choices[0].message.content))

display_summary("https://example.com")
```

---

## DAY 2: Chat Completions API & OpenAI-Compatible Endpoints

### Core Concept

The Chat Completions API is the standard way to call LLMs. Other providers create "OpenAI-compatible endpoints" so you can use the same client library with different models.

### OpenAI Client Initialization

```python
from openai import OpenAI

# Default - uses OPENAI_API_KEY from environment
openai_client = OpenAI()

# Custom endpoint - Google Gemini
gemini_client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key="AIz..."  # Google API key
)

# Custom endpoint - Ollama (local)
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key='ollama'  # Can be anything for local
)
```

### Using OpenAI-Compatible Endpoints

```python
# All use the same syntax
response = gemini_client.chat.completions.create(
    model="gemini-3.1-flash-lite",
    messages=[
        {"role": "user", "content": "Tell me a fun fact"}
    ]
)
print(response.choices[0].message.content)
```

### Ollama Setup (Local Models)

```python
# Run in terminal first:
# $ ollama serve
# $ ollama pull llama3.2

ollama = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key='ollama'
)

# Call local model
response = ollama.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a fun fact"}]
)
print(response.choices[0].message.content)
```

### Key Providers

- **OpenAI**: `https://api.openai.com/v1` (gpt-4, gpt-3.5-turbo)
- **Google Gemini**: `https://generativelanguage.googleapis.com/v1beta/openai/` (gemini-3.1-flash-lite)
- **Ollama**: `http://localhost:11434/v1` (llama3.2, deepseek-r1:1.5b)

---

## DAY 4: Tokenization & Stateless LLM Behavior

### Core Concepts

#### 1. Tokenization

Tokens are the basic units that LLMs process. Not the same as words.

```python
import tiktoken

# Get tokenizer for a model
encoding = tiktoken.encoding_for_model("gpt-4")

# Encode text to tokens
text = "Hi my name is Ed and I like banoffee pie"
tokens = encoding.encode(text)
# Output: [12194, 922, 1308, 382, 6117, 326, 357, 1299, 9171, 26458, 5148]

# Decode tokens back to text
for token_id in tokens:
    token_text = encoding.decode([token_id])
    print(f"{token_id} = {token_text}")
```

**Key Insight**: "banoffee" splits into two tokens: "ban" + "offee"

#### 2. Stateless Nature of LLMs

**Every call to an LLM is completely independent.** The LLM has NO memory between calls.

```python
# First call
messages = [
    {"role": "user", "content": "Hi! I'm Ed!"}
]
response = client.chat.completions.create(model="gpt-4", messages=messages)
# Response: "Hi Ed! Nice to meet you."

# Second call - LLM doesn't remember!
messages = [
    {"role": "user", "content": "What's my name?"}
]
response = client.chat.completions.create(model="gpt-4", messages=messages)
# Response: "I don't know your name."
```

#### 3. Creating Illusion of Memory

To make LLM "remember" context, pass the entire conversation history every time:

```python
# Include full conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hi! I'm Ed!"},
    {"role": "assistant", "content": "Hi Ed! How can I assist you?"},
    {"role": "user", "content": "What's my name?"}
]

response = client.chat.completions.create(model="gpt-4", messages=messages)
# Response: "Your name is Ed!"
```

### Messages Structure

```python
messages = [
    # System role - gives model instructions
    {"role": "system", "content": "You are a helpful assistant"},
  
    # User role - user's message
    {"role": "user", "content": "Tell me a joke"},
  
    # Assistant role - model's previous response (for context)
    {"role": "assistant", "content": "Why did the chicken..."},
  
    # New user message
    {"role": "user", "content": "That's funny!"}
]
```

**Cost Implication**: Each call processes the entire message history, so longer conversations cost more.

---

## DAY 5: JSON Responses & Streaming

### Core Concept

Build a multi-step LLM application: extract links → fetch pages → generate brochure with streaming responses.

### 1. Structured JSON Output

Force the model to respond in valid JSON:

```python
import json

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Respond in valid JSON only"},
        {"role": "user", "content": "Extract these links as JSON"}
    ],
    response_format={"type": "json_object"}
)

# Parse response
result = response.choices[0].message.content
parsed_json = json.loads(result)
```

### 2. One-Shot Prompting

Provide an example of desired output in the system prompt:

```python
link_system_prompt = """
You extract links from a webpage.
Respond in JSON like this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/about"},
        {"type": "careers page", "url": "https://another.url/careers"}
    ]
}
"""

def select_relevant_links(url):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": f"Extract links from {url}"}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    links = json.loads(result)
    return links
```

### 3. Multi-Step LLM Pipeline

Chain multiple API calls together:

```python
def fetch_page_and_all_relevant_links(url):
    # Step 1: Get main page content
    main_content = fetch_website_contents(url)
  
    # Step 2: Get relevant links
    relevant_links = select_relevant_links(url)
  
    # Step 3: Fetch content from each link
    result = f"## Main Page:\n{main_content}\n## Relevant Links:\n"
  
    for link in relevant_links['links']:
        result += f"\n### {link['type']}:\n"
        result += fetch_website_contents(link["url"])
  
    return result
```

### 4. Streaming Responses

Stream responses back from the API for real-time display (typewriter effect):

```python
from IPython.display import Markdown, display, update_display

def stream_brochure(company_name, url):
    # Create stream with stream=True parameter
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Write a brochure"},
            {"role": "user", "content": "Create brochure for " + company_name}
        ],
        stream=True  # KEY: Enable streaming
    )
  
    response = ""
  
    # Initialize empty display with ID
    display_handle = display(Markdown(""), display_id=True)
  
    # Process stream chunks
    for chunk in stream:
        # Extract token from chunk
        delta_content = chunk.choices[0].delta.content or ''
        response += delta_content
      
        # Update display with accumulated response
        update_display(Markdown(response), display_id=display_handle.display_id)
```

**Key points about streaming:**

- Set `stream=True` in API call
- Each chunk contains partial response
- Use `update_display()` to update Jupyter cell in real-time
- Gives user immediate feedback while response completes

### 5. String Truncation Pattern

Limit prompt size to stay within token limits:

```python
def get_brochure_prompt(company_name, url):
    prompt = f"Create brochure for {company_name}:\n"
    prompt += fetch_page_and_all_relevant_links(url)
  
    # Truncate to 5000 characters
    prompt = prompt[:5_000]
  
    return prompt
```

### 6. Complete Brochure Application Pattern

```python
def create_brochure(company_name, url):
    # Step 1: Collect data
    page_content = fetch_page_and_all_relevant_links(url)
  
    # Step 2: Build prompt
    user_prompt = f"Create brochure for {company_name}:\n{page_content[:5_000]}"
  
    # Step 3: Call API without streaming (wait for full response)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You write compelling brochures"},
            {"role": "user", "content": user_prompt}
        ]
    )
  
    # Step 4: Display
    display(Markdown(response.choices[0].message.content))

create_brochure("Hugging Face", "https://huggingface.co")
```

---

## Essential Code Patterns Summary

### Pattern 1: Basic Completion

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Your message"}]
)
print(response.choices[0].message.content)
```

### Pattern 2: Multi-turn Conversation

```python
messages = [
    {"role": "system", "content": "System instructions"},
    {"role": "user", "content": "First user message"},
    {"role": "assistant", "content": "First assistant response"},
    {"role": "user", "content": "Follow-up message"}
]
response = client.chat.completions.create(model="gpt-4", messages=messages)
```

### Pattern 3: JSON Extraction

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    response_format={"type": "json_object"}
)
result = json.loads(response.choices[0].message.content)
```

### Pattern 4: Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or '', end='')
```

### Pattern 5: Using Different Providers

```python
# All providers use same OpenAI client syntax
client = OpenAI(base_url="PROVIDER_URL", api_key="API_KEY")
response = client.chat.completions.create(model="MODEL_NAME", messages=[...])
```

---

## Common Gotchas & Tips

1. **API Key**: Must be set in `.env` file and loaded with `load_dotenv()`
2. **Statelessness**: Always pass full conversation history to LLM
3. **Tokens**: Not the same as words; "banoffee" = 2 tokens
4. **Streaming**: Use `stream=True` and iterate over chunks
5. **JSON Mode**: Need `response_format={"type": "json_object"}` for structured output
6. **One-Shot Examples**: Include examples in system prompt for better output format
7. **Cost**: Streaming full conversation history each time adds to API costs
8. **Model Names**: Different models available per provider (gpt-4, gemini-3.1-flash-lite, llama3.2, etc.)
