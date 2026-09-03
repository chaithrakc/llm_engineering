
# LLM Engineering Course - Study Notes

## DAY 1: SDK and Caching

### Key Concepts

- **API Key Management**: Use `.env` file to store sensitive API keys
- **OpenAI Python Client**: Universal wrapper that works with multiple LLM providers by changing `base_url`
- **Message Structure**: Conversations are built as lists of message dictionaries with `role` and `content` keys
- **Multi-Model Comparison**: Can easily swap between different LLM providers using the same client interface

### Essential Code Syntax

**Loading Environment Variables:**

```python
from dotenv import load_dotenv
import os

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
```

**Connecting to Different LLM Providers:**

```python
from openai import OpenAI

# OpenAI (default)
openai = OpenAI()

# Gemini (via OpenAI-compatible endpoint)
gemini = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

# DeepSeek
deepseek = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

# Groq
groq = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

# Local Ollama
ollama = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
```

**Building and Using Message History:**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "First question"},
    {"role": "assistant", "content": "First answer"},
    {"role": "user", "content": "Follow up question"}
]

response = client.chat.completions.create(
    model="model-name",
    messages=messages
)

answer = response.choices[0].message.content
```

**Key Pattern**: Messages list maintains full conversation history for context

---

## DAY 2: Gradio and User Interfaces

### Key Concepts

- **Gradio**: Turns Python functions into web interfaces with zero HTML/CSS knowledge
- **Three UI Types**: `gr.Interface` (simple), `gr.ChatInterface` (chatbots), `gr.Blocks` (custom)
- **Callbacks**: Functions you pass to Gradio that get called when users interact with the UI
- **Launch Options**: Control visibility, authentication, and browser behavior

### Essential Code Syntax

**Simple Interface:**

```python
import gradio as gr

def process(text):
    return text.upper()

gr.Interface(
    fn=process,
    inputs="textbox",
    outputs="textbox",
    title="My App",
    examples=["hello", "world"],
    flagging_mode="never"
).launch()
```

**Interface with Custom Components:**

```python
input_box = gr.Textbox(
    label="Your message:",
    info="Enter text here",
    lines=7
)
output_box = gr.Textbox(label="Response:", lines=8)

gr.Interface(
    fn=process,
    inputs=[input_box],
    outputs=[output_box],
    title="My App"
).launch()
```

**Launch Options:**

```python
.launch(
    inbrowser=True,           # Auto-open browser
    share=True,               # Create public link (temporary)
    auth=("username", "password")  # Require login
)
```

**Dark Mode (JavaScript):**

```python
force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
gr.Interface(..., js=force_dark_mode).launch()
```

---

## DAY 3: Chat Interfaces

### Key Concepts

- **ChatInterface**: Specialized Gradio component for building chatbots
- **Streaming Responses**: Use `stream=True` to yield responses gradually for better UX
- **History Format**: Gradio ChatInterface expects messages with `role` and `content` keys
- **Dynamic System Prompts**: Can modify system message based on user input patterns

### Essential Code Syntax

**Basic Chat Callback:**

```python
def chat(message, history):
    # Convert Gradio history format to OpenAI format
    history = [{"role": h["role"], "content": h["content"]} for h in history]
  
    # Build message list with system prompt
    messages = [
        {"role": "system", "content": system_message}
    ] + history + [
        {"role": "user", "content": message}
    ]
  
    # Call LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
  
    return response.choices[0].message.content

# Launch
gr.ChatInterface(fn=chat, type="messages").launch()
```

**Streaming Chat (Typing Effect):**

```python
def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
  
    # Use stream=True
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )
  
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response  # Yield partial responses
```

**Dynamic System Prompts:**

```python
def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
  
    # Modify system message based on input
    relevant_system_message = system_message
    if 'belt' in message.lower():
        relevant_system_message += " The store does not sell belts."
  
    messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]
  
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )
  
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response
```

---

## DAY 4: LLM Tools and Function Calling

### Key Concepts

- **Function Calling**: Let LLMs decide when and how to call your functions
- **Tool Definition Schema**: JSON schema that describes function parameters
- **Tool Response Handling**: Manual loop to handle tool calls and feed results back to LLM
- **Multi-Tool Loops**: Can handle multiple sequential tool calls

### Essential Code Syntax

**Define a Tool Function:**

```python
def get_ticket_price(destination_city):
    print(f"Tool called for {destination_city}")
    prices = {"london": "$799", "paris": "$899", "tokyo": "$1400"}
    price = prices.get(destination_city.lower(), "Unknown")
    return f"Ticket to {destination_city} is {price}"
```

**Describe Tool in Schema:**

```python
tool_definition = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to"
            }
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": tool_definition}]
```

**Chat with Tools (Single Call):**

```python
def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
  
    # Request with tools
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
  
    # Check if LLM wants to use a tool
    if response.choices[0].finish_reason == "tool_calls":
        message_obj = response.choices[0].message
        tool_response = handle_tool_call(message_obj)
      
        # Add tool call and response back to messages
        messages.append(message_obj)
        messages.append(tool_response)
      
        # Get final response with tool results
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
  
    return response.choices[0].message.content
```

**Handle Tool Call:**

```python
import json

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
  
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get('destination_city')
        result = get_ticket_price(city)
      
        return {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call.id
        }
```

**Handle Multiple Tool Calls:**

```python
def handle_tool_calls(message):  # Note: plural
    responses = []
    for tool_call in message.tool_calls:  # Loop through all
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            result = get_ticket_price(city)
          
            responses.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })
    return responses
```

**Tool Loop with Multiple Sequential Calls:**

```python
def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
  
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
  
    # Keep looping while LLM wants to call tools
    while response.choices[0].finish_reason == "tool_calls":
        message_obj = response.choices[0].message
        tool_responses = handle_tool_calls(message_obj)
      
        messages.append(message_obj)
        messages.extend(tool_responses)  # Add all responses
      
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
  
    return response.choices[0].message.content
```

---

## DAY 5: Multi-Modal AI Assistant (Airline Project)

### Key Concepts

- **Combining All Skills**: Chatbot + Tools + Image Generation + Audio
- **gr.Blocks**: Custom UI for complex layouts with multiple components
- **Tool Extraction**: Extract tool call data to use outside the LLM response
- **Multi-Output Callbacks**: Return multiple outputs (text, images, audio)

### Essential Code Syntax

**Database Tool Example:**

```python
import sqlite3

def get_ticket_price(city):
    with sqlite3.connect("prices.db") as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No data available"
```

**Image Generation (OpenAI API):**

```python
import base64
from io import BytesIO
from PIL import Image

def artist(city):
    image_response = openai.images.generate(
        model="gpt-image-1-mini",
        prompt=f"An image representing a vacation in {city}, showing unique features",
        size="1024x1024",
        n=1
    )
  
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))
```

**Text-to-Speech (OpenAI API):**

```python
def talker(message):
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",  # or "alloy", "coral"
        input=message
    )
    return response.content
```

**Tool Calls with City Extraction:**

```python
def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
  
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            cities.append(city)  # Extract for later use
          
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
  
    return responses, cities
```

**Complex Chat Callback:**

```python
def chat(history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history
  
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
  
    cities = []
  
    # Handle tool calls
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)
        messages.append(message)
        messages.extend(responses)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
  
    # Extract final response and update history
    reply = response.choices[0].message.content
    history += [{"role": "assistant", "content": reply}]
  
    # Generate audio and image
    voice = talker(reply)
    image = artist(cities[0]) if cities else None
  
    return history, voice, image
```

**Custom UI with gr.Blocks:**

```python
def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, interactive=False)
  
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
  
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")
  
    # Chain events: add message → run chat → display outputs
    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, audio_output, image_output]
    )

ui.launch(inbrowser=True, auth=("ed", "bananas"))
```

---

## Key Patterns Summary

### Always Use

1. **System Message**: Set context and tone at the start
2. **Message History**: Keep full conversation for context
3. **Error Handling**: Check `finish_reason` for tool calls
4. **Environment Variables**: Store keys in `.env`, never hardcode

### Best Practices

- **Streaming for UX**: Use `stream=True` and `yield` for real-time feedback
- **Tool Loops**: Always use `while finish_reason == "tool_calls"` for multiple calls
- **History Format**: Convert between Gradio format and API format
- **Graceful Fallbacks**: Handle missing data, unknown cities, etc.

### Common Code Structure

```python
# 1. Load keys
load_dotenv()
client = OpenAI(base_url=url, api_key=key)

# 2. Define system message
system_message = "..."

# 3. Define tools (if needed)
tools = [{"type": "function", "function": {...}}]

# 4. Define callback
def callback(input, history):
    # Format history
    # Build messages
    # Call LLM with tools if applicable
    # Handle tool calls if needed
    # Return response

# 5. Create UI
gr.ChatInterface(fn=callback, type="messages").launch()
```
