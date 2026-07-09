"""Agent core: Agentic tool-use loop with Claude API.

Manages the conversation loop between the user, Claude LLM, and the
data analysis tools. Claude decides which tools to call based on
natural language input, and the agent executes them and returns results.
"""

import json
from typing import Any

import anthropic

from data_analyst.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_BASE_URL,
    MODEL_NAME,
    MAX_TOKENS,
    MAX_TURNS,
    ensure_output_dirs,
)
from data_analyst.tools import ALL_TOOLS, TOOL_HANDLERS


SYSTEM_PROMPT = """You are an expert data analysis agent. You help users analyze data, create visualizations, and build machine learning models.

## Your Capabilities

1. **Data Loading**: Load CSV, Excel, JSON, TSV files and understand their structure
2. **Statistical Analysis**: Descriptive stats, correlation, distribution tests, hypothesis testing, group-by analysis, outlier detection
3. **Visualization**: Create charts (line, bar, scatter, histogram, box, violin, heatmap, pairplot, pie, count, area) saved as PNG/SVG
4. **Machine Learning**: Train regression, classification, and clustering models using scikit-learn, then make predictions

## How You Work

1. When a user provides a data file, first load it with `load_data` to understand its structure
2. Based on the user's request, choose the appropriate analysis/visualization/ML tools
3. For complex requests, break them into steps and call tools sequentially
4. Always explain your findings in clear, natural language after each analysis step
5. When creating visualizations, suggest the most appropriate chart type for the data and question
6. For ML tasks, explain which algorithm you chose and why, and interpret the metrics

## Important Guidelines

- Always load data first before performing any analysis
- Use the `data_id` returned by `load_data` in subsequent tool calls
- For visualizations, provide meaningful titles and labels (support Chinese characters)
- When training ML models, start with a simple algorithm and only tune hyperparameters if requested
- Explain statistical results in plain language - don't just report numbers
- If a tool returns an error, explain the issue and suggest alternatives
- For correlation analysis, use heatmap visualization to make it intuitive
- When data has missing values, suggest cleaning options before analysis

## Response Language

Respond in the same language the user uses. If the user writes in Chinese, respond in Chinese. If in English, respond in English.
"""


class DataAnalystAgent:
    """Core agent that manages the Claude API tool-use loop."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or MODEL_NAME
        self.max_tokens = MAX_TOKENS
        self.max_turns = MAX_TURNS

        # Initialize Anthropic client
        client_kwargs = {"api_key": self.api_key}
        if ANTHROPIC_BASE_URL:
            client_kwargs["base_url"] = ANTHROPIC_BASE_URL
        self.client = anthropic.Anthropic(**client_kwargs)

        # Conversation history
        self.messages: list[dict] = []

        # Ensure output directories exist
        ensure_output_dirs()

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool and return the result as a JSON string."""
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = handler(**tool_input)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"Tool execution error: {str(e)}"})

    def run(self, user_message: str) -> str:
        """Run a single user message through the agentic loop.

        Returns the final assistant response text.
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        final_response = ""

        for turn in range(self.max_turns):
            # Call Claude API
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=ALL_TOOLS,
                    messages=self.messages,
                )
            except anthropic.APIError as e:
                error_msg = f"API Error: {str(e)}"
                final_response = error_msg
                break

            # Process response
            assistant_content = response.content
            self.messages.append({"role": "assistant", "content": assistant_content})

            # Check if we have tool calls or just text
            tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
            text_blocks = [b for b in assistant_content if b.type == "text"]

            # Collect text response
            for block in text_blocks:
                final_response += block.text

            # If no tool calls, we're done
            if not tool_use_blocks:
                break

            # Execute tool calls and add results to conversation
            tool_results = []
            for tool_block in tool_use_blocks:
                result_str = self._execute_tool(tool_block.name, tool_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_str,
                })

            self.messages.append({"role": "user", "content": tool_results})

        else:
            # Max turns reached
            final_response += "\n\n[Agent reached maximum number of turns]"

        return final_response

    def reset(self):
        """Reset the conversation history."""
        self.messages = []

    def get_conversation_summary(self) -> dict[str, Any]:
        """Return a summary of the current conversation state."""
        from data_analyst.tools.data_loader import list_loaded_data
        return {
            "message_count": len(self.messages),
            "loaded_datasets": list_loaded_data(),
            "model": self.model,
        }
