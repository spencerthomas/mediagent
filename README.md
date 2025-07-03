# LangGraph.js Medical Diagnostic Agent

<!-- TODO: Add CI badges -->
<!-- [![CI](https://github.com/langchain-ai/react-agent-js/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/react-agent-js/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/react-agent/actions/workflows/integration-tests.yml) -->

[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/react-agent-js)

This application showcases an **Interactive Medical Diagnostic Agent** implemented using [LangGraph.js](https://github.com/langchain-ai/langgraphjs), designed for [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio). The medical agent uses a sophisticated multi-agent approach based on Microsoft's AI Diagnostic Orchestrator (MAI-DxO) research, featuring **interactive patient consultation with intelligent interrupt logic**.

![Graph view in LangGraph studio UI](./static/studio_ui.png)

The core logic, defined in `src/medical_agent/medical_graph.ts`, demonstrates an advanced medical superintelligence system that combines multi-agent deliberation with real-time patient interaction, showcasing the power of this approach for complex medical diagnostic tasks.

## How the Interactive Medical Agent Works

### 🔄 **Interactive Diagnostic Workflow**

The Medical Diagnostic Agent transforms traditional AI consultation into a dynamic, question-driven process:

1. **Initial Case Assessment** - Takes a medical case as input and extracts patient information
2. **Multi-Agent Deliberation** - 5 specialized physician agents debate and analyze the case
3. **🔄 Smart Interrupts** - System pauses when more information is needed
4. **📋 Targeted Questions** - Generates 3-5 specific follow-up questions based on diagnostic gaps
5. **💬 Patient Interaction** - Waits for user responses and processes them intelligently
6. **📊 Information Integration** - Structures responses and updates case information
7. **🏁 Continued Analysis** - Resumes multi-agent debate with enhanced information
8. **✅ Final Diagnosis** - Provides comprehensive diagnostic assessment

### 🧠 **Multi-Agent Physician Team**

**Dr. Hypothesis** - Maintains probability-ranked differential diagnoses
**Dr. Test-Chooser** - Selects optimal diagnostic tests and procedures  
**Dr. Challenger** - Identifies biases and proposes alternative theories
**Dr. Stewardship** - Enforces cost-effective medical decision making
**Dr. Checklist** - Performs systematic quality control and verification

### 🔄 **Interrupt Logic Implementation**

**Key Features:**
- **Enhanced State Management** - Tracks patient interactions and question history
- **Patient Question Agent** - Generates contextually relevant follow-up questions
- **Enhanced Gatekeeper** - Processes user responses using AI extraction
- **Interrupt Workflow** - Pauses execution when questions are needed

**When Interrupts Trigger:**
- Diagnostic confidence < 60%
- Multiple competing diagnoses exist
- Agents explicitly request more information
- Critical information gaps identified

**Example Generated Questions:**
- **Timeline**: "When did your symptoms first begin?"
- **Symptom Details**: "On a scale of 1-10, how would you rate your pain?"
- **Medical History**: "Are you currently taking any medications?"
- **Associated Symptoms**: "Any recent fever or chills?"

This implementation mirrors the MAI-DxO pattern of sequential, question-driven information gathering, preventing endless internal loops and enabling realistic diagnostic consultation that adapts based on patient responses.

## Getting Started

Assuming you have already [installed LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download), to set up:

1. Create a `.env` file.

```bash
cp .env.example .env
```

2. Define required API keys in your `.env` file.



<!--
Setup instruction auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
-->

### Setup Model

The defaults values for `model` are shown below:

```yaml
model: anthropic/claude-3-5-sonnet-20240620
```

Follow the instructions below to get set up, or pick one of the additional options.

#### Anthropic

To use Anthropic's chat models:

1. Sign up for an [Anthropic API key](https://console.anthropic.com/) if you haven't already.
2. Once you have your API key, add it to your `.env` file:

```
ANTHROPIC_API_KEY=your-api-key
```

#### OpenAI

To use OpenAI's chat models:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:

```
OPENAI_API_KEY=your-api-key
```

<!--
End setup instructions
-->

3. Customize whatever you'd like in the code.
4. Open the folder in LangGraph Studio!

## How to customize

1. **Add new tools**: Extend the agent's capabilities by adding new tools in `src/react_agent/tools.ts`. These can be any TypeScript functions that perform specific tasks.
2. **Select a different model**: We default to Anthropic's Claude 3.5 Sonnet. You can select a compatible chat model using `provider/model-name` via configuration, then installing the proper [chat model integration package](https://js.langchain.com/docs/integrations/chat/). Example: `openai/gpt-4-turbo-preview`, then run `npm i @langchain/openai`.
3. **Customize the prompt**: We provide a default system prompt in `src/react_agent/prompts.ts`. You can easily update this via configuration in the studio.

You can also quickly extend this template by:

- Modifying the agent's reasoning process in `src/react_agent/graph.ts`.
- Adjusting the ReAct loop or adding additional steps to the agent's decision-making process.

## Development

While iterating on your graph, you can edit past state and rerun your app from past states to debug specific nodes. Local changes will be automatically applied via hot reload. Try adding an interrupt before the agent calls tools, updating the default system message in `src/react_agent/configuration.ts` to take on a persona, or adding additional nodes and edges!

Follow up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

You can find the latest (under construction) docs on [LangGraph](https://langchain-ai.github.io/langgraphjs/) here, including examples and other references. Using those guides can help you pick the right patterns to adapt here for your use case.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates.

[^1]: https://js.langchain.com/docs/concepts#tools

<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "agent": {
      "type": "object",
      "properties": {
        "model": {
          "type": "string",
          "default": "anthropic/claude-3-5-sonnet-20240620",
          "description": "The name of the language model to use for the agent's main interactions. Should be in the form: provider/model-name.",
          "environment": [
            {
              "value": "anthropic/claude-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.0",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.1",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-5-sonnet-20240620",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-haiku-20240307",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-opus-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-sonnet-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-instant-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0125",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0301",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-1106",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0125-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-1106-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-vision-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o-mini",
              "variables": "OPENAI_API_KEY"
            }
          ]
        }
      },
      "environment": [
        "TAVILY_API_KEY"
      ]
    }
  }
}
-->
