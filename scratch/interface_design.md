Awesome, I'll build a clean Python setup using Pydantic models for Message and Response types, a base client interface to handle message conversion and response parsing (especially function calls), and methods to call LLM APIs in both streaming and non-streaming modes.

I'll create a full structure with a base interface and then concrete implementations for OpenAI, Bedrock (Claude), and Google Vertex AI (Gemini). Let's get started!

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Union
from pydantic import BaseModel
import json

# Common data models for chat messages and function calls.

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class Message(BaseModel):
    """
    A unified chat message format. Matches OpenAI roles and content style.
    Claude messages use 'role' and 'content', as do OpenAI and Vertex.
    Content may be a string or list of content blocks (for Claude multi-part content) ([Anthropic Claude Messages API - Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#:~:text=Each%20input%20message%20content%20may,following%20input%20messages%20are%20equivalent)).
    """
    role: str  # 'system', 'user', 'assistant'
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None  # for function-name (OpenAI function responses)
    function_call: Optional[FunctionCall] = None

class ChatResponseChoice(BaseModel):
    message: Message
    finish_reason: Optional[str] = None

class ChatResponse(BaseModel):
    """
    Unified chat response with choices. Each choice has a message and optional finish_reason.
    """
    choices: List[ChatResponseChoice]

# Base interface for LLM clients.
class BaseClient(ABC):
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize client with model name or ID.
        Authentication (API keys, etc.) is handled outside or via kwargs (not implemented here).
        """
        self.model_name = model_name

    @abstractmethod
    def _format_messages(self, messages: List[Message]) -> Any:
        """
        Convert internal Message list to provider-specific request format.
        """
        pass

    @abstractmethod
    def _parse_response(self, api_response: Any) -> ChatResponse:
        """
        Parse the provider-specific API response into a ChatResponse.
        Extract function_call if present.
        """
        pass

    @abstractmethod
    def call_llm(self,
                 messages: List[Message],
                 functions: Optional[List[Dict[str, Any]]] = None,
                 stream: bool = False) -> Union[ChatResponse, Any]:
        """
        Call the LLM API with given messages and optional function definitions.
        Supports streaming if specified (returning a generator or stream).
        """
        pass

# OpenAI LLM client implementation.
class OpenAIClient(BaseClient):
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name)
        import openai
        self.api_key = api_key
        openai.api_key = api_key
        self.client = openai

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """
        OpenAI expects list of messages with 'role' and 'content'.
        """
        formatted = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.name:
                # OpenAI uses 'name' for assistant message representing function output
                m["name"] = msg.name
            formatted.append(m)
        return formatted

    def _parse_response(self, api_response: Any) -> ChatResponse:
        """
        Parse OpenAI ChatCompletion response. 
        Extract 'role', 'content', and any function_call data.
        """
        choices = []
        for choice in api_response.choices:
            msg_data = choice.message
            role = msg_data.get("role")
            content = msg_data.get("content")
            name = msg_data.get("name")
            message = Message(role=role, content=content or "", name=name)
            # Extract function_call if present
            if msg_data.get("function_call"):
                fc = msg_data["function_call"]
                message.function_call = FunctionCall(
                    name=fc.get("name"), 
                    arguments=json.loads(fc.get("arguments", "{}"))
                )
            finish_reason = choice.get("finish_reason")
            choices.append(ChatResponseChoice(message=message, finish_reason=finish_reason))
        return ChatResponse(choices=choices)

    def call_llm(self,
                 messages: List[Message],
                 functions: Optional[List[Dict[str, Any]]] = None,
                 stream: bool = False) -> Union[ChatResponse, Any]:
        """
        Call OpenAI ChatCompletion API.
        """
        formatted_msgs = self._format_messages(messages)
        if stream:
            return self.client.ChatCompletion.create(
                model=self.model_name,
                messages=formatted_msgs,
                functions=functions,
                stream=True
            )
        else:
            response = self.client.ChatCompletion.create(
                model=self.model_name,
                messages=formatted_msgs,
                functions=functions
            )
            return self._parse_response(response)

# Amazon Bedrock (Anthropic Claude) client implementation.
class BedrockClaudeClient(BaseClient):
    def __init__(self, model_id: str, **kwargs):
        """
        model_id should be the Bedrock model identifier (e.g., 'anthropic.claude-v1').
        """
        super().__init__(model_id)
        import boto3
        self.client = boto3.client('bedrock', region_name=kwargs.get("region"))

    def _format_messages(self, messages: List[Message]) -> Dict[str, Any]:
        """
        Bedrock Claude Messages API expects a dict with 'messages' (list of role/content) ([Anthropic Claude Messages API - Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#:~:text=Each%20input%20message%20must%20be,always%20use%20the%20user%20role)).
        Roles: 'system', 'user', 'assistant'.
        Content may be str or list of content blocks (for Claude multi-part content) ([Anthropic Claude Messages API - Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#:~:text=Each%20input%20message%20content%20may,following%20input%20messages%20are%20equivalent)).
        """
        claude_messages = []
        for msg in messages:
            claude_msg = {"role": msg.role, "content": msg.content}
            claude_messages.append(claude_msg)
        return {"modelId": self.model_name, "messages": claude_messages}

    def _parse_response(self, api_response: Any) -> ChatResponse:
        """
        Parse Bedrock Claude response from InvokeModel (non-streaming).
        Extract assistant message. Handle tool_use (function call) if present.
        """
        content = api_response.get("body")
        if isinstance(content, (bytes, str)):
            body = json.loads(content)
        else:
            body = content
        msg = body.get("completion", {}).get("message", {})
        role = msg.get("role")
        raw_content = msg.get("content")
        message = Message(role=role, content=raw_content or "")
        # The model may include a content block of type 'tool_use' (function call) ([Anthropic Claude Messages API - Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html#:~:text=match%20at%20L198%20When%20the,tool_use)).
        if body.get("stop_reason") == "tool_use" or isinstance(raw_content, list):
            tool_block = None
            if isinstance(raw_content, list):
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "tool_use":
                        tool_block = part
                        break
            if tool_block:
                fc_name = tool_block.get("name")
                fc_args = tool_block.get("input", {})
                message.function_call = FunctionCall(name=fc_name, arguments=fc_args)
        choice = ChatResponseChoice(message=message, finish_reason=body.get("stop_reason"))
        return ChatResponse(choices=[choice])

    def call_llm(self,
                 messages: List[Message],
                 functions: Optional[List[Dict[str, Any]]] = None,
                 stream: bool = False) -> ChatResponse:
        """
        Call Bedrock Claude via InvokeModel (messages API).
        For streaming, use InvokeModelWithResponseStream (not implemented here).
        """
        payload = self._format_messages(messages)
        if functions:
            # In Claude, tools (function definitions) are provided separately
            payload["tools"] = functions
        if stream:
            raise NotImplementedError("Streaming not implemented for BedrockClaudeClient.")
        response = self.client.invoke_model(**payload)
        return self._parse_response(response)

# Google Vertex AI (Gemini) client implementation.
class VertexGeminiClient(BaseClient):
    def __init__(self, model_id: str, **kwargs):
        """
        model_id: Vertex model alias (e.g., 'chat-bison@001').
        Assumes google-cloud-aiplatform is configured with credentials externally.
        """
        super().__init__(model_id)
        from google.cloud import aiplatform
        aiplatform.init(project=kwargs.get("project"), location=kwargs.get("location"))
        self.chat_model = aiplatform.language_models.ChatModel.from_pretrained(model_id)

    def _format_messages(self, messages: List[Message]) -> List[Any]:
        """
        Convert to Vertex ChatMessage dicts. Vertex uses 'author' (user/bot) and 'content'.
        """
        formatted = []
        for msg in messages:
            if msg.role == "system":
                author = "user"
            elif msg.role == "assistant":
                author = "bot"
            else:
                author = msg.role
            formatted.append({"author": author, "content": msg.content})
        return formatted

    def _parse_response(self, api_response: Any) -> ChatResponse:
        """
        Parse Vertex AI chat response. The model may respond with JSON for function calls or plain text.
        """
        message_text = api_response.text if hasattr(api_response, "text") else api_response.content
        message = Message(role="assistant", content=message_text)
        # Vertex (Gemini) responds with JSON for function calls ([Function Calling with the Gemini API  |  Google AI for Developers](https://ai.google.dev/gemini-api/docs/function-calling#:~:text=a%20function%20call%20would%20be,this)).
        try:
            json_resp = json.loads(message_text)
            if isinstance(json_resp, dict) and "name" in json_resp and "arguments" in json_resp:
                message.function_call = FunctionCall(name=json_resp["name"], arguments=json_resp["arguments"])
                message.content = ""
        except Exception:
            pass
        choice = ChatResponseChoice(message=message, finish_reason=None)
        return ChatResponse(choices=[choice])

    def call_llm(self,
                 messages: List[Message],
                 functions: Optional[List[Dict[str, Any]]] = None,
                 stream: bool = False) -> ChatResponse:
        """
        Call Vertex AI chat model. Uses google-cloud-aiplatform ChatModel.
        """
        if stream:
            raise NotImplementedError("Streaming not implemented for VertexGeminiClient.")
        formatted = self._format_messages(messages)
        # Start chat session and send last user message
        chat = self.chat_model.start_chat(message_history=formatted)
        response = chat.send_message(formatted[-1]["content"])
        return self._parse_response(response)
```