from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_groq import ChatGroq
from typing import List, Dict, Optional, Union
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TEMPERATURE = float(os.getenv("TEMPERATURE"))

class MessageType(Enum):
    """Message type enumeration"""
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"


@dataclass
class MessageRequest:
    """Request object for message-based API"""
    message_type: str  # "system", "human", or "ai"
    content: str
    metadata: Optional[Dict] = None


@dataclass
class ConversationMessage:
    """Conversation message object"""
    type: str
    content: str
    metadata: Optional[Dict] = None
    
    def to_dict(self):
        return asdict(self)


class LangChainMessageAPI:
    """
    API interface for LangChain message-based communication
    Handles SystemMessage, AIMessage, and HumanMessage
    """
    
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = MODEL_NAME, temperature: float = TEMPERATURE):
        """Initialize the API with LLM client"""
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=temperature
        )
        self.conversation_history: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
        self.system_prompt: Optional[str] = None
    
    def set_system_message(self, content: str) -> None:
        """Set system message for the conversation"""
        self.system_prompt = content
        self.conversation_history = [SystemMessage(content=content)]
        print(f"✓ System message set")
    
    def add_human_message(self, content: str, metadata: Optional[Dict] = None) -> ConversationMessage:
        """Add a human message to the conversation"""
        message = HumanMessage(content=content)
        self.conversation_history.append(message)
        
        conv_msg = ConversationMessage(
            type="human",
            content=content,
            metadata=metadata
        )
        return conv_msg
    
    def add_ai_message(self, content: str, metadata: Optional[Dict] = None) -> ConversationMessage:
        """Add an AI message to the conversation"""
        message = AIMessage(content=content)
        self.conversation_history.append(message)
        
        conv_msg = ConversationMessage(
            type="ai",
            content=content,
            metadata=metadata
        )
        return conv_msg
    
    def invoke(self, user_input: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Invoke the LLM with current conversation history
        Returns response with message type and content
        """
        try:
            # Add human message
            self.add_human_message(user_input, metadata)
            
            # Invoke LLM
            response = self.llm.invoke(self.conversation_history)
            
            # Extract and add AI message
            ai_response = response.content
            self.add_ai_message(ai_response)
            
            return {
                "success": True,
                "response": ai_response,
                "message_type": "ai",
                "conversation_length": len(self.conversation_history)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message_type": None
            }
    
    def batch_invoke(self, requests: List[Dict]) -> List[Dict]:
        """
        Process multiple requests in batch
        Each request should have 'user_input' and optional 'metadata'
        """
        results = []
        for request in requests:
            user_input = request.get("user_input", "")
            metadata = request.get("metadata", None)
            result = self.invoke(user_input, metadata)
            results.append(result)
        
        return results
    
    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        history = []
        for msg in self.conversation_history:
            if isinstance(msg, SystemMessage):
                history.append({
                    "type": "system",
                    "content": msg.content
                })
            elif isinstance(msg, HumanMessage):
                history.append({
                    "type": "human",
                    "content": msg.content
                })
            elif isinstance(msg, AIMessage):
                history.append({
                    "type": "ai",
                    "content": msg.content
                })
        
        return history
    
    def clear_history(self) -> None:
        """Clear conversation history but keep system message"""
        if self.system_prompt:
            self.conversation_history = [SystemMessage(content=self.system_prompt)]
        else:
            self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def export_conversation(self, filepath: str) -> None:
        """Export conversation to JSON file"""
        history = self.get_conversation_history()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"✓ Conversation exported to {filepath}")
    
    def import_conversation(self, filepath: str) -> None:
        """Import conversation from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        self.conversation_history = []
        for msg in history:
            if msg['type'] == 'system':
                self.conversation_history.append(SystemMessage(content=msg['content']))
            elif msg['type'] == 'human':
                self.conversation_history.append(HumanMessage(content=msg['content']))
            elif msg['type'] == 'ai':
                self.conversation_history.append(AIMessage(content=msg['content']))
        
        print(f"✓ Conversation imported from {filepath}")



def interactive_cli():
    """Interactive CLI for testing the API"""
    print("=" * 70)
    print("LangChain Message Interface - Interactive CLI")
    print("=" * 70)
    
    api = LangChainMessageAPI()
    
    # Set custom system message
    system_msg = input("\nEnter system message (or press Enter for default): ").strip()
    if system_msg:
        api.set_system_message(system_msg)
    else:
        api.set_system_message("You are a helpful AI assistant.")
    
    # Interactive conversation
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        result = api.invoke(user_input)
        if result['success']:
            print(f"\nAssistant: {result['response']}")
        else:
            print(f"\nError: {result['error']}")
    
    # Save conversation
    save_choice = input("\nSave conversation? (y/n): ").strip().lower()
    if save_choice == 'y':
        api.export_conversation("conversation_history.json")


def main():
    """Main function"""
    interactive_cli()


if __name__ == "__main__":
    main()
