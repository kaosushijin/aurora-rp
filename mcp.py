# Chunk 1 of 1 - Complete remodularized mcp.py preserving Node.js ollama MCP server configuration
# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
DevName RPG Client - MCP Communication Module (mcp.py)
Remodularized for hub-and-spoke architecture
ONLY accessed through orch.py orchestrator - no direct spoke access
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Check for httpx availability
try:
    import httpx
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# MCP Configuration Constants - PRESERVED from original Node.js ollama setup
MCP_SERVER_URL = "http://127.0.0.1:3456/chat"
MCP_MODEL = "qwen2.5:14b-instruct-q4_k_m"
MCP_TIMEOUT = 300.0  # 5 minutes for ollama processing
MCP_MAX_RETRIES = 2

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant for a fantasy RPG game.
Guide the player through immersive adventures with rich descriptions, engaging NPCs, and dynamic storytelling. 
Maintain story continuity and respond to player actions creatively."""

@dataclass
class MCPState:
    """State management for MCP client"""
    connected: bool = False
    initialized: bool = False
    last_request_time: float = 0.0
    request_count: int = 0

class MCPClient:
    """
    HTTP client for Node.js ollama MCP server communication
    Remodularized for exclusive orchestrator access
    Implements service module interface standards
    """
    
    def __init__(self, server_url: str = MCP_SERVER_URL, model: str = MCP_MODEL, 
                 system_prompt: str = None, debug_logger=None):
        """Initialize MCP client with remodularized interface"""
        self.server_url = server_url
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.debug_logger = debug_logger
        self.state = MCPState()
        
        # Orchestrator callback for service module interface
        self.orchestrator_callback: Optional[Callable] = None
        
        # Validate httpx availability
        if not MCP_AVAILABLE:
            if self.debug_logger:
                self.debug_logger.error("httpx not available - install with: pip install httpx", "MCP")
    
    def initialize(self, orchestrator_callback: Callable) -> bool:
        """
        Initialize MCP client with orchestrator callback
        Implements standard service module interface
        """
        try:
            self.orchestrator_callback = orchestrator_callback
            
            if not MCP_AVAILABLE:
                if self.debug_logger:
                    self.debug_logger.error("Cannot initialize - httpx not available", "MCP")
                return False
            
            # Test connection to Node.js ollama MCP server
            connection_test = self.test_connection()
            
            self.state.initialized = True
            self.state.connected = connection_test
            
            if self.debug_logger:
                status = "SUCCESS" if connection_test else "FAILED"
                self.debug_logger.debug(f"MCP client initialized - Connection: {status}", "MCP")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"MCP initialization failed: {e}", "MCP")
            return False
    
    def shutdown(self) -> None:
        """
        Shutdown MCP client
        Implements standard service module interface
        """
        try:
            self.state.connected = False
            self.state.initialized = False
            self.orchestrator_callback = None
            
            if self.debug_logger:
                self.debug_logger.debug("MCP client shutdown complete", "MCP")
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"MCP shutdown error: {e}", "MCP")
    
    def cleanup(self) -> None:
        """
        Cleanup method for orchestrator shutdown compatibility
        MCPClient uses httpx.AsyncClient which handles its own cleanup
        """
        self.shutdown()
    
    def test_connection(self) -> bool:
        """Test connection to Node.js ollama MCP server"""
        if not MCP_AVAILABLE:
            return False
            
        try:
            async def _test():
                async with httpx.AsyncClient(timeout=5.0) as client:
                    test_payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "stream": False
                    }
                    try:
                        response = await client.post(self.server_url, json=test_payload)
                        return response.status_code == 200
                    except:
                        return False
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_test())
            loop.close()
            
            self.state.connected = result
            if self.debug_logger:
                self.debug_logger.debug(f"Connection test: {'SUCCESS' if result else 'FAILED'}", "MCP")
            
            return result
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Connection test error: {str(e)}", "MCP")
            return False
    
    def send_message(self, user_input: str, conversation_history: List[Dict[str, str]] = None, 
                     story_context: str = None) -> str:
        """Send message to Node.js ollama MCP server with context integration"""
        if not MCP_AVAILABLE:
            raise Exception("httpx not installed - run: pip install httpx")
        
        # Build message context
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add story context from SME if provided
        if story_context:
            messages.append({"role": "system", "content": f"Story Context: {story_context}"})
        
        # Add conversation history from EMM (last 10 messages for efficiency)
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append(msg)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Prepare request payload - PRESERVED Node.js ollama format
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        if self.debug_logger:
            self.debug_logger.debug(f"Request: {len(user_input)} chars, {len(messages)} messages", "MCP")
        
        # Update state
        self.state.last_request_time = time.time()
        self.state.request_count += 1
        
        # Execute request with retry logic
        return self._execute_request(payload)
    
    def _execute_request(self, payload: Dict[str, Any]) -> str:
        """Execute MCP request with simplified retry mechanism for Node.js ollama server"""
        last_error = None
        
        for attempt in range(MCP_MAX_RETRIES + 1):
            try:
                # Use asyncio for HTTP request
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def _send():
                    async with httpx.AsyncClient(timeout=MCP_TIMEOUT) as client:
                        response = await client.post(self.server_url, json=payload)
                        response.raise_for_status()
                        return response.json()
                
                start_time = time.time()
                response_data = loop.run_until_complete(_send())
                loop.close()
                
                # Validate and extract response - PRESERVED Node.js ollama format
                if self._validate_response(response_data):
                    response_time = time.time() - start_time
                    if self.debug_logger:
                        self.debug_logger.debug(f"Response received: {response_time:.2f}s", "MCP")
                    
                    return response_data["message"]["content"]
                else:
                    raise Exception("Invalid response format from Node.js ollama server")
                    
            except Exception as e:
                last_error = e
                if self.debug_logger:
                    self.debug_logger.debug(f"Attempt {attempt + 1} failed: {str(e)}", "MCP")
                
                if attempt < MCP_MAX_RETRIES:
                    time.sleep(1)  # Brief delay before retry
                    continue
        
        # All retries exhausted
        error_msg = f"MCP request failed after {MCP_MAX_RETRIES + 1} attempts: {str(last_error)}"
        if self.debug_logger:
            self.debug_logger.error(error_msg, "MCP")
        raise Exception(error_msg)
    
    def _validate_response(self, response_data: Dict[str, Any]) -> bool:
        """Validate Node.js ollama MCP response structure"""
        return (isinstance(response_data, dict) and 
                "message" in response_data and 
                isinstance(response_data["message"], dict) and 
                "content" in response_data["message"])
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get basic server information for diagnostics"""
        return {
            "server_url": self.server_url,
            "model": self.model,
            "connected": self.state.connected,
            "initialized": self.state.initialized,
            "httpx_available": MCP_AVAILABLE,
            "last_request_time": self.state.last_request_time,
            "request_count": self.state.request_count
        }
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update system prompt for different story contexts"""
        self.system_prompt = new_prompt
        if self.debug_logger:
            self.debug_logger.debug("System prompt updated", "MCP")

# Utility functions for external validation
def validate_mcp_request(payload: Dict[str, Any]) -> bool:
    """Validate MCP request payload structure"""
    required_fields = ['model', 'messages']
    
    if not all(field in payload for field in required_fields):
        return False
    
    messages = payload.get('messages', [])
    if not isinstance(messages, list):
        return False
    
    for message in messages:
        if not isinstance(message, dict):
            return False
        if not all(key in message for key in ['role', 'content']):
            return False
        if message['role'] not in ['system', 'user', 'assistant']:
            return False
    
    return True

def quick_mcp_test(server_url: str = MCP_SERVER_URL, model: str = MCP_MODEL) -> bool:
    """Quick connection test to Node.js ollama server without creating full client instance"""
    if not MCP_AVAILABLE:
        return False
    
    try:
        async def _test():
            async with httpx.AsyncClient(timeout=5.0) as client:
                test_payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": False
                }
                try:
                    response = await client.post(server_url, json=test_payload)
                    return response.status_code == 200
                except:
                    return False
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_test())
        loop.close()
        return result
        
    except:
        return False

# Module test when run directly
if __name__ == "__main__":
    print("DevName RPG Client - MCP Communication Module")
    print(f"MCP Available: {MCP_AVAILABLE}")
    print(f"Node.js ollama Server: {MCP_SERVER_URL}")
    print(f"Model: {MCP_MODEL}")
    print(f"Timeout: {MCP_TIMEOUT}s")
    
    if MCP_AVAILABLE:
        print("\nTesting Node.js ollama MCP connection...")
        if quick_mcp_test():
            print("✓ Node.js ollama MCP connection successful")
        else:
            print("✗ Node.js ollama MCP connection failed")
    else:
        print("✗ httpx not available - install with: pip install httpx")

# End of mcp.py - DevName RPG Client MCP Communication Module
# Remodularized for hub-and-spoke architecture with preserved Node.js ollama configuration
