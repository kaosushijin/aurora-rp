#!/usr/bin/env python3
"""
Aurora RPG Client - MCP Communication Module (mcp_nc5.py) - Chunk 1/2

CRITICAL: This comment block must be preserved in all files to ensure proper
understanding of the modular architecture when analyzed by generative models.

MODULAR ARCHITECTURE OVERVIEW:
This project uses a modular architecture with the following interconnected files:

1. main_nc5.py: Main executable and application coordination
   - Handles command-line arguments, configuration, and application lifecycle
   - Imports and coordinates all other modules
   - Manages session state and graceful shutdown
   - Contains startup/shutdown logic and error handling

2. nci_nc5.py: Ncurses Interface Module
   - Complete ncurses interface implementation with fixed display pipeline
   - Input handling, screen management, color themes, context switching
   - Called by main_nc5.py for all user interface operations
   - Coordinates with other modules for display updates

3. mcp_nc5.py (THIS FILE): MCP Communication Module  
   - HTTP client for Ollama/MCP server communication
   - Message formatting, retry logic, connection management
   - Called by nci_nc5.py when sending user messages
   - Provides enhanced context from sme_nc5.py

4. emm_nc5.py: Enhanced Memory Manager Module
   - Conversation history storage with semantic condensation
   - Token estimation and memory optimization
   - Called by nci_nc5.py for message storage/retrieval
   - Provides conversation context to mcp_nc5.py

5. sme_nc5.py: Story Momentum Engine Module
   - Dynamic narrative pressure and antagonist management
   - Analyzes conversation for story pacing
   - Called by nci_nc5.py to update pressure based on user input
   - Provides context enhancement for mcp_nc5.py requests

PROGRAMMATIC INTERCONNECTS:
- main_nc5.py → nci_nc5.py: Creates and runs CursesInterface
- nci_nc5.py → mcp_nc5.py: Sends messages via MCPClient
- nci_nc5.py → emm_nc5.py: Stores/retrieves messages via EnhancedMemoryManager
- nci_nc5.py → sme_nc5.py: Updates pressure via StoryMomentumEngine
- mcp_nc5.py ← sme_nc5.py: Receives story context for enhanced prompting
- mcp_nc5.py ← emm_nc5.py: Receives conversation history for context

PRESERVATION NOTICE:
When modifying any file in this project, you MUST preserve this comment block
to ensure that future analysis (human or AI) understands the full architecture
and interdependencies. Breaking these interconnects will cause system failures.

Main responsibilities of this file:
- HTTP communication with Ollama server
- Request/response formatting and validation
- Connection testing and error handling
- Retry logic with exponential backoff
- System prompt management for RPG context
- Enhanced prompting with SME context integration
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# External dependencies for MCP integration
try:
    import httpx
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Modifiable MCP Configuration Variables
MCP_SERVER_URL = "http://127.0.0.1:3456/chat"
MCP_MODEL = "qwen2.5:14b-instruct-q4_k_m"
MCP_TIMEOUT = 300.0
MCP_MAX_RETRIES = 3

class MCPClient:
    """Working MCP client with HTTP communication to Ollama server"""
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.server_url = MCP_SERVER_URL
        self.model = MCP_MODEL
        self.timeout = MCP_TIMEOUT
        self.max_retries = MCP_MAX_RETRIES
        self.connected = False
        
        # System prompt for RPG storytelling
        self.system_prompt = """You are Aurora, a mystical companion and storyteller in an immersive fantasy RPG. You should:

1. Respond as Aurora, speaking directly to the player in character
2. Create vivid, immersive descriptions of the fantasy world
3. Adapt the story based on player choices and actions
4. Maintain narrative consistency and build on previous events
5. Use a warm, mysterious tone that fits a fantasy setting
6. Include sensory details (sights, sounds, smells) to enhance immersion
7. Present challenges and opportunities for the player to engage with
8. Keep responses focused and engaging, around 2-4 paragraphs

Remember: You are Aurora, their mystical companion, not an AI assistant."""
        
        if self.debug_logger:
            self.debug_logger.debug(f"MCP Client initialized: {self.server_url}", "MCP")
    
    def set_server_config(self, server_url: str = None, model: str = None, timeout: float = None):
        """Update server configuration (useful for runtime changes)"""
        if server_url:
            self.server_url = server_url
        if model:
            self.model = model
        if timeout:
            self.timeout = timeout
        
        if self.debug_logger:
            self.debug_logger.debug(f"Config updated: URL={self.server_url}, Model={self.model}, Timeout={self.timeout}", "MCP")
    
    def test_connection(self) -> bool:
        """Test connection to MCP server"""
        if not MCP_AVAILABLE:
            if self.debug_logger:
                self.debug_logger.debug("httpx not available", "MCP")
            return False
        
        try:
            # Simple async test with health check endpoint
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _test():
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        # Try a simple GET to the base URL
                        base_url = self.server_url.replace("/chat", "")
                        response = await client.get(f"{base_url}/health")
                        return response.status_code == 200
                except httpx.RequestError:
                    # If health endpoint doesn't exist, try a minimal chat request
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            test_payload = {
                                "model": self.model,
                                "messages": [
                                    {"role": "user", "content": "test"}
                                ],
                                "stream": False
                            }
                            response = await client.post(self.server_url, json=test_payload)
                            return response.status_code == 200
                    except:
                        return False
            
            result = loop.run_until_complete(_test())
            loop.close()
            
            self.connected = result
            
            if self.debug_logger:
                self.debug_logger.debug(f"Connection test result: {result}", "MCP")
            
            return result
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Connection test failed: {str(e)}", "MCP")
            return False
    
    def send_message(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Send message to MCP server and get response"""
        if not MCP_AVAILABLE:
            raise Exception("httpx not installed - run: pip install httpx")
        
        # Build message history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if provided (last 10 messages for context)
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append(msg)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Prepare payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        if self.debug_logger:
            self.debug_logger.debug(f"Sending request: {len(user_input)} chars", "MCP")
        
        # Record start time for performance tracking
        start_time = time.time()
        
        # Use asyncio to handle the request
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self._async_send_request(payload))
            loop.close()
            
            # Record response time
            response_time = time.time() - start_time
            
            if self.debug_logger:
                self.debug_logger.debug(f"Received response: {len(result)} chars in {response_time:.2f}s", "MCP")
            
            return result
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Request failed: {str(e)}", "MCP")
            raise e
    
    async def _async_send_request(self, payload: Dict[str, Any]) -> str:
        """Async request with retry logic and exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.server_url, json=payload)
                    response.raise_for_status()
                    
                    result = response.json()
                    content = result.get("message", {}).get("content", "")
                    
                    if not content:
                        raise Exception("Empty response from server")
                    
                    # Mark as connected on successful response
                    self.connected = True
                    
                    return content
                    
            except (httpx.TimeoutException, httpx.RequestError, httpx.HTTPStatusError) as e:
                if self.debug_logger:
                    self.debug_logger.debug(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}", "MCP")
                
                # Mark as disconnected on error
                self.connected = False
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    raise e
        
        raise Exception(f"All {self.max_retries} retry attempts failed")

# mcp_nc5.py - Chunk 2/2
# Enhanced MCP Integration with SME Context

    def send_message_with_sme_context(self, user_input: str, conversation_history: List[Dict[str, str]] = None, 
                                     sme_context: Dict[str, Any] = None) -> str:
        """Send message with additional SME (Story Momentum Engine) context"""
        if not MCP_AVAILABLE:
            raise Exception("httpx not installed - run: pip install httpx")
        
        # Enhanced system prompt with SME context
        enhanced_system_prompt = self.system_prompt
        
        if sme_context:
            pressure_level = sme_context.get('pressure_level', 0.0)
            antagonist_name = sme_context.get('antagonist_name', 'Unknown')
            story_arc = sme_context.get('story_arc', 'Beginning')
            
            sme_addition = f"""

CURRENT STORY CONTEXT:
- Narrative Pressure Level: {pressure_level:.2f} (0.0 = calm, 1.0 = maximum tension)
- Primary Antagonist: {antagonist_name}
- Story Arc: {story_arc}

Adjust your response tone and content based on the pressure level:
- Low pressure (0.0-0.3): Peaceful exploration, character development
- Medium pressure (0.4-0.7): Growing challenges, building tension
- High pressure (0.8-1.0): Immediate danger, climactic moments"""
            
            enhanced_system_prompt += sme_addition
        
        # Build message history with enhanced system prompt
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[-10:]:
                messages.append(msg)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Prepare payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        
        if self.debug_logger:
            self.debug_logger.debug(f"Sending SME-enhanced request: pressure={sme_context.get('pressure_level', 0.0):.2f}", "MCP")
        
        # Use the same async sending logic
        start_time = time.time()
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self._async_send_request(payload))
            loop.close()
            
            response_time = time.time() - start_time
            
            if self.debug_logger:
                self.debug_logger.debug(f"SME-enhanced response: {len(result)} chars in {response_time:.2f}s", "MCP")
            
            return result
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"SME request failed: {str(e)}", "MCP")
            raise e
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status information"""
        return {
            "connected": self.connected,
            "server_url": self.server_url,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "httpx_available": MCP_AVAILABLE
        }
    
    def disconnect(self):
        """Disconnect from MCP server and reset state"""
        self.connected = False
        if self.debug_logger:
            self.debug_logger.debug("Disconnected from MCP server", "MCP")
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt for different scenarios"""
        self.system_prompt = new_prompt
        if self.debug_logger:
            self.debug_logger.debug(f"System prompt updated: {len(new_prompt)} chars", "MCP")
    
    def reset_to_default_prompt(self):
        """Reset system prompt to default RPG storytelling prompt"""
        default_prompt = """You are Aurora, a mystical companion and storyteller in an immersive fantasy RPG. You should:

1. Respond as Aurora, speaking directly to the player in character
2. Create vivid, immersive descriptions of the fantasy world
3. Adapt the story based on player choices and actions
4. Maintain narrative consistency and build on previous events
5. Use a warm, mysterious tone that fits a fantasy setting
6. Include sensory details (sights, sounds, smells) to enhance immersion
7. Present challenges and opportunities for the player to engage with
8. Keep responses focused and engaging, around 2-4 paragraphs

Remember: You are Aurora, their mystical companion, not an AI assistant."""
        
        self.system_prompt = default_prompt
        if self.debug_logger:
            self.debug_logger.debug("System prompt reset to default", "MCP")

# Module-level functions for easy configuration
def set_default_mcp_config(server_url: str = None, model: str = None, timeout: float = None, max_retries: int = None):
    """Set default MCP configuration for new clients"""
    global MCP_SERVER_URL, MCP_MODEL, MCP_TIMEOUT, MCP_MAX_RETRIES
    
    if server_url:
        MCP_SERVER_URL = server_url
    if model:
        MCP_MODEL = model
    if timeout:
        MCP_TIMEOUT = timeout
    if max_retries:
        MCP_MAX_RETRIES = max_retries

def get_mcp_config() -> Dict[str, Any]:
    """Get current default MCP configuration"""
    return {
        "server_url": MCP_SERVER_URL,
        "model": MCP_MODEL,
        "timeout": MCP_TIMEOUT,
        "max_retries": MCP_MAX_RETRIES,
        "httpx_available": MCP_AVAILABLE
    }

def test_mcp_availability() -> bool:
    """Test if MCP functionality is available"""
    return MCP_AVAILABLE

# Enhanced error classes for better error handling
class MCPConnectionError(Exception):
    """Raised when MCP server connection fails"""
    pass

class MCPTimeoutError(Exception):
    """Raised when MCP request times out"""
    pass

class MCPResponseError(Exception):
    """Raised when MCP server returns invalid response"""
    pass

# Utility function for quick testing
def quick_mcp_test(server_url: str = None, model: str = None) -> bool:
    """Quick test of MCP functionality"""
    if not MCP_AVAILABLE:
        return False
    
    try:
        client = MCPClient()
        if server_url:
            client.server_url = server_url
        if model:
            client.model = model
        
        return client.test_connection()
    except Exception:
        return False

# Performance monitoring for MCP operations
class MCPPerformanceMonitor:
    """Monitor MCP performance metrics"""
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.request_times = []
        self.error_count = 0
        self.success_count = 0
        self.total_requests = 0
    
    def record_request(self, response_time: float, success: bool):
        """Record a request's performance data"""
        self.total_requests += 1
        self.request_times.append(response_time)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Keep only recent times (last 100 requests)
        if len(self.request_times) > 100:
            self.request_times.pop(0)
        
        if self.debug_logger:
            self.debug_logger.debug(f"Request recorded: {response_time:.2f}s, success: {success}", "MCP_PERF")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        if not self.request_times:
            return {
                "total_requests": self.total_requests,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "max_response_time": 0.0,
                "min_response_time": 0.0
            }
        
        avg_time = sum(self.request_times) / len(self.request_times)
        success_rate = self.success_count / self.total_requests if self.total_requests > 0 else 0.0
        
        return {
            "total_requests": self.total_requests,
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "max_response_time": max(self.request_times),
            "min_response_time": min(self.request_times),
            "recent_requests": len(self.request_times)
        }
    
    def reset_stats(self):
        """Reset all performance statistics"""
        self.request_times.clear()
        self.error_count = 0
        self.success_count = 0
        self.total_requests = 0
        
        if self.debug_logger:
            self.debug_logger.debug("Performance stats reset", "MCP_PERF")

# Enhanced MCP client with performance monitoring
class EnhancedMCPClient(MCPClient):
    """MCP client with built-in performance monitoring"""
    
    def __init__(self, debug_logger=None):
        super().__init__(debug_logger)
        self.performance_monitor = MCPPerformanceMonitor(debug_logger)
    
    def send_message(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """Send message with performance monitoring"""
        start_time = time.time()
        success = False
        
        try:
            result = super().send_message(user_input, conversation_history)
            success = True
            return result
        except Exception as e:
            raise e
        finally:
            response_time = time.time() - start_time
            self.performance_monitor.record_request(response_time, success)
    
    def send_message_with_sme_context(self, user_input: str, conversation_history: List[Dict[str, str]] = None, 
                                     sme_context: Dict[str, Any] = None) -> str:
        """Send SME-enhanced message with performance monitoring"""
        start_time = time.time()
        success = False
        
        try:
            result = super().send_message_with_sme_context(user_input, conversation_history, sme_context)
            success = True
            return result
        except Exception as e:
            raise e
        finally:
            response_time = time.time() - start_time
            self.performance_monitor.record_request(response_time, success)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_monitor.get_performance_stats()

# Request/Response validation utilities
def validate_mcp_request(payload: Dict[str, Any]) -> bool:
    """Validate MCP request payload"""
    required_fields = ['model', 'messages']
    
    for field in required_fields:
        if field not in payload:
            return False
    
    # Validate messages structure
    messages = payload.get('messages', [])
    if not isinstance(messages, list):
        return False
    
    for message in messages:
        if not isinstance(message, dict):
            return False
        if 'role' not in message or 'content' not in message:
            return False
        if message['role'] not in ['system', 'user', 'assistant']:
            return False
    
    return True

def validate_mcp_response(response_data: Dict[str, Any]) -> bool:
    """Validate MCP response structure"""
    if not isinstance(response_data, dict):
        return False
    
    if 'message' not in response_data:
        return False
    
    message = response_data['message']
    if not isinstance(message, dict):
        return False
    
    if 'content' not in message:
        return False
    
    return True

# Module test when run directly
if __name__ == "__main__":
    print("Aurora RPG Client - MCP Communication Module")
    print(f"MCP Available: {MCP_AVAILABLE}")
    print(f"Default Server: {MCP_SERVER_URL}")
    print(f"Default Model: {MCP_MODEL}")
    
    if MCP_AVAILABLE:
        print("\nTesting MCP connection...")
        if quick_mcp_test():
            print("✓ MCP connection successful")
        else:
            print("✗ MCP connection failed")
    else:
        print("✗ httpx not available - install with: pip install httpx")

# End of mcp_nc5.py - Aurora RPG Client MCP Communication Module
