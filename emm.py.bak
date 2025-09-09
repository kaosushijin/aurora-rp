#!/usr/bin/env python3
# emm.py - Enhanced Memory Manager Module
# CRITICAL: Read genai.txt to understand the holistic program across modules.

import json
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import threading
import asyncio
import httpx

class MessageType(Enum):
    """Message type enumeration for conversation tracking"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message:
    """Individual conversation message with metadata"""
    
    def __init__(self, content: str, message_type: MessageType, timestamp: Optional[str] = None):
        self.content = content
        self.message_type = message_type
        self.timestamp = timestamp or datetime.now().isoformat()
        self.token_estimate = self._estimate_tokens(content)
        self.id = str(uuid4())
    
    def _estimate_tokens(self, text: str) -> int:
        """Conservative token estimation for memory planning"""
        if not text:
            return 0
        # Rough estimation: 1 token per 4 characters
        return max(1, len(text) // 4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.message_type.value,
            "timestamp": self.timestamp,
            "tokens": self.token_estimate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        msg = cls(
            content=data["content"],
            message_type=MessageType(data["type"]),
            timestamp=data["timestamp"]
        )
        msg.id = data.get("id", str(uuid4()))
        return msg

class EnhancedMemoryManager:
    """Memory management with LLM-powered semantic condensation"""
    
    def __init__(self, max_memory_tokens: int = 16000, debug_logger=None):
        self.max_memory_tokens = max_memory_tokens
        self.debug_logger = debug_logger
        self.messages: List[Message] = []
        self.condensation_count = 0
        self.lock = threading.Lock()
        
        # Semantic category preservation ratios
        self.preservation_ratios = {
            "story_critical": 0.9,
            "character_focused": 0.8,
            "relationship_dynamics": 0.8,
            "emotional_significance": 0.75,
            "world_building": 0.7,
            "standard": 0.4
        }
        
        # MCP client configuration
        self.mcp_config = self._load_mcp_config()
        
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration for LLM calls"""
        return {
            "server_url": "http://127.0.0.1:3456/chat",
            "model": "qwen2.5:14b-instruct-q4_k_m",
            "timeout": 300
        }
    
    def add_message(self, content: str, message_type: MessageType) -> None:
        """Add new message and manage memory"""
        with self.lock:
            message = Message(content, message_type)
            self.messages.append(message)
            
            if self.debug_logger:
                self.debug_logger.memory(f"Added {message_type.value} message: {len(content)} chars, {message.token_estimate} tokens")
            
            # Check if condensation needed
            current_tokens = sum(msg.token_estimate for msg in self.messages)
            if current_tokens > self.max_memory_tokens:
                self._perform_semantic_condensation()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Retrieve messages with optional limit"""
        with self.lock:
            if limit:
                return self.messages[-limit:]
            return self.messages.copy()
    
    def get_conversation_for_mcp(self) -> List[Dict[str, str]]:
        """Format conversation for MCP requests"""
        with self.lock:
            return [
                {"role": msg.message_type.value, "content": msg.content}
                for msg in self.messages
            ]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Return current memory statistics"""
        with self.lock:
            total_tokens = sum(msg.token_estimate for msg in self.messages)
            return {
                "message_count": len(self.messages),
                "total_tokens": total_tokens,
                "max_tokens": self.max_memory_tokens,
                "utilization": total_tokens / self.max_memory_tokens,
                "condensations_performed": self.condensation_count
            }

# Chunk 2 - LLM Semantic Analysis and Condensation Logic

    async def _call_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make LLM request for semantic analysis"""
        try:
            async with httpx.AsyncClient(timeout=self.mcp_config.get("timeout", 30)) as client:
                response = await client.post(
                    f"{self.mcp_config['server_url']}/api/chat",
                    json={
                        "model": self.mcp_config.get("model", "llama3.1:8b"),
                        "messages": messages,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")
                    
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"LLM call failed: {e}")
            return None
    
    async def _analyze_message_semantics(self, target_idx: int) -> Optional[Dict[str, Any]]:
        """Analyze message semantics with context window - 3 retry attempts"""
        
        # Create context window (5 before + target + 5 after)
        start_idx = max(0, target_idx - 5)
        end_idx = min(len(self.messages), target_idx + 6)
        context_messages = self.messages[start_idx:end_idx]
        
        target_message = self.messages[target_idx]
        
        # Attempt 1: Full analysis with fragmentation
        prompt1 = self._create_full_analysis_prompt(context_messages, target_idx - start_idx)
        result = await self._call_llm([{"role": "system", "content": prompt1}])
        
        if result:
            parsed = self._parse_semantic_response(result, attempt=1)
            if parsed:
                return parsed
        
        # Attempt 2: Simplified analysis
        prompt2 = self._create_simple_analysis_prompt(target_message.content)
        result = await self._call_llm([{"role": "system", "content": prompt2}])
        
        if result:
            parsed = self._parse_semantic_response(result, attempt=2)
            if parsed:
                return parsed
        
        # Attempt 3: Binary preserve/condense decision
        prompt3 = self._create_binary_prompt(target_message.content)
        result = await self._call_llm([{"role": "system", "content": prompt3}])
        
        if result:
            parsed = self._parse_semantic_response(result, attempt=3)
            if parsed:
                return parsed
        
        # All attempts failed - return default
        return {
            "importance_score": 0.4,
            "categories": ["standard"],
            "fragments": None
        }
    
    def _create_full_analysis_prompt(self, context_messages: List[Message], target_idx: int) -> str:
        """Create detailed semantic analysis prompt"""
        context_text = "\n".join([
            f"[{i}] {msg.message_type.value}: {msg.content}"
            for i, msg in enumerate(context_messages)
        ])
        
        return f"""Analyze message [{target_idx}] in context for semantic importance and categorization.

Context:
{context_text}

Categories:
- story_critical: Major plot developments, character deaths, world-changing events
- character_focused: Relationship changes, character development, personality reveals
- relationship_dynamics: Evolving relationships between characters
- emotional_significance: Dramatic moments, trust/betrayal, conflict resolution
- world_building: New locations, lore, cultural info, political changes
- standard: General interactions, travel, routine activities

If the target message contains multiple semantic elements, fragment it.

Return JSON format:
{{
  "importance_score": 0.0-1.0,
  "categories": ["category1", "category2"],
  "fragments": [
    {{"text": "portion1", "categories": ["category"], "importance": 0.0-1.0}},
    {{"text": "portion2", "categories": ["category"], "importance": 0.0-1.0}}
  ]
}}

For single-category messages, set fragments to null."""
    
    def _create_simple_analysis_prompt(self, content: str) -> str:
        """Create simplified analysis prompt"""
        return f"""Categorize this message and rate its story importance:

Message: {content}

Categories: story_critical, character_focused, relationship_dynamics, emotional_significance, world_building, standard

Return JSON:
{{
  "importance_score": 0.0-1.0,
  "categories": ["primary_category"]
}}"""
    
    def _create_binary_prompt(self, content: str) -> str:
        """Create binary preserve/condense prompt"""
        return f"""Should this message be preserved or condensed?

Message: {content}

Return JSON:
{{
  "preserve": true/false
}}"""
    
    def _parse_semantic_response(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Parse LLM semantic analysis response"""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                if attempt == 3:  # Binary response
                    preserve = data.get("preserve", False)
                    return {
                        "importance_score": 0.8 if preserve else 0.2,
                        "categories": ["story_critical"] if preserve else ["standard"],
                        "fragments": None
                    }
                
                elif attempt == 2:  # Simple response
                    return {
                        "importance_score": data.get("importance_score", 0.4),
                        "categories": data.get("categories", ["standard"]),
                        "fragments": None
                    }
                
                else:  # Full response
                    return data
                    
        except (json.JSONDecodeError, KeyError) as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to parse semantic response attempt {attempt}: {e}")
            
        return None

# Chunk 3 - Condensation Execution and File Persistence

    def _perform_semantic_condensation(self) -> None:
        """Execute LLM-powered semantic condensation"""
        if len(self.messages) < 10:  # Need minimum messages for context
            return
            
        # Run async condensation in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_condensation())
        finally:
            loop.close()
    
    async def _async_condensation(self) -> None:
        """Async condensation with category-weighted preservation"""
        aggressiveness_level = 0
        max_passes = 3
        
        for pass_num in range(max_passes):
            current_tokens = sum(msg.token_estimate for msg in self.messages)
            
            if current_tokens <= self.max_memory_tokens:
                break
                
            if self.debug_logger:
                self.debug_logger.memory(f"Condensation pass {pass_num + 1}, tokens: {current_tokens}")
            
            # Analyze messages for semantic importance
            message_analyses = []
            preserve_messages = []
            condense_candidates = []
            
            # Skip recent messages (last 5) from condensation
            condensable_range = len(self.messages) - 5
            
            for i in range(min(condensable_range, len(self.messages))):
                analysis = await self._analyze_message_semantics(i)
                message_analyses.append((i, analysis))
                
                # Determine preservation based on categories and aggressiveness
                should_preserve = self._should_preserve_message(analysis, aggressiveness_level)
                
                if should_preserve:
                    preserve_messages.append(self.messages[i])
                else:
                    condense_candidates.append(self.messages[i])
            
            # Always preserve recent messages
            preserve_messages.extend(self.messages[condensable_range:])
            
            if condense_candidates:
                # Create condensed summary of candidates
                condensed_message = await self._create_condensed_summary(condense_candidates)
                
                if condensed_message:
                    # Replace condensed messages with summary
                    self.messages = [condensed_message] + preserve_messages
                    self.condensation_count += 1
                    
                    if self.debug_logger:
                        new_tokens = sum(msg.token_estimate for msg in self.messages)
                        self.debug_logger.memory(
                            f"Condensed {len(condense_candidates)} messages, "
                            f"tokens: {current_tokens} → {new_tokens}"
                        )
                else:
                    break  # Condensation failed, exit
            else:
                # No candidates for condensation, increase aggressiveness
                aggressiveness_level += 1
                if self.debug_logger:
                    self.debug_logger.memory(f"Increased aggressiveness to level {aggressiveness_level}")
    
    def _should_preserve_message(self, analysis: Dict[str, Any], aggressiveness: int) -> bool:
        """Determine if message should be preserved based on analysis and aggressiveness"""
        categories = analysis.get("categories", ["standard"])
        importance = analysis.get("importance_score", 0.4)
        
        # Get highest preservation ratio for multi-category messages
        max_ratio = max(
            self.preservation_ratios.get(cat, 0.4) 
            for cat in categories
        )
        
        # Apply aggressiveness reduction
        adjusted_ratio = max(0.2, max_ratio - (aggressiveness * 0.1))
        
        # Preserve based on importance score vs adjusted ratio
        return importance >= adjusted_ratio
    
    async def _create_condensed_summary(self, messages: List[Message]) -> Optional[Message]:
        """Create condensed summary of message group"""
        if not messages:
            return None
            
        # Format messages for condensation
        content_to_condense = "\n".join([
            f"[{msg.message_type.value}] {msg.content}"
            for msg in messages
        ])
        
        prompt = (
            "Condense the following conversation messages into a concise summary that preserves "
            "key facts, character interactions, story developments, and emotional significance. "
            "Maintain narrative continuity while minimizing length.\n\n"
            f"{content_to_condense}\n\n"
            "Return only the condensed summary text."
        )
        
        summary_content = await self._call_llm([{"role": "system", "content": prompt}])
        
        if summary_content:
            return Message(
                content=f"[CONDENSED] {summary_content}",
                message_type=MessageType.SYSTEM
            )
        
        return None
    
    def save_conversation(self, filename: Optional[str] = None) -> bool:
        """Save conversation to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_history_{timestamp}.json"
        
        try:
            with self.lock:
                conversation_data = {
                    "metadata": {
                        "saved_at": datetime.now().isoformat(),
                        "message_count": len(self.messages),
                        "total_tokens": sum(msg.token_estimate for msg in self.messages),
                        "condensations": self.condensation_count
                    },
                    "messages": [msg.to_dict() for msg in self.messages]
                }
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            if self.debug_logger:
                self.debug_logger.system(f"Conversation saved to {filename}")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save conversation: {e}")
            return False
    
    def load_conversation(self, filename: str) -> bool:
        """Load conversation from file"""
        try:
            if not os.path.exists(filename):
                return False
                
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            with self.lock:
                self.messages = [
                    Message.from_dict(msg_data) 
                    for msg_data in data.get("messages", [])
                ]
                
                metadata = data.get("metadata", {})
                self.condensation_count = metadata.get("condensations", 0)
            
            if self.debug_logger:
                self.debug_logger.system(f"Conversation loaded from {filename}")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to load conversation: {e}")
            return False
    
    def clear_memory(self) -> None:
        """Clear all stored messages"""
        with self.lock:
            self.messages.clear()
            self.condensation_count = 0
            
        if self.debug_logger:
            self.debug_logger.memory("Memory cleared")
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Generate conversation statistics for debugging"""
        with self.lock:
            if not self.messages:
                return {"status": "no_messages"}
            
            message_types = {}
            for msg in self.messages:
                msg_type = msg.message_type.value
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
            
            total_tokens = sum(msg.token_estimate for msg in self.messages)
            avg_tokens = total_tokens / len(self.messages) if self.messages else 0
            
            return {
                "total_messages": len(self.messages),
                "message_types": message_types,
                "total_tokens": total_tokens,
                "average_tokens_per_message": round(avg_tokens, 2),
                "memory_utilization": round(total_tokens / self.max_memory_tokens, 3),
                "condensations_performed": self.condensation_count,
                "oldest_message": self.messages[0].timestamp if self.messages else None,
                "newest_message": self.messages[-1].timestamp if self.messages else None
            }

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Enhanced Memory Manager Module")
    print("Testing memory management functionality...")
    
    emm = EnhancedMemoryManager()
    
    # Test basic functionality
    emm.add_message("Hello there!", MessageType.USER)
    emm.add_message("Greetings, traveler!", MessageType.ASSISTANT)
    
    stats = emm.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    patterns = emm.analyze_conversation_patterns()
    print(f"Conversation patterns: {patterns}")
    
    print("Memory manager test completed successfully.")
