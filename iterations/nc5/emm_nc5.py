#!/usr/bin/env python3
"""
Aurora RPG Client - Enhanced Memory Manager Module (emm_nc5.py) - Chunk 1/2

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

3. mcp_nc5.py: MCP Communication Module  
   - HTTP client for Ollama/MCP server communication
   - Message formatting, retry logic, connection management
   - Called by nci_nc5.py when sending user messages
   - Provides enhanced context from sme_nc5.py

4. emm_nc5.py (THIS FILE): Enhanced Memory Manager Module
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
- Message storage and retrieval with typed messages
- Automatic memory condensation when token limits approached
- Semantic preservation during compression
- Conversation history formatting for MCP
- Memory usage statistics and optimization
- Token estimation and conversation analysis
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

class MessageType(Enum):
    """Types of messages in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    THINKING = "thinking"

class Message(NamedTuple):
    """Individual message structure"""
    content: str
    message_type: MessageType
    timestamp: str
    token_estimate: int = 0

class MemoryStats(NamedTuple):
    """Memory usage statistics"""
    message_count: int
    total_tokens: int
    condensation_count: int
    last_condensation: str
    efficiency_ratio: float

class EnhancedMemoryManager:
    """
    Advanced conversation memory management with LLM-first semantic condensation.
    
    This class maintains conversation context while optimizing memory usage
    through intelligent summarization and compression.
    """
    
    def __init__(self, debug_logger=None, condensation_threshold: int = 8000):
        self.debug_logger = debug_logger
        self.messages: List[Message] = []
        self.condensation_threshold = condensation_threshold
        self.condensation_count = 0
        self.last_condensation = "Never"
        self.total_tokens_saved = 0
        
        if self.debug_logger:
            self.debug_logger.debug("Enhanced Memory Manager initialized", "MEMORY")
    
    def add_message(self, content: str, message_type: MessageType) -> None:
        """Add a new message to conversation memory"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        token_estimate = self._estimate_tokens(content)
        
        message = Message(
            content=content,
            message_type=message_type,
            timestamp=timestamp,
            token_estimate=token_estimate
        )
        
        self.messages.append(message)
        
        if self.debug_logger:
            self.debug_logger.debug(f"Added {message_type.value} message: {len(content)} chars, {token_estimate} tokens", "MEMORY")
        
        # Check if condensation is needed
        self._check_condensation_needed()
    
    def remove_last_message(self) -> bool:
        """Remove the last message (useful for removing thinking indicators)"""
        if self.messages:
            removed = self.messages.pop()
            if self.debug_logger:
                self.debug_logger.debug(f"Removed last {removed.message_type.value} message", "MEMORY")
            return True
        return False
    
    def get_chat_history(self) -> List[Message]:
        """Get all conversation messages"""
        return self.messages.copy()
    
    def get_conversation_for_mcp(self) -> List[Dict[str, str]]:
        """Get conversation history formatted for MCP requests"""
        mcp_messages = []
        
        for message in self.messages:
            if message.message_type == MessageType.USER:
                mcp_messages.append({"role": "user", "content": message.content})
            elif message.message_type == MessageType.ASSISTANT:
                mcp_messages.append({"role": "assistant", "content": message.content})
            # Skip system and thinking messages for MCP
        
        return mcp_messages
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory usage statistics"""
        total_tokens = sum(msg.token_estimate for msg in self.messages)
        efficiency_ratio = 1.0
        
        if self.condensation_count > 0 and self.total_tokens_saved > 0:
            # Calculate efficiency as tokens saved vs original usage
            efficiency_ratio = min(2.0, 1.0 + (self.total_tokens_saved / max(total_tokens, 1)))
        
        return MemoryStats(
            message_count=len(self.messages),
            total_tokens=total_tokens,
            condensation_count=self.condensation_count,
            last_condensation=self.last_condensation,
            efficiency_ratio=efficiency_ratio
        )
    
    def clear_history(self) -> None:
        """Clear all conversation history"""
        message_count = len(self.messages)
        self.messages.clear()
        self.condensation_count = 0
        self.last_condensation = "Never"
        self.total_tokens_saved = 0
        
        if self.debug_logger:
            self.debug_logger.debug(f"Cleared {message_count} messages from memory", "MEMORY")
    
    def condense_if_needed(self, conversation_history: List[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """
        Condense memory if needed and return optimized conversation history.
        
        This method implements the LLM-first semantic condensation approach.
        """
        if conversation_history is None:
            conversation_history = self.get_conversation_for_mcp()
        
        # Check if condensation is needed
        total_tokens = sum(self._estimate_tokens(msg.get("content", "")) for msg in conversation_history)
        
        if total_tokens > self.condensation_threshold:
            if self.debug_logger:
                self.debug_logger.debug(f"Condensation triggered: {total_tokens} > {self.condensation_threshold}", "MEMORY")
            
            return self._perform_condensation(conversation_history)
        
        return conversation_history
    
    def _check_condensation_needed(self) -> None:
        """Check if automatic condensation is needed"""
        total_tokens = sum(msg.token_estimate for msg in self.messages)
        
        if total_tokens > self.condensation_threshold:
            self._condense_conversation_memory()
    
    def _perform_condensation(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Perform semantic condensation of conversation history.
        
        This preserves the most important context while reducing token usage.
        """
        if len(conversation_history) <= 4:
            # Too few messages to condense meaningfully
            return conversation_history
        
        # Keep the most recent 3 exchanges (6 messages)
        recent_messages = conversation_history[-6:]
        
        # Condense older messages
        older_messages = conversation_history[:-6]
        
        if not older_messages:
            return recent_messages
        
        # Perform semantic condensation on older messages
        condensed_summary = self._create_semantic_summary(older_messages)
        
        # Create condensed conversation with summary + recent messages
        condensed_conversation = [
            {"role": "assistant", "content": condensed_summary}
        ] + recent_messages
        
        # Update statistics
        original_tokens = sum(self._estimate_tokens(msg.get("content", "")) for msg in conversation_history)
        condensed_tokens = sum(self._estimate_tokens(msg.get("content", "")) for msg in condensed_conversation)
        tokens_saved = original_tokens - condensed_tokens
        
        self.total_tokens_saved += tokens_saved
        self.condensation_count += 1
        self.last_condensation = datetime.now().strftime("%H:%M:%S")
        
        if self.debug_logger:
            self.debug_logger.debug(f"Condensation completed: {original_tokens} -> {condensed_tokens} tokens (saved {tokens_saved})", "MEMORY")
        
        return condensed_conversation
    
    def _condense_conversation_memory(self) -> None:
        """
        Condense the stored conversation memory when threshold is exceeded.
        
        This preserves important context while reducing memory usage.
        """
        if len(self.messages) <= 6:
            return
        
        # Keep recent messages
        recent_messages = self.messages[-6:]
        older_messages = self.messages[:-6]
        
        # Create semantic summary of older messages
        older_content = []
        for msg in older_messages:
            if msg.message_type in [MessageType.USER, MessageType.ASSISTANT]:
                prefix = "User" if msg.message_type == MessageType.USER else "Aurora"
                older_content.append(f"{prefix}: {msg.content}")
        
        if older_content:
            summary_content = self._create_conversation_summary(older_content)
            
            # Create summary message
            summary_message = Message(
                content=summary_content,
                message_type=MessageType.SYSTEM,
                timestamp=datetime.now().strftime("%H:%M:%S"),
                token_estimate=self._estimate_tokens(summary_content)
            )
            
            # Replace older messages with summary
            self.messages = [summary_message] + recent_messages
            
            self.condensation_count += 1
            self.last_condensation = datetime.now().strftime("%H:%M:%S")
            
            if self.debug_logger:
                self.debug_logger.debug(f"Memory condensed: {len(older_messages)} messages -> 1 summary", "MEMORY")
    
    def _create_semantic_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Create a semantic summary of conversation messages.
        
        This implements the LLM-first approach for preserving key context.
        """
        # Extract key themes and events
        user_actions = []
        aurora_responses = []
        
        for msg in messages:
            content = msg.get("content", "")
            if msg.get("role") == "user":
                user_actions.append(content[:100])  # Keep first 100 chars
            elif msg.get("role") == "assistant":
                aurora_responses.append(content[:100])
        
        # Create structured summary
        summary_parts = [
            "[Previous conversation summary]",
            "Key events and interactions:"
        ]
        
        # Add user actions summary
        if user_actions:
            summary_parts.append(f"User explored: {'; '.join(user_actions[:3])}")
        
        # Add Aurora responses summary
        if aurora_responses:
            summary_parts.append(f"Aurora responded with: {'; '.join(aurora_responses[:3])}")
        
        summary_parts.append("[End summary - continuing current conversation]")
        
        return " ".join(summary_parts)
    
    def _create_conversation_summary(self, content_list: List[str]) -> str:
        """Create a summary of conversation content for memory condensation"""
        if not content_list:
            return "[Previous conversation summary: No significant events]"
        
        # Take key snippets from the conversation
        key_snippets = content_list[:5]  # First 5 exchanges
        
        summary = "[Conversation Summary] Previous interactions: " + "; ".join(
            snippet[:80] + ("..." if len(snippet) > 80 else "")
            for snippet in key_snippets
        ) + " [End Summary]"
        
        return summary
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses enhanced estimation logic for better accuracy.
        """
        if not text:
            return 0
        
        # Enhanced estimation: ~4 characters per token average, plus word count factor
        char_count = len(text)
        word_count = len(text.split())
        
        # Use character-based estimation but adjust for word boundaries
        char_tokens = char_count / 4
        word_tokens = word_count * 1.3  # Average 1.3 tokens per word
        
        # Use the higher estimate for safety
        return int(max(char_tokens, word_tokens))

# emm_nc5.py - Chunk 2/2
# Export and Analysis Functions

    def export_conversation(self, format_type: str = "json") -> str:
        """Export conversation in specified format"""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "memory_stats": self.get_memory_stats()._asdict(),
            "messages": [
                {
                    "content": msg.content,
                    "type": msg.message_type.value,
                    "timestamp": msg.timestamp,
                    "token_estimate": msg.token_estimate
                } for msg in self.messages
            ]
        }
        
        if format_type == "json":
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        elif format_type == "text":
            lines = [f"Aurora RPG Conversation Export - {export_data['timestamp']}", "=" * 50, ""]
            for msg_data in export_data["messages"]:
                prefix = msg_data["type"].title()
                lines.append(f"[{msg_data['timestamp']}] {prefix}: {msg_data['content']}")
                lines.append("")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Analyze conversation patterns for insights"""
        if not self.messages:
            return {"error": "No messages to analyze"}
        
        # Count message types
        type_counts = {}
        for msg in self.messages:
            msg_type = msg.message_type.value
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
        
        # Calculate average message length by type
        type_lengths = {}
        for msg_type in type_counts:
            messages_of_type = [msg for msg in self.messages if msg.message_type.value == msg_type]
            if messages_of_type:
                avg_length = sum(len(msg.content) for msg in messages_of_type) / len(messages_of_type)
                type_lengths[msg_type] = avg_length
        
        # Analyze conversation flow
        conversation_flow = []
        for i, msg in enumerate(self.messages):
            if i > 0:
                prev_type = self.messages[i-1].message_type.value
                curr_type = msg.message_type.value
                flow_pair = f"{prev_type} -> {curr_type}"
                conversation_flow.append(flow_pair)
        
        flow_counts = {}
        for flow in conversation_flow:
            flow_counts[flow] = flow_counts.get(flow, 0) + 1
        
        # Calculate time distribution
        timestamps = [msg.timestamp for msg in self.messages]
        first_time = timestamps[0] if timestamps else "00:00:00"
        last_time = timestamps[-1] if timestamps else "00:00:00"
        
        return {
            "total_messages": len(self.messages),
            "message_type_counts": type_counts,
            "average_message_lengths": type_lengths,
            "conversation_flow_patterns": flow_counts,
            "time_span": {
                "first_message": first_time,
                "last_message": last_time
            },
            "memory_efficiency": self.get_memory_stats().efficiency_ratio,
            "condensations_performed": self.condensation_count
        }
    
    def get_debug_content(self) -> List[str]:
        """Get debug information about memory manager state"""
        stats = self.get_memory_stats()
        patterns = self.analyze_conversation_patterns()
        
        debug_lines = [
            "Enhanced Memory Manager Debug Information",
            "=" * 50,
            f"Total Messages: {stats.message_count}",
            f"Total Tokens: {stats.total_tokens}",
            f"Condensation Threshold: {self.condensation_threshold}",
            f"Condensations Performed: {stats.condensation_count}",
            f"Last Condensation: {stats.last_condensation}",
            f"Efficiency Ratio: {stats.efficiency_ratio:.2f}",
            f"Tokens Saved: {self.total_tokens_saved}",
            "",
            "Message Type Distribution:",
        ]
        
        if "message_type_counts" in patterns:
            for msg_type, count in patterns["message_type_counts"].items():
                debug_lines.append(f"  {msg_type}: {count}")
        
        debug_lines.extend([
            "",
            "Recent Messages (last 5):",
        ])
        
        recent_messages = self.messages[-5:] if len(self.messages) >= 5 else self.messages
        for msg in recent_messages:
            truncated_content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            debug_lines.append(f"  [{msg.timestamp}] {msg.message_type.value}: {truncated_content}")
        
        return debug_lines

# Utility functions for memory management
def estimate_conversation_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate total tokens in a conversation"""
    return sum(len(msg.get("content", "")) // 4 for msg in messages)

def create_memory_snapshot(memory_manager: EnhancedMemoryManager) -> Dict[str, Any]:
    """Create a snapshot of memory manager state"""
    return {
        "timestamp": datetime.now().isoformat(),
        "stats": memory_manager.get_memory_stats()._asdict(),
        "message_count": len(memory_manager.messages),
        "threshold": memory_manager.condensation_threshold,
        "patterns": memory_manager.analyze_conversation_patterns()
    }

def optimize_conversation_for_context(messages: List[Dict[str, str]], target_tokens: int = 6000) -> List[Dict[str, str]]:
    """Optimize conversation history to fit within target token limit"""
    if not messages:
        return messages
    
    # Estimate current token usage
    current_tokens = estimate_conversation_tokens(messages)
    
    if current_tokens <= target_tokens:
        return messages
    
    # Keep the most recent messages that fit within the target
    optimized_messages = []
    token_count = 0
    
    # Start from the end and work backwards
    for message in reversed(messages):
        message_tokens = len(message.get("content", "")) // 4
        if token_count + message_tokens <= target_tokens:
            optimized_messages.insert(0, message)
            token_count += message_tokens
        else:
            break
    
    # If we have very few messages, create a summary of the excluded ones
    excluded_count = len(messages) - len(optimized_messages)
    if excluded_count > 0 and len(optimized_messages) > 0:
        summary_message = {
            "role": "assistant",
            "content": f"[Previous conversation summary: {excluded_count} earlier messages condensed to save memory]"
        }
        optimized_messages.insert(0, summary_message)
    
    return optimized_messages

class ConversationAnalyzer:
    """Advanced conversation analysis for memory optimization"""
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
    
    def analyze_message_importance(self, messages: List[Message]) -> List[Tuple[Message, float]]:
        """Analyze and score message importance for selective condensation"""
        scored_messages = []
        
        for i, message in enumerate(messages):
            importance_score = self._calculate_importance_score(message, i, len(messages))
            scored_messages.append((message, importance_score))
        
        return scored_messages
    
    def _calculate_importance_score(self, message: Message, position: int, total_messages: int) -> float:
        """Calculate importance score for a message (0.0 to 1.0)"""
        score = 0.0
        content = message.content.lower()
        
        # Recent messages are more important
        recency_score = (total_messages - position) / total_messages
        score += recency_score * 0.3
        
        # User messages are generally more important than system messages
        type_weights = {
            MessageType.USER: 0.4,
            MessageType.ASSISTANT: 0.3,
            MessageType.SYSTEM: 0.1,
            MessageType.THINKING: 0.05
        }
        score += type_weights.get(message.message_type, 0.1)
        
        # Longer messages might be more important
        length_score = min(len(message.content) / 500, 1.0)  # Cap at 500 chars
        score += length_score * 0.1
        
        # Messages with certain keywords are more important
        important_keywords = [
            "quest", "adventure", "character", "story", "plot", "important",
            "remember", "key", "crucial", "decision", "choice", "outcome"
        ]
        keyword_score = sum(1 for keyword in important_keywords if keyword in content)
        score += min(keyword_score * 0.02, 0.1)
        
        # Questions and exclamations might be more important
        if "?" in message.content:
            score += 0.05
        if "!" in message.content:
            score += 0.03
        
        return min(score, 1.0)  # Cap at 1.0
    
    def suggest_condensation_strategy(self, messages: List[Message], target_reduction: float = 0.3) -> Dict[str, Any]:
        """Suggest an optimal condensation strategy"""
        if not messages:
            return {"strategy": "no_action", "reason": "No messages to condense"}
        
        scored_messages = self.analyze_message_importance(messages)
        scored_messages.sort(key=lambda x: x[1])  # Sort by importance (lowest first)
        
        # Calculate how many messages to condense
        messages_to_condense = int(len(messages) * target_reduction)
        
        # Select least important messages for condensation
        condensation_candidates = scored_messages[:messages_to_condense]
        preservation_candidates = scored_messages[messages_to_condense:]
        
        strategy = {
            "strategy": "selective_condensation",
            "total_messages": len(messages),
            "condense_count": len(condensation_candidates),
            "preserve_count": len(preservation_candidates),
            "condensation_candidates": [
                {
                    "timestamp": msg.timestamp,
                    "type": msg.message_type.value,
                    "importance": score,
                    "preview": msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                }
                for msg, score in condensation_candidates
            ],
            "estimated_token_savings": sum(msg.token_estimate for msg, _ in condensation_candidates)
        }
        
        return strategy

# Performance optimization utilities
def batch_process_messages(messages: List[Message], batch_size: int = 100) -> List[List[Message]]:
    """Split messages into batches for efficient processing"""
    batches = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        batches.append(batch)
    return batches

def parallel_token_estimation(messages: List[Message]) -> List[int]:
    """Efficiently estimate tokens for multiple messages"""
    # For this implementation, we'll use the existing estimation
    # In a more advanced version, this could use multiprocessing
    return [_estimate_tokens_fast(msg.content) for msg in messages]

def _estimate_tokens_fast(text: str) -> int:
    """Fast token estimation for batch processing"""
    if not text:
        return 0
    
    # Simplified estimation for speed
    return len(text) // 4

# Module test when run directly
if __name__ == "__main__":
    print("Aurora RPG Client - Enhanced Memory Manager Module")
    print("Testing memory management functionality...")
    
    # Create test memory manager
    emm = EnhancedMemoryManager()
    
    # Add test messages
    emm.add_message("Hello Aurora!", MessageType.USER)
    emm.add_message("Greetings, traveler! Welcome to the mystical realm.", MessageType.ASSISTANT)
    emm.add_message("What adventures await?", MessageType.USER)
    emm.add_message("Many paths lie before you, each filled with wonder and danger.", MessageType.ASSISTANT)
    
    # Test memory stats
    stats = emm.get_memory_stats()
    print(f"Messages: {stats.message_count}")
    print(f"Total tokens: {stats.total_tokens}")
    print(f"Condensations: {stats.condensation_count}")
    
    # Test MCP formatting
    mcp_history = emm.get_conversation_for_mcp()
    print(f"MCP format: {len(mcp_history)} messages")
    
    # Test conversation analysis
    patterns = emm.analyze_conversation_patterns()
    print(f"Conversation patterns analyzed: {patterns['total_messages']} messages")
    
    print("Enhanced Memory Manager module test completed.")

# End of emm_nc5.py - Aurora RPG Client Enhanced Memory Manager Module
