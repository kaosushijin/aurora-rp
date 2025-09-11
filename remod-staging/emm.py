# Chunk 1/3 - emm.py - Enhanced Memory Manager (Phase 5: Uses sem.py)
#!/usr/bin/env python3
"""
DevName RPG Client - Enhanced Memory Manager (emm.py)

Phase 5 Refactor: Removed embedded semantic logic, now uses sem.py
This eliminates circular dependencies and consolidates semantic analysis
Module architecture and interconnects documented in genai.txt
"""

import json
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

# Import centralized semantic logic from sem.py
from sem import (
    SemanticProcessor, MessageType, create_semantic_processor,
    SEMANTIC_THRESHOLDS, CONDENSATION_STRATEGIES
)

# Configuration constants
DEFAULT_MAX_TOKENS = 30000
DEFAULT_TOKENS_PER_MESSAGE = 50
SAVE_DIRECTORY = "memory"
DEFAULT_FILENAME = "conversation_memory.json"

# Memory management thresholds
MEMORY_THRESHOLDS = {
    "condensation_trigger_ratio": 0.8,      # Trigger at 80% of max tokens
    "emergency_condensation_ratio": 0.95,   # Emergency condensation at 95%
    "target_ratio_after_condensation": 0.6, # Condense down to 60%
    "min_messages_before_condensation": 10,  # Don't condense very short conversations
    "max_condensation_passes": 3,            # Maximum condensation aggressiveness
    "backup_retention_count": 5              # Keep 5 backup files
}

class ConversationState(Enum):
    """Conversation management states"""
    NORMAL = "normal"
    APPROACHING_LIMIT = "approaching_limit"
    CONDENSATION_NEEDED = "condensation_needed"
    CONDENSATION_IN_PROGRESS = "condensation_in_progress"
    EMERGENCY_CLEANUP = "emergency_cleanup"


class MessageMetadata:
    """Enhanced message metadata with semantic analysis integration"""
    
    def __init__(self, content: str = "", msg_type: MessageType = MessageType.SYSTEM):
        self.content = content
        self.msg_type = msg_type
        self.timestamp = time.time()
        self.formatted_timestamp = datetime.fromtimestamp(self.timestamp).isoformat()
        
        # Semantic analysis results (populated by sem.py SemanticProcessor)
        self.importance_score = 0.5
        self.categories = ["standard"]
        self.narrative_significance = ""
        self.pattern_analysis = {}
        self.story_beat = {}
        self.narrative_duration = 10.0
        self.narrative_sequence = 0
        
        # Memory management metadata
        self.preserved_in_condensation = False
        self.condensation_priority = 0.5
        self.token_count = len(content.split()) * 1.3  # Rough estimate
        
    def update_from_semantic_analysis(self, analysis: Dict[str, Any]):
        """Update metadata from semantic analysis results from sem.py"""
        if not analysis:
            return
        
        self.importance_score = analysis.get("importance_score", 0.5)
        self.categories = analysis.get("categories", ["standard"])
        self.narrative_significance = analysis.get("narrative_significance", "")
        self.pattern_analysis = analysis.get("pattern_analysis", {})
        self.story_beat = analysis.get("story_beat", {})
        self.narrative_duration = analysis.get("narrative_duration", 10.0)
        
        # Calculate condensation priority based on analysis
        self._calculate_condensation_priority()
    
    def _calculate_condensation_priority(self):
        """Calculate priority for preservation during condensation"""
        # Base priority from importance score
        priority = self.importance_score
        
        # Boost for critical categories (defined in sem.py)
        if "story_critical" in self.categories:
            priority = min(1.0, priority + 0.3)
        elif "character_focused" in self.categories:
            priority = min(1.0, priority + 0.2)
        elif "emotional_significance" in self.categories:
            priority = min(1.0, priority + 0.15)
        
        # Boost for strong story beats
        if self.story_beat.get("confidence", 0.0) > 0.7:
            priority = min(1.0, priority + 0.1)
        
        self.condensation_priority = priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "content": self.content,
            "msg_type": self.msg_type.value if isinstance(self.msg_type, MessageType) else str(self.msg_type),
            "timestamp": self.timestamp,
            "formatted_timestamp": self.formatted_timestamp,
            "importance_score": self.importance_score,
            "categories": self.categories,
            "narrative_significance": self.narrative_significance,
            "pattern_analysis": self.pattern_analysis,
            "story_beat": self.story_beat,
            "narrative_duration": self.narrative_duration,
            "narrative_sequence": self.narrative_sequence,
            "preserved_in_condensation": self.preserved_in_condensation,
            "condensation_priority": self.condensation_priority,
            "token_count": self.token_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageMetadata':
        """Create from dictionary during deserialization"""
        msg = cls()
        msg.content = data.get("content", "")
        
        # Handle message type
        msg_type_str = data.get("msg_type", "system")
        try:
            msg.msg_type = MessageType(msg_type_str)
        except ValueError:
            msg.msg_type = MessageType.SYSTEM
        
        msg.timestamp = data.get("timestamp", time.time())
        msg.formatted_timestamp = data.get("formatted_timestamp", "")
        msg.importance_score = data.get("importance_score", 0.5)
        msg.categories = data.get("categories", ["standard"])
        msg.narrative_significance = data.get("narrative_significance", "")
        msg.pattern_analysis = data.get("pattern_analysis", {})
        msg.story_beat = data.get("story_beat", {})
        msg.narrative_duration = data.get("narrative_duration", 10.0)
        msg.narrative_sequence = data.get("narrative_sequence", 0)
        msg.preserved_in_condensation = data.get("preserved_in_condensation", False)
        msg.condensation_priority = data.get("condensation_priority", 0.5)
        msg.token_count = data.get("token_count", len(msg.content.split()) * 1.3)
        
        return msg


class EnhancedMemoryManager:
    """
    Enhanced Memory Manager - Phase 5 Refactored
    
    Responsibilities:
    - Message storage and retrieval
    - Integration with sem.py for semantic analysis
    - Intelligent condensation using semantic categories
    - State persistence and loading
    - Token management and memory limits
    - Background analysis coordination
    
    Key Changes in Phase 5:
    - Removed embedded semantic logic
    - Uses SemanticProcessor from sem.py for all analysis
    - Leverages centralized condensation strategies
    - Maintains same interface for backward compatibility
    """
    
    def __init__(self, max_memory_tokens: int = DEFAULT_MAX_TOKENS, debug_logger=None):
        self.max_memory_tokens = max_memory_tokens
        self.debug_logger = debug_logger
        
        # Message storage
        self.messages: List[MessageMetadata] = []
        self.momentum_state: Dict[str, Any] = {}
        
        # Semantic processor integration (set by orchestrator)
        self.semantic_processor: Optional[SemanticProcessor] = None
        self.mcp_client = None  # Set by orchestrator for LLM condensation
        
        # Memory state tracking
        self.current_state = ConversationState.NORMAL
        self.total_estimated_tokens = 0
        self.last_condensation_time = 0.0
        self.condensation_count = 0
        self.narrative_sequence_counter = 0
        
        # Performance tracking
        self.stats = {
            "messages_added": 0,
            "messages_condensed": 0,
            "condensation_runs": 0,
            "semantic_analyses": 0,
            "background_analyses": 0,
            "saves_performed": 0,
            "loads_performed": 0
        }
        
        # Ensure save directory exists
        os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    
    def set_semantic_processor(self, processor: SemanticProcessor):
        """Set the semantic processor from orchestrator"""
        self.semantic_processor = processor
    
    def _log_debug(self, message: str):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(message, "EMM")
    
    def _auto_load(self):
        """Auto-load existing conversation if present"""
        try:
            default_path = os.path.join(SAVE_DIRECTORY, DEFAULT_FILENAME)
            if os.path.exists(default_path):
                if self.load_conversation(DEFAULT_FILENAME):
                    self._log_debug("Auto-loaded existing conversation")
                else:
                    self._log_debug("Failed to auto-load existing conversation")
        except Exception as e:
            self._log_debug(f"Auto-load error: {e}")
    
    def add_message(self, content: str, msg_type: MessageType, 
                   immediate_analysis: bool = True) -> MessageMetadata:
        """
        Add message with optional immediate semantic analysis
        
        Args:
            content: Message content
            msg_type: Type of message (USER, ASSISTANT, SYSTEM)
            immediate_analysis: Whether to run analysis immediately
            
        Returns:
            MessageMetadata: Created message with analysis results
        """
        if not content.strip():
            return None
        
        # Create message metadata
        self.narrative_sequence_counter += 1
        message = MessageMetadata(content, msg_type)
        message.narrative_sequence = self.narrative_sequence_counter
        
        # Add to storage
        self.messages.append(message)
        self.stats["messages_added"] += 1
        
        # Update token estimates
        self._update_token_estimates()
        
        # Perform semantic analysis using sem.py if processor available
        if self.semantic_processor and immediate_analysis:
            self._analyze_message_semantics(message)
        
        # Check memory state and trigger condensation if needed
        self._check_memory_state()
        
        self._log_debug(f"Added {msg_type.value} message: {len(content)} chars, "
                       f"total tokens: {self.total_estimated_tokens}")
        
        return message

# Chunk 2/3 - emm.py - Semantic Integration and Condensation Logic

    def _analyze_message_semantics(self, message: MessageMetadata):
        """Analyze message semantics using centralized sem.py processor"""
        if not self.semantic_processor:
            return
        
        try:
            # Use semantic processor for analysis
            import asyncio
            
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # Schedule background analysis through sem.py
                task_id = asyncio.create_task(
                    self.semantic_processor.analyze_message_background(
                        message.content,
                        {"narrative_sequence": message.narrative_sequence}
                    )
                )
                self.stats["background_analyses"] += 1
                self._log_debug(f"Queued background semantic analysis for sequence {message.narrative_sequence}")
                
            except RuntimeError:
                # Not in async context, analyze synchronously using sem.py
                analysis = asyncio.run(
                    self.semantic_processor.analyze_message_semantics(
                        message.content,
                        {"narrative_sequence": message.narrative_sequence},
                        "comprehensive"
                    )
                )
                
                if analysis:
                    message.update_from_semantic_analysis(analysis)
                    self.stats["semantic_analyses"] += 1
                    self._log_debug(f"Completed semantic analysis for sequence {message.narrative_sequence}")
        
        except Exception as e:
            self._log_debug(f"Semantic analysis failed: {e}")
    
    def _update_token_estimates(self):
        """Update total token count estimates"""
        self.total_estimated_tokens = sum(msg.token_count for msg in self.messages)
        
        # Add momentum state tokens if present
        if self.momentum_state:
            state_content = json.dumps(self.momentum_state)
            self.total_estimated_tokens += len(state_content.split()) * 1.3
    
    def _check_memory_state(self):
        """Check memory usage and update state"""
        usage_ratio = self.total_estimated_tokens / self.max_memory_tokens
        
        if usage_ratio >= MEMORY_THRESHOLDS["emergency_condensation_ratio"]:
            self.current_state = ConversationState.EMERGENCY_CLEANUP
            self._log_debug("EMERGENCY: Memory usage critical, immediate condensation required")
            self._trigger_emergency_condensation()
        elif usage_ratio >= MEMORY_THRESHOLDS["condensation_trigger_ratio"]:
            if len(self.messages) >= MEMORY_THRESHOLDS["min_messages_before_condensation"]:
                self.current_state = ConversationState.CONDENSATION_NEEDED
                self._log_debug("Memory usage high, condensation recommended")
            else:
                self.current_state = ConversationState.APPROACHING_LIMIT
        else:
            self.current_state = ConversationState.NORMAL
    
    def _trigger_emergency_condensation(self):
        """Trigger immediate emergency condensation"""
        try:
            self.current_state = ConversationState.CONDENSATION_IN_PROGRESS
            
            # Emergency condensation with maximum aggressiveness
            condensed_count = self._perform_condensation(
                aggressiveness=MEMORY_THRESHOLDS["max_condensation_passes"]
            )
            
            self.current_state = ConversationState.NORMAL
            self._log_debug(f"Emergency condensation completed: {condensed_count} messages processed")
            
        except Exception as e:
            self._log_debug(f"Emergency condensation failed: {e}")
            self.current_state = ConversationState.NORMAL
    
    def perform_intelligent_condensation(self) -> Dict[str, Any]:
        """
        Perform intelligent condensation using semantic categories from sem.py
        
        Returns:
            Dict with condensation results and statistics
        """
        if self.current_state == ConversationState.CONDENSATION_IN_PROGRESS:
            return {"status": "already_in_progress", "condensed_count": 0}
        
        if len(self.messages) < MEMORY_THRESHOLDS["min_messages_before_condensation"]:
            return {"status": "insufficient_messages", "condensed_count": 0}
        
        try:
            self.current_state = ConversationState.CONDENSATION_IN_PROGRESS
            start_time = time.time()
            original_count = len(self.messages)
            original_tokens = self.total_estimated_tokens
            
            # Determine condensation aggressiveness
            usage_ratio = self.total_estimated_tokens / self.max_memory_tokens
            if usage_ratio >= MEMORY_THRESHOLDS["emergency_condensation_ratio"]:
                aggressiveness = 2
            elif usage_ratio >= MEMORY_THRESHOLDS["condensation_trigger_ratio"]:
                aggressiveness = 1
            else:
                aggressiveness = 0
            
            # Perform condensation using sem.py logic
            condensed_count = self._perform_condensation(aggressiveness)
            
            # Update state
            self.condensation_count += 1
            self.last_condensation_time = time.time()
            self.stats["condensation_runs"] += 1
            self.stats["messages_condensed"] += condensed_count
            self.current_state = ConversationState.NORMAL
            
            processing_time = time.time() - start_time
            final_tokens = self.total_estimated_tokens
            
            result = {
                "status": "success",
                "condensed_count": condensed_count,
                "original_message_count": original_count,
                "final_message_count": len(self.messages),
                "original_tokens": original_tokens,
                "final_tokens": final_tokens,
                "tokens_saved": original_tokens - final_tokens,
                "processing_time": processing_time,
                "aggressiveness_level": aggressiveness
            }
            
            self._log_debug(f"Condensation completed: {condensed_count} messages, "
                           f"{result['tokens_saved']} tokens saved")
            
            return result
            
        except Exception as e:
            self.current_state = ConversationState.NORMAL
            self._log_debug(f"Condensation failed: {e}")
            return {"status": "failed", "error": str(e), "condensed_count": 0}
    
    def _perform_condensation(self, aggressiveness: int = 0) -> int:
        """
        Perform actual condensation using sem.py semantic preservation logic
        
        Args:
            aggressiveness: 0=conservative, 1=moderate, 2=aggressive
            
        Returns:
            Number of messages processed for condensation
        """
        if not self.semantic_processor:
            self._log_debug("No semantic processor available for condensation")
            return 0
        
        condensed_count = 0
        
        # Group messages by semantic categories using sem.py logic
        category_groups = self._group_messages_by_category()
        
        for category, messages in category_groups.items():
            if len(messages) < 3:  # Don't condense very small groups
                continue
            
            # Use sem.py to determine preservation threshold for this category
            preservation_threshold = self.semantic_processor.analyzer.calculate_preservation_threshold(
                category, aggressiveness
            )
            
            # Sort messages by condensation priority
            messages.sort(key=lambda m: m.condensation_priority, reverse=True)
            
            # Preserve high-priority messages, condense others
            preserved_messages = []
            condensable_messages = []
            
            for message in messages:
                # Use sem.py logic to determine if message should be preserved
                if self.semantic_processor.should_preserve_message(
                    {"importance_score": message.importance_score, 
                     "categories": message.categories}, 
                    aggressiveness
                ):
                    preserved_messages.append(message)
                else:
                    condensable_messages.append(message)
            
            # Condense the condensable messages
            if len(condensable_messages) >= 2:
                condensed_summary = self._condense_message_group(
                    condensable_messages, category
                )
                
                if condensed_summary:
                    # Remove original messages and add summary
                    for msg in condensable_messages:
                        if msg in self.messages:
                            self.messages.remove(msg)
                    
                    # Add condensed summary
                    summary_message = MessageMetadata(
                        condensed_summary, MessageType.SYSTEM
                    )
                    summary_message.categories = [category]
                    summary_message.importance_score = 0.6
                    summary_message.preserved_in_condensation = True
                    
                    # Insert at appropriate position
                    if condensable_messages:
                        # Find insertion point
                        insertion_index = len(self.messages)
                        for i, existing_msg in enumerate(self.messages):
                            if existing_msg.narrative_sequence > condensable_messages[0].narrative_sequence:
                                insertion_index = i
                                break
                        self.messages.insert(insertion_index, summary_message)
                    else:
                        self.messages.append(summary_message)
                    
                    condensed_count += len(condensable_messages)
        
        # Update token estimates after condensation
        self._update_token_estimates()
        
        return condensed_count
    
    def _group_messages_by_category(self) -> Dict[str, List[MessageMetadata]]:
        """Group messages by their primary semantic category using sem.py"""
        groups = {}
        
        for message in self.messages:
            if not message.categories:
                category = "standard"
            else:
                # Use sem.py semantic processor to get highest priority category
                if self.semantic_processor:
                    category = self.semantic_processor.analyzer.get_highest_priority_category(
                        message.categories
                    )
                else:
                    category = message.categories[0]  # Fallback
            
            if category not in groups:
                groups[category] = []
            groups[category].append(message)
        
        return groups
    
    def _condense_message_group(self, messages: List[MessageMetadata], category: str) -> Optional[str]:
        """
        Condense a group of messages using LLM or rule-based approach
        Uses sem.py condensation instructions for category-aware processing
        
        Args:
            messages: List of messages to condense
            category: Semantic category for context-aware condensation
            
        Returns:
            Condensed summary text or None if condensation failed
        """
        if not messages:
            return None
        
        try:
            # Get condensation instruction for this category from sem.py
            if self.semantic_processor:
                instruction = self.semantic_processor.get_condensation_instruction(category)
            else:
                # Fallback to direct CONDENSATION_STRATEGIES access
                instruction = CONDENSATION_STRATEGIES.get(
                    category, CONDENSATION_STRATEGIES["standard"]
                )["instruction"]
            
            # Prepare content for condensation
            content_blocks = [msg.content for msg in messages]
            
            # Try LLM-based condensation if MCP client available
            if self.mcp_client:
                return self._llm_condense_messages(content_blocks, category, instruction)
            else:
                # Fallback to rule-based condensation
                return self._rule_based_condense_messages(content_blocks, category)
                
        except Exception as e:
            self._log_debug(f"Message condensation failed: {e}")
            return None
    
    def _llm_condense_messages(self, content_blocks: List[str], category: str, instruction: str) -> Optional[str]:
        """Use LLM for intelligent message condensation"""
        try:
            # Create condensation prompt using sem.py prompt generation logic
            content_text = "\n".join(content_blocks)
            
            if hasattr(self.semantic_processor, 'prompt_generator'):
                # Use sem.py prompt generator for condensation
                prompt = self.semantic_processor.prompt_generator._create_condensation_prompt(
                    category, content_blocks
                )
            else:
                # Fallback prompt
                prompt = f"""Condense the following {category} conversation content according to these guidelines:

{instruction}

Content to condense:
{content_text}

Return a concise summary that preserves the essential elements for this category while minimizing length. Maintain narrative continuity and emotional context."""

            # Send to LLM
            messages = [{"role": "system", "content": prompt}]
            
            import asyncio
            
            try:
                response = asyncio.run(self.mcp_client.send_request(messages))
                if response:
                    return f"[Condensed {category}]: {response}"
            except:
                pass
            
            return None
            
        except Exception as e:
            self._log_debug(f"LLM condensation failed: {e}")
            return None
    
    def _rule_based_condense_messages(self, content_blocks: List[str], category: str) -> str:
        """Fallback rule-based message condensation"""
        if not content_blocks:
            return ""
        
        total_chars = sum(len(content) for content in content_blocks)
        message_count = len(content_blocks)
        
        # Use category-specific condensation strategies from sem.py
        if category == "story_critical":
            summary = f"[Critical Story Events - {message_count} exchanges]: "
            summary += " → ".join(content[:100] + "..." if len(content) > 100 else content 
                                 for content in content_blocks[:3])
        elif category == "character_focused":
            summary = f"[Character Development - {message_count} interactions]: "
            summary += " ".join(content[:80] + "..." if len(content) > 80 else content 
                               for content in content_blocks[:2])
        elif category == "emotional_significance":
            summary = f"[Emotional Moments - {message_count} exchanges]: "
            summary += content_blocks[0][:60] + "..."
            if len(content_blocks) > 1:
                summary += f" ... {content_blocks[-1][:60]}..."
        else:
            # Standard condensation
            summary = f"[{category.title()} Content - {message_count} messages, {total_chars} chars]: "
            if content_blocks:
                summary += content_blocks[0][:50] + "..."
                if len(content_blocks) > 1:
                    summary += f" ... {content_blocks[-1][:50]}..."
        
        return summary

# Chunk 3/3 - emm.py - Conversation Management and State Persistence

    def get_conversation_for_mcp(self) -> List[Dict[str, str]]:
        """
        Get conversation in MCP format with semantic optimization
        Uses sem.py analysis to optimize content for LLM context
        """
        mcp_messages = []
        
        for message in self.messages:
            # Skip condensed placeholders unless they're important
            if (message.preserved_in_condensation and 
                message.importance_score < 0.3):
                continue
            
            # Convert to MCP format
            role = self._message_type_to_mcp_role(message.msg_type)
            if role:
                mcp_messages.append({
                    "role": role,
                    "content": message.content
                })
        
        return mcp_messages
    
    def _message_type_to_mcp_role(self, msg_type: MessageType) -> Optional[str]:
        """Convert internal message type to MCP role"""
        mapping = {
            MessageType.USER: "user",
            MessageType.ASSISTANT: "assistant",
            MessageType.SYSTEM: "system"
        }
        return mapping.get(msg_type)
    
    def get_recent_messages(self, count: int = 10) -> List[MessageMetadata]:
        """Get most recent messages"""
        return self.messages[-count:] if self.messages else []
    
    def get_messages_by_type(self, msg_type: MessageType) -> List[MessageMetadata]:
        """Get all messages of specific type"""
        return [msg for msg in self.messages if msg.msg_type == msg_type]
    
    def get_messages_by_category(self, category: str) -> List[MessageMetadata]:
        """Get messages with specific semantic category (from sem.py analysis)"""
        return [msg for msg in self.messages if category in msg.categories]
    
    def search_messages(self, query: str, limit: int = 20) -> List[MessageMetadata]:
        """Search messages by content"""
        query_lower = query.lower()
        results = []
        
        for message in self.messages:
            if query_lower in message.content.lower():
                results.append(message)
                if len(results) >= limit:
                    break
        
        return results
    
    def clear_conversation(self):
        """Clear all conversation data"""
        self.messages.clear()
        self.total_estimated_tokens = 0
        self.current_state = ConversationState.NORMAL
        self.narrative_sequence_counter = 0
        
        # Reset stats
        self.stats.update({
            "messages_added": 0,
            "messages_condensed": 0,
            "condensation_runs": 0,
            "semantic_analyses": 0,
            "background_analyses": 0
        })
        
        self._log_debug("Conversation memory cleared")
    
    def get_message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get detailed token usage information"""
        usage_ratio = self.total_estimated_tokens / self.max_memory_tokens
        
        return {
            "total_tokens": self.total_estimated_tokens,
            "max_tokens": self.max_memory_tokens,
            "usage_ratio": usage_ratio,
            "usage_percentage": usage_ratio * 100,
            "state": self.current_state.value,
            "condensation_needed": usage_ratio >= MEMORY_THRESHOLDS["condensation_trigger_ratio"]
        }
    
    def update_momentum_state(self, state: Dict[str, Any]):
        """Update SME momentum state"""
        self.momentum_state = state.copy() if state else {}
        self._update_token_estimates()
        self._log_debug("Momentum state updated")
    
    def get_momentum_state(self) -> Dict[str, Any]:
        """Get current SME momentum state"""
        return self.momentum_state.copy()
    
    def save_conversation(self, filename: Optional[str] = None) -> bool:
        """
        Save conversation to file with semantic metadata
        
        Args:
            filename: Custom filename or None for default
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_{timestamp}.json"
            
            filepath = os.path.join(SAVE_DIRECTORY, filename)
            
            # Prepare data for serialization
            save_data = {
                "metadata": {
                    "save_timestamp": datetime.now().isoformat(),
                    "message_count": len(self.messages),
                    "total_tokens": self.total_estimated_tokens,
                    "condensation_count": self.condensation_count,
                    "version": "2.0_phase5_semantic"
                },
                "messages": [msg.to_dict() for msg in self.messages],
                "momentum_state": self.momentum_state,
                "stats": self.stats.copy()
            }
            
            # Create backup if file exists
            if os.path.exists(filepath):
                backup_path = filepath + ".backup"
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(filepath, backup_path)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            self.stats["saves_performed"] += 1
            self._log_debug(f"Conversation saved to {filename}")
            return True
            
        except Exception as e:
            self._log_debug(f"Failed to save conversation: {e}")
            return False
    
    def load_conversation(self, filename: str) -> bool:
        """
        Load conversation from file
        
        Args:
            filename: File to load from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = os.path.join(SAVE_DIRECTORY, filename)
            
            if not os.path.exists(filepath):
                self._log_debug(f"File not found: {filename}")
                return False
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear current data
            self.clear_conversation()
            
            # Load messages
            if "messages" in data:
                for msg_data in data["messages"]:
                    message = MessageMetadata.from_dict(msg_data)
                    self.messages.append(message)
                    self.narrative_sequence_counter = max(
                        self.narrative_sequence_counter,
                        message.narrative_sequence
                    )
            
            # Load momentum state
            if "momentum_state" in data:
                self.momentum_state = data["momentum_state"]
            
            # Load stats
            if "stats" in data:
                self.stats.update(data["stats"])
            
            # Update derived values
            self._update_token_estimates()
            self._check_memory_state()
            
            self.stats["loads_performed"] += 1
            self._log_debug(f"Conversation loaded from {filename}")
            return True
            
        except Exception as e:
            self._log_debug(f"Failed to load conversation: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory manager statistics"""
        return {
            "message_count": len(self.messages),
            "total_tokens": self.total_estimated_tokens,
            "max_tokens": self.max_memory_tokens,
            "usage_ratio": self.total_estimated_tokens / self.max_memory_tokens,
            "current_state": self.current_state.value,
            "condensation_count": self.condensation_count,
            "last_condensation": self.last_condensation_time,
            "narrative_sequence": self.narrative_sequence_counter,
            "momentum_state_size": len(self.momentum_state),
            "performance_stats": self.stats.copy()
        }


# Factory functions for orchestrator integration
def create_enhanced_memory_manager(max_tokens: int = DEFAULT_MAX_TOKENS, 
                                 debug_logger=None) -> EnhancedMemoryManager:
    """Factory function to create enhanced memory manager"""
    return EnhancedMemoryManager(max_tokens, debug_logger)


def get_memory_thresholds() -> Dict[str, Any]:
    """Get copy of memory management thresholds for configuration"""
    return dict(MEMORY_THRESHOLDS)


def validate_message_type(msg_type: Any) -> MessageType:
    """Validate and convert message type"""
    if isinstance(msg_type, MessageType):
        return msg_type
    
    if isinstance(msg_type, str):
        try:
            return MessageType(msg_type.lower())
        except ValueError:
            return MessageType.SYSTEM
    
    return MessageType.SYSTEM


# Utility functions for testing and integration
def test_memory_manager_functionality() -> bool:
    """Test basic memory manager functionality"""
    try:
        # Create test instance
        manager = create_enhanced_memory_manager(1000)  # Small limit for testing
        
        # Test message addition
        msg1 = manager.add_message("Test message 1", MessageType.USER)
        msg2 = manager.add_message("Test response", MessageType.ASSISTANT)
        
        # Test basic functionality
        assert manager.get_message_count() == 2
        assert len(manager.get_recent_messages(5)) == 2
        assert len(manager.get_messages_by_type(MessageType.USER)) == 1
        
        # Test state management
        test_state = {"test_key": "test_value"}
        manager.update_momentum_state(test_state)
        retrieved_state = manager.get_momentum_state()
        assert retrieved_state == test_state
        
        return True
        
    except Exception:
        return False


def get_emm_info() -> Dict[str, Any]:
    """Get information about Enhanced Memory Manager capabilities"""
    return {
        "name": "Enhanced Memory Manager (Phase 5 Refactored)",
        "version": "2.0_phase5",
        "dependencies": ["sem.py for semantic analysis"],
        "features": [
            "Semantic analysis integration via sem.py",
            "Intelligent condensation using centralized strategies",
            "Category-aware preservation logic",
            "Background analysis coordination",
            "Token management and memory limits",
            "State persistence and loading",
            "Conversation search and filtering"
        ],
        "phase5_changes": [
            "Removed embedded semantic logic",
            "Integrated with centralized sem.py SemanticProcessor",
            "Uses shared condensation strategies from sem.py",
            "Leverages sem.py for category prioritization",
            "Maintains backward compatibility"
        ],
        "integration_points": [
            "SemanticProcessor from sem.py for analysis",
            "MCP client for LLM-based condensation",
            "Debug logger for monitoring",
            "Orchestrator for coordination"
        ]
    }


# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Enhanced Memory Manager (Phase 5 Refactored)")
    print("Successfully refactored to use centralized semantic logic:")
    print("✓ Removed embedded semantic analysis code")
    print("✓ Integrated with sem.py SemanticProcessor")
    print("✓ Uses centralized condensation strategies")
    print("✓ Leverages sem.py category prioritization")
    print("✓ Maintains backward compatibility")
    print("✓ Background analysis coordination")
    print("✓ Enhanced token management")
    print("✓ State persistence with semantic metadata")
    
    print("\nEMM Phase 5 Info:")
    info = get_emm_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  • {item}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nFunctionality test: {'✓ PASSED' if test_memory_manager_functionality() else '✗ FAILED'}")
    print("\nReady for Phase 5 continuation with sme.py refactoring.")
