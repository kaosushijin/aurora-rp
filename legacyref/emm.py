# Chunk 1/1 - emm.py - Enhanced Memory Manager with Complete LLM Semantic Analysis
#!/usr/bin/env python3

import json
import os
import shutil
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import threading
import asyncio
import httpx

# Default memory file configuration
DEFAULT_MEMORY_FILE = "memory.json"

class MessageType(Enum):
    """Message type enumeration for conversation tracking"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    MOMENTUM_STATE = "momentum_state"  # Special type for SME state storage

class Message:
    """Individual conversation message with metadata"""
    
    def __init__(self, content: str, message_type: MessageType, timestamp: Optional[str] = None):
        self.content = content
        self.message_type = message_type
        self.timestamp = timestamp or datetime.now().isoformat()
        self.token_estimate = self._estimate_tokens(content)
        self.id = str(uuid4())
        self.content_category = "standard"  # Default category for semantic analysis
        self.condensed = False
    
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
            "tokens": self.token_estimate,
            "content_category": self.content_category,
            "condensed": self.condensed
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
        msg.content_category = data.get("content_category", "standard")
        msg.condensed = data.get("condensed", False)
        return msg

# Semantic category preservation ratios
CONDENSATION_STRATEGIES = {
    "story_critical": {
        "threshold": 100,
        "preservation_ratio": 0.8,
        "instruction": (
            "Preserve all major plot developments, character deaths, world-changing events, "
            "key player decisions, and their consequences. Use decisive language highlighting "
            "the significance of events. Compress dialogue while maintaining essential meaning."
        )
    },
    "character_focused": {
        "threshold": 80,
        "preservation_ratio": 0.7,
        "instruction": (
            "Preserve relationship changes, trust/betrayal moments, character motivations, "
            "personality reveals, Aurora's development, and NPC traits. Emphasize emotional "
            "weight and relationship dynamics. Condense descriptions while keeping character essence."
        )
    },
    "relationship_dynamics": {
        "threshold": 80,
        "preservation_ratio": 0.8,
        "instruction": (
            "Preserve evolving relationships between characters, trust building, conflicts, "
            "alliances, and interpersonal dynamics. Maintain emotional context and progression."
        )
    },
    "emotional_significance": {
        "threshold": 70,
        "preservation_ratio": 0.75,
        "instruction": (
            "Preserve dramatic moments, emotional peaks, character growth, conflict resolution, "
            "and significant emotional revelations. Maintain the emotional weight of scenes."
        )
    },
    "world_building": {
        "threshold": 60,
        "preservation_ratio": 0.6,
        "instruction": (
            "Preserve new locations, lore revelations, cultural information, political changes, "
            "economic systems, magical discoveries, and historical context. Provide rich "
            "foundational details. Compress atmospheric descriptions while keeping key world facts."
        )
    },
    "standard": {
        "threshold": 40,
        "preservation_ratio": 0.4,
        "instruction": (
            "Preserve player actions and immediate consequences for continuity. Compress "
            "everything else aggressively while maintaining basic story flow."
        )
    }
}

class EnhancedMemoryManager:
    """Memory management with LLM-powered semantic condensation and auto-persistence"""
    
    def __init__(self, max_memory_tokens: int = 16000, debug_logger=None, 
                 auto_save_enabled: bool = True, memory_file: str = DEFAULT_MEMORY_FILE):
        self.max_memory_tokens = max_memory_tokens
        self.debug_logger = debug_logger
        self.auto_save_enabled = auto_save_enabled
        self.memory_file = memory_file
        self.messages: List[Message] = []
        self.condensation_count = 0
        self.lock = threading.Lock()
        
        # MCP client configuration
        self.mcp_config = self._load_mcp_config()
        
        # Auto-load existing memory on initialization
        if self.auto_save_enabled:
            self._auto_load()
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration for LLM calls"""
        return {
            "server_url": "http://127.0.0.1:3456/chat",
            "model": "qwen2.5:14b-instruct-q4_k_m",
            "timeout": 300
        }
    
    async def _call_llm(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Make LLM request for semantic analysis using working MCP format"""
        try:
            async with httpx.AsyncClient(timeout=self.mcp_config.get("timeout", 30)) as client:
                # Use same payload format as working mcp.py
                payload = {
                    "model": self.mcp_config["model"],
                    "messages": messages,
                    "stream": False
                }

                response = await client.post(self.mcp_config["server_url"], json=payload)
                response.raise_for_status()

                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")

        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"EMM LLM call failed: {e}")
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
            parsed = self._parse_semantic_response_robust(result, attempt=1)
            if parsed:
                return parsed
        
        # Attempt 2: Simplified analysis
        prompt2 = self._create_simple_analysis_prompt(target_message.content)
        result = await self._call_llm([{"role": "system", "content": prompt2}])
        
        if result:
            parsed = self._parse_semantic_response_robust(result, attempt=2)
            if parsed:
                return parsed
        
        # Attempt 3: Binary preserve/condense decision
        prompt3 = self._create_binary_prompt(target_message.content)
        result = await self._call_llm([{"role": "system", "content": prompt3}])
        
        if result:
            parsed = self._parse_semantic_response_robust(result, attempt=3)
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
    
    def _parse_semantic_response_robust(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Parse LLM semantic analysis response with 5-strategy defensive handling"""
        
        # Strategy 1: Direct JSON parsing
        try:
            data = json.loads(response.strip())
            if self._validate_semantic_data(data, attempt):
                return self._inject_missing_fields(data, attempt)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Substring extraction
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                if self._validate_semantic_data(data, attempt):
                    return self._inject_missing_fields(data, attempt)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Strategy 3: Field validation and extraction
        try:
            # Look for specific patterns in response
            if attempt == 3:  # Binary response
                if "true" in response.lower() or "preserve" in response.lower():
                    return {"importance_score": 0.8, "categories": ["story_critical"], "fragments": None}
                else:
                    return {"importance_score": 0.2, "categories": ["standard"], "fragments": None}
            
            # Try to extract numeric importance
            import re
            importance_match = re.search(r'"?importance_score"?\s*:\s*([0-9.]+)', response)
            if importance_match:
                importance = float(importance_match.group(1))
                return {"importance_score": importance, "categories": ["standard"], "fragments": None}
        except:
            pass
        
        # Strategy 4: Default injection based on attempt type
        if attempt == 3:  # Binary response
            return {"importance_score": 0.4, "categories": ["standard"], "fragments": None}
        elif attempt == 2:  # Simple response
            return {"importance_score": 0.4, "categories": ["standard"], "fragments": None}
        
        # Strategy 5: Complete fallback
        return {"importance_score": 0.4, "categories": ["standard"], "fragments": None}
    
    def _validate_semantic_data(self, data: Dict[str, Any], attempt: int) -> bool:
        """Validate that semantic analysis data has required fields"""
        if not isinstance(data, dict):
            return False
        
        if attempt == 3:  # Binary response
            return "preserve" in data
        
        required_fields = ["importance_score", "categories"]
        return all(field in data for field in required_fields)
    
    def _inject_missing_fields(self, data: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Inject missing fields with sensible defaults"""
        if attempt == 3:  # Binary response
            preserve = data.get("preserve", False)
            return {
                "importance_score": 0.8 if preserve else 0.2,
                "categories": ["story_critical"] if preserve else ["standard"],
                "fragments": None
            }
        
        # Ensure importance_score is valid
        importance = data.get("importance_score", 0.4)
        if not isinstance(importance, (int, float)) or importance < 0 or importance > 1:
            importance = 0.4
        data["importance_score"] = importance
        
        # Ensure categories is a list
        categories = data.get("categories", ["standard"])
        if not isinstance(categories, list):
            categories = ["standard"]
        data["categories"] = categories
        
        # Ensure fragments field exists
        if "fragments" not in data:
            data["fragments"] = None
        
        return data
    
    def _auto_load(self) -> None:
        """Auto-load memory from file if it exists"""
        try:
            if os.path.exists(self.memory_file):
                success = self.load_conversation(self.memory_file)
                if success and self.debug_logger:
                    self.debug_logger.debug(f"Auto-loaded {len(self.messages)} messages from {self.memory_file}")
                elif not success and self.debug_logger:
                    self.debug_logger.error(f"Failed to auto-load from {self.memory_file}")
            elif self.debug_logger:
                self.debug_logger.debug(f"No existing memory file found at {self.memory_file}")
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Auto-load error: {e}")
    
    def _auto_save(self) -> None:
        """Auto-save memory to file"""
        if not self.auto_save_enabled:
            return
            
        try:
            # Create backup of existing file
            if os.path.exists(self.memory_file):
                backup_file = f"{self.memory_file}.bak"
                shutil.copy2(self.memory_file, backup_file)
            
            # Save current state
            success = self.save_conversation(self.memory_file)
            if not success and self.debug_logger:
                self.debug_logger.error(f"Auto-save failed to {self.memory_file}")
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Auto-save error: {e}")
    
    def add_message(self, content: str, message_type: MessageType) -> None:
        """Add new message and manage memory with background auto-save"""
        with self.lock:
            message = Message(content, message_type)
            self.messages.append(message)

            if self.debug_logger:
                self.debug_logger.debug(f"Added {message_type.value} message: {len(content)} chars, {message.token_estimate} tokens")

        # Move ALL auto-save operations to background thread to avoid blocking main thread
        if self.auto_save_enabled:
            auto_save_thread = threading.Thread(
                target=self._background_auto_save,
                daemon=True,
                name="EMM-AutoSave"
            )
            auto_save_thread.start()

    def _background_auto_save(self) -> None:
        """Handle auto-save and condensation in background thread"""
        try:
            # Auto-save first (file operations)
            self._auto_save()

            # Check if condensation needed
            with self.lock:
                current_tokens = sum(msg.token_estimate for msg in self.messages)

            if current_tokens > self.max_memory_tokens:
                if self.debug_logger:
                    self.debug_logger.debug(f"Starting background condensation: {current_tokens} > {self.max_memory_tokens}")
                self._perform_semantic_condensation()

        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Background auto-save failed: {e}")
    
    def _perform_semantic_condensation(self) -> None:
        """Execute multi-pass LLM-powered semantic condensation with auto-save"""
        if len(self.messages) < 10:  # Need minimum messages for context
            return
            
        # Run async condensation in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_multi_pass_condensation())
        finally:
            loop.close()
    
    async def _async_multi_pass_condensation(self) -> None:
        """Multi-pass async condensation with increasing aggressiveness and auto-save"""
        max_passes = 3
        
        for pass_num in range(max_passes):
            current_tokens = sum(msg.token_estimate for msg in self.messages)
            
            if current_tokens <= self.max_memory_tokens:
                if self.debug_logger:
                    self.debug_logger.debug(f"Condensation complete after {pass_num} passes: {current_tokens} tokens")
                break
                
            if self.debug_logger:
                self.debug_logger.debug(f"Condensation pass {pass_num + 1}/{max_passes}, tokens: {current_tokens}")
            
            # Ensure all messages have semantic categories
            await self._categorize_uncategorized_messages()
            
            # Collect preservation candidates with increasing aggressiveness
            preserve_messages, condense_candidates = await self._collect_preservation_candidates(pass_num)
            
            if not condense_candidates:
                if self.debug_logger:
                    self.debug_logger.debug(f"No condensation candidates found at aggressiveness level {pass_num}")
                continue
            
            # Create condensed summary for candidates by category
            condensed_message = await self._create_category_aware_summary(condense_candidates)
            
            if condensed_message:
                # Replace candidates with condensed summary
                with self.lock:
                    self.messages = [condensed_message] + preserve_messages
                    self.condensation_count += 1
                
                # Auto-save after successful condensation
                self._auto_save()
                
                new_tokens = sum(msg.token_estimate for msg in self.messages)
                if self.debug_logger:
                    self.debug_logger.debug(
                        f"Pass {pass_num + 1} complete: condensed {len(condense_candidates)} messages, "
                        f"tokens: {current_tokens} → {new_tokens}"
                    )
            else:
                if self.debug_logger:
                    self.debug_logger.debug(f"Condensation failed at pass {pass_num + 1}")
                break
        
        # Final status
        final_tokens = sum(msg.token_estimate for msg in self.messages)
        if final_tokens > self.max_memory_tokens:
            if self.debug_logger:
                self.debug_logger.debug(f"Condensation incomplete: {final_tokens} tokens still exceed limit")
    
    async def _categorize_uncategorized_messages(self) -> None:
        """Ensure all messages have semantic categories"""
        uncategorized = [
            (i, msg) for i, msg in enumerate(self.messages)
            if msg.message_type in [MessageType.USER, MessageType.ASSISTANT] 
            and msg.content_category == "standard"
            and not msg.condensed
        ]
        
        if self.debug_logger and uncategorized:
            self.debug_logger.debug(f"Categorizing {len(uncategorized)} uncategorized messages")
        
        # Process in batches to avoid overwhelming LLM
        batch_size = 10
        for batch_start in range(0, len(uncategorized), batch_size):
            batch = uncategorized[batch_start:batch_start + batch_size]
            
            for msg_idx, message in batch:
                analysis = await self._analyze_message_semantics(msg_idx)
                if analysis:
                    categories = analysis.get("categories", ["standard"])
                    # Use highest-priority category
                    message.content_category = self._get_highest_priority_category(categories)
    
    def _get_highest_priority_category(self, categories: List[str]) -> str:
        """Get highest priority category from list"""
        priority_order = [
            "story_critical", "character_focused", "relationship_dynamics", 
            "emotional_significance", "world_building", "standard"
        ]
        
        for category in priority_order:
            if category in categories:
                return category
        return "standard"
    
    async def _collect_preservation_candidates(self, aggressiveness_level: int) -> Tuple[List[Message], List[Message]]:
        """Collect messages for preservation vs condensation with increasing aggressiveness"""
        preserve_messages = []
        condense_candidates = []
        
        # Always preserve recent messages (last 5)
        recent_cutoff = max(0, len(self.messages) - 5)
        
        for i, message in enumerate(self.messages[:recent_cutoff]):
            # Skip already condensed messages
            if message.condensed:
                preserve_messages.append(message)
                continue
            
            # Skip non-conversation messages
            if message.message_type not in [MessageType.USER, MessageType.ASSISTANT]:
                preserve_messages.append(message)
                continue
            
            # Analyze message for preservation decision
            analysis = await self._analyze_message_semantics(i)
            should_preserve = self._should_preserve_with_aggressiveness(analysis, aggressiveness_level)
            
            if should_preserve:
                preserve_messages.append(message)
            else:
                condense_candidates.append(message)
        
        # Always preserve recent messages
        preserve_messages.extend(self.messages[recent_cutoff:])
        
        if self.debug_logger:
            self.debug_logger.debug(
                f"Aggressiveness {aggressiveness_level}: preserve {len(preserve_messages)}, "
                f"condense {len(condense_candidates)}"
            )
        
        return preserve_messages, condense_candidates
    
    def _should_preserve_with_aggressiveness(self, analysis: Dict[str, Any], aggressiveness: int) -> bool:
        """Determine preservation with increasing aggressiveness"""
        categories = analysis.get("categories", ["standard"])
        importance = analysis.get("importance_score", 0.4)
        
        # Get base preservation ratio for highest priority category
        highest_category = self._get_highest_priority_category(categories)
        base_ratio = CONDENSATION_STRATEGIES.get(highest_category, CONDENSATION_STRATEGIES["standard"])["preservation_ratio"]
        
        # Apply aggressiveness reduction
        # Pass 0: base ratio, Pass 1: -0.15, Pass 2: -0.3
        aggressiveness_reduction = aggressiveness * 0.15
        adjusted_ratio = max(0.1, base_ratio - aggressiveness_reduction)
        
        # Preserve if importance exceeds adjusted threshold
        preserve = importance >= adjusted_ratio
        
        return preserve
    
    async def _create_category_aware_summary(self, messages: List[Message]) -> Optional[Message]:
        """Create condensed summary organized by semantic categories"""
        if not messages:
            return None
        
        # Group messages by category
        category_groups = {}
        for msg in messages:
            category = msg.content_category
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(msg)
        
        # Create summaries for each category
        category_summaries = []
        
        for category, category_messages in category_groups.items():
            strategy = CONDENSATION_STRATEGIES.get(category, CONDENSATION_STRATEGIES["standard"])
            instruction = strategy["instruction"]
            
            content_to_condense = "\n".join([
                f"[{msg.message_type.value}] {msg.content}"
                for msg in category_messages
            ])
            
            prompt = f"""Condense the following {category} conversation content according to these guidelines:

{instruction}

Content to condense:
{content_to_condense}

Return a concise summary that preserves the essential elements for this category while minimizing length. Maintain narrative continuity and emotional context."""
            
            summary_content = await self._call_llm([{"role": "system", "content": prompt}])
            
            if summary_content:
                category_summaries.append(f"[{category.upper()}] {summary_content}")
        
        if category_summaries:
            final_summary = "\n\n".join(category_summaries)
            condensed_msg = Message(
                content=f"[CONDENSED - {len(messages)} messages] {final_summary}",
                message_type=MessageType.SYSTEM
            )
            condensed_msg.condensed = True
            condensed_msg.content_category = "condensed_summary"
            return condensed_msg
        
        return None
    
    def clear_memory_file(self) -> bool:
        """Clear memory file and reset in-memory state"""
        try:
            with self.lock:
                # Clear in-memory state
                self.messages.clear()
                self.condensation_count = 0
                
                # Remove memory file if it exists
                if os.path.exists(self.memory_file):
                    os.remove(self.memory_file)
                
                # Remove backup file if it exists
                backup_file = f"{self.memory_file}.bak"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
            
            if self.debug_logger:
                self.debug_logger.debug(f"Memory file {self.memory_file} cleared")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to clear memory file: {e}")
            return False
    
    def get_memory_file_info(self) -> Dict[str, Any]:
        """Get memory file information for status reporting"""
        try:
            if not os.path.exists(self.memory_file):
                return {
                    "file_exists": False,
                    "file_path": self.memory_file,
                    "auto_save_enabled": self.auto_save_enabled
                }
            
            stat = os.stat(self.memory_file)
            
            with self.lock:
                return {
                    "file_exists": True,
                    "file_path": self.memory_file,
                    "file_size": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "message_count": len(self.messages),
                    "auto_save_enabled": self.auto_save_enabled,
                    "backup_exists": os.path.exists(f"{self.memory_file}.bak")
                }
                
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to get memory file info: {e}")
            return {
                "file_exists": False,
                "error": str(e),
                "auto_save_enabled": self.auto_save_enabled
            }
    
    def backup_memory_file(self, backup_filename: Optional[str] = None) -> bool:
        """Create backup of memory file"""
        try:
            if not os.path.exists(self.memory_file):
                return False
            
            if not backup_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"memory_backup_{timestamp}.json"
            
            shutil.copy2(self.memory_file, backup_filename)
            
            if self.debug_logger:
                self.debug_logger.debug(f"Memory backed up to {backup_filename}")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to backup memory file: {e}")
            return False

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Retrieve messages with optional limit"""
        with self.lock:
            if limit:
                return self.messages[-limit:]
            return self.messages.copy()
    
    def get_conversation_for_mcp(self) -> List[Dict[str, str]]:
        """Format conversation for MCP requests, excluding momentum state"""
        with self.lock:
            return [
                {"role": msg.message_type.value, "content": msg.content}
                for msg in self.messages
                if msg.message_type != MessageType.MOMENTUM_STATE
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

    def save_conversation(self, filename: Optional[str] = None) -> bool:
        """Save conversation to file with robust error handling"""
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
                        "condensations": self.condensation_count,
                        "auto_save_enabled": self.auto_save_enabled
                    },
                    "messages": [msg.to_dict() for msg in self.messages]
                }
            
            # Create backup if file already exists
            if os.path.exists(filename):
                backup_filename = f"{filename}.bak"
                shutil.copy2(filename, backup_filename)
            
            # Write to temporary file first, then move to prevent corruption
            temp_filename = f"{filename}.tmp"
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            shutil.move(temp_filename, filename)
            
            if self.debug_logger:
                self.debug_logger.debug(f"Conversation saved to {filename}")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to save conversation: {e}")
            
            # Clean up temporary file if it exists
            temp_filename = f"{filename}.tmp"
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass
            
            return False
    
    def load_conversation(self, filename: str) -> bool:
        """Load conversation from file with corruption recovery"""
        try:
            if not os.path.exists(filename):
                if self.debug_logger:
                    self.debug_logger.debug(f"File does not exist: {filename}")
                return False
            
            # Try loading main file
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                if self.debug_logger:
                    self.debug_logger.error(f"Main file corrupted, trying backup: {e}")
                
                # Try backup file
                backup_filename = f"{filename}.bak"
                if os.path.exists(backup_filename):
                    with open(backup_filename, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if self.debug_logger:
                        self.debug_logger.debug("Recovered from backup file")
                else:
                    raise e
            
            with self.lock:
                # Load messages
                self.messages = [
                    Message.from_dict(msg_data) 
                    for msg_data in data.get("messages", [])
                ]
                
                # Load metadata
                metadata = data.get("metadata", {})
                self.condensation_count = metadata.get("condensations", 0)
            
            if self.debug_logger:
                self.debug_logger.debug(f"Conversation loaded from {filename}: {len(self.messages)} messages")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to load conversation from {filename}: {e}")
            return False
    
    def clear_memory(self) -> None:
        """Clear all stored messages (in-memory only)"""
        with self.lock:
            self.messages.clear()
            self.condensation_count = 0
            
        if self.debug_logger:
            self.debug_logger.debug("In-memory state cleared")
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Generate conversation statistics for debugging"""
        with self.lock:
            if not self.messages:
                return {"status": "no_messages"}
            
            message_types = {}
            category_counts = {}
            
            for msg in self.messages:
                msg_type = msg.message_type.value
                message_types[msg_type] = message_types.get(msg_type, 0) + 1
                
                category = getattr(msg, 'content_category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            total_tokens = sum(msg.token_estimate for msg in self.messages)
            avg_tokens = total_tokens / len(self.messages) if self.messages else 0
            
            condensed_count = sum(1 for msg in self.messages if getattr(msg, 'condensed', False))
            
            return {
                "total_messages": len(self.messages),
                "message_types": message_types,
                "semantic_categories": category_counts,
                "condensed_messages": condensed_count,
                "total_tokens": total_tokens,
                "average_tokens_per_message": round(avg_tokens, 2),
                "memory_utilization": round(total_tokens / self.max_memory_tokens, 3),
                "condensations_performed": self.condensation_count,
                "oldest_message": self.messages[0].timestamp if self.messages else None,
                "newest_message": self.messages[-1].timestamp if self.messages else None,
                "auto_save_enabled": self.auto_save_enabled,
                "memory_file": self.memory_file
            }

    # SME Integration Methods
    def get_momentum_state(self) -> Optional[Dict[str, Any]]:
        """Retrieve current momentum state from memory for SME"""
        with self.lock:
            for message in reversed(self.messages):
                if message.message_type == MessageType.MOMENTUM_STATE:
                    try:
                        if isinstance(message.content, str):
                            return json.loads(message.content)
                        else:
                            return message.content
                    except json.JSONDecodeError:
                        continue
            return None
    
    def update_momentum_state(self, state_data: Dict[str, Any]) -> None:
        """Update or create momentum state in memory for SME"""
        with self.lock:
            # Remove existing momentum state
            self.messages = [msg for msg in self.messages if msg.message_type != MessageType.MOMENTUM_STATE]
            
            # Add new momentum state
            momentum_msg = Message(
                content=json.dumps(state_data),
                message_type=MessageType.MOMENTUM_STATE
            )
            self.messages.append(momentum_msg)
        
        # Trigger auto-save for momentum state
        if self.auto_save_enabled:
            self._auto_save()

# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Enhanced Memory Manager with Complete LLM Semantic Analysis")
    print("Testing memory management functionality...")
    
    # Test with auto-save enabled
    emm = EnhancedMemoryManager(auto_save_enabled=True, memory_file="test_memory.json")
    
    # Test basic functionality
    emm.add_message("Hello there!", MessageType.USER)
    emm.add_message("Greetings, traveler!", MessageType.ASSISTANT)
    
    stats = emm.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    file_info = emm.get_memory_file_info()
    print(f"Memory file info: {file_info}")
    
    patterns = emm.analyze_conversation_patterns()
    print(f"Conversation patterns: {patterns}")
    
    # Test SME integration
    test_state = {
        "narrative_pressure": 0.3,
        "pressure_source": "antagonist",
        "manifestation_type": "tension",
        "escalation_count": 1,
        "base_pressure_floor": 0.0,
        "last_analysis_count": 5,
        "antagonist": {"name": "Test Villain", "motivation": "test purposes"}
    }
    
    emm.update_momentum_state(test_state)
    retrieved_state = emm.get_momentum_state()
    print(f"SME state integration: {retrieved_state}")
    
    # Test memory clearing
    emm.clear_memory_file()
    
    print("Memory manager test completed successfully.")
