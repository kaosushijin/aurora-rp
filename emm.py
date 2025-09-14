# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
DevName RPG Client - Enhanced Memory Manager (emm.py)
Remodularized for hub-and-spoke architecture - semantic analysis moved to sem.py
"""

import json
import threading
import time
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Check for httpx availability (for any remaining LLM calls that need to be moved to orchestrator)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

import os
import shutil
from enum import Enum
from uuid import uuid4

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
        self.content_category = "standard"  # Set by sem.py categorization
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
        """Create message from dictionary data"""
        message = cls(
            content=data["content"],
            message_type=MessageType(data["type"]),
            timestamp=data.get("timestamp")
        )
        message.id = data.get("id", str(uuid4()))
        message.token_estimate = data.get("tokens", message.token_estimate)
        message.content_category = data.get("content_category", "standard")
        message.condensed = data.get("condensed", False)
        return message

class EnhancedMemoryManager:
    """
    SIMPLIFIED: Storage and state management without semantic analysis
    All semantic logic moved to sem.py
    All LLM requests coordinated through orch.py hub
    """
    
    def __init__(self, memory_file: str = DEFAULT_MEMORY_FILE, auto_save_enabled: bool = True, debug_logger=None):
        self.memory_file = memory_file
        self.auto_save_enabled = auto_save_enabled
        self.debug_logger = debug_logger
        
        # Core storage
        self.messages: List[Message] = []
        self.lock = threading.RLock()
        
        # Memory management settings
        self.max_memory_tokens = 25000  # Conservative limit for context window
        self.condensation_count = 0
        
        # Orchestrator communication
        self.orchestrator_callback = None  # Set by orchestrator for condensation requests
        
        # Initialize from existing file
        self._auto_load()
    
    def set_orchestrator_callback(self, callback):
        """Set callback function to orchestrator for semantic operations"""
        self.orchestrator_callback = callback
    
    # =============================================================================
    # CORE STORAGE OPERATIONS
    # =============================================================================
    
    def add_message(self, content: str, message_type: MessageType) -> None:
        """Add new message and manage memory with background auto-save"""
        with self.lock:
            message = Message(content, message_type)
            self.messages.append(message)

            if self.debug_logger:
                self.debug_logger.debug(f"Added {message_type.value} message: {len(content)} chars, {message.token_estimate} tokens")

        # Background auto-save to avoid blocking main thread
        if self.auto_save_enabled:
            auto_save_thread = threading.Thread(
                target=self._background_auto_save,
                daemon=True,
                name="EMM-AutoSave"
            )
            auto_save_thread.start()

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages with optional limit"""
        with self.lock:
            if limit is None:
                return self.messages.copy()
            else:
                return self.messages[-limit:].copy()
    
    def get_conversation_for_mcp(self) -> List[Dict[str, str]]:
        """Get conversation in MCP format (excluding momentum state messages)"""
        with self.lock:
            return [
                {"role": msg.message_type.value, "content": msg.content}
                for msg in self.messages
                if msg.message_type != MessageType.MOMENTUM_STATE
            ]
    
    def update_message_category(self, message_id: str, category: str) -> bool:
        """Update message category (called by orchestrator after sem.py analysis)"""
        with self.lock:
            for message in self.messages:
                if message.id == message_id:
                    message.content_category = category
                    if self.debug_logger:
                        self.debug_logger.debug(f"Updated message {message_id} category to {category}")
                    return True
        return False
    
    def replace_messages_with_condensed(self, message_ids: List[str], condensed_content: str) -> bool:
        """Replace specified messages with condensed version (called by orchestrator)"""
        with self.lock:
            # Find messages to replace
            messages_to_remove = []
            for i, message in enumerate(self.messages):
                if message.id in message_ids:
                    messages_to_remove.append(i)
            
            if not messages_to_remove:
                return False
            
            # Create condensed message
            condensed_message = Message(condensed_content, MessageType.SYSTEM)
            condensed_message.condensed = True
            condensed_message.content_category = "condensed_summary"
            
            # Replace messages (keep earliest position)
            earliest_index = min(messages_to_remove)
            
            # Remove messages in reverse order to maintain indices
            for i in reversed(messages_to_remove):
                del self.messages[i]
            
            # Insert condensed message at earliest position
            self.messages.insert(earliest_index, condensed_message)
            
            self.condensation_count += 1
            
            if self.debug_logger:
                self.debug_logger.debug(f"Replaced {len(message_ids)} messages with condensed summary")
            
            # Auto-save after condensation
            self._auto_save()
            
            return True

# Chunk 2/3 - emm.py - File Operations and State Management

    # =============================================================================
    # FILE OPERATIONS 
    # =============================================================================
    
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
            try:
                with open(temp_filename, "w", encoding="utf-8") as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)

                # Atomic move
                shutil.move(temp_filename, filename)
            except Exception as temp_error:
                # Fallback: direct write if atomic operation fails
                if self.debug_logger:
                    self.debug_logger.debug(f"Atomic save failed, using direct write: {temp_error}")

                # Clean up failed temp file
                try:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                except:
                    pass

                # Direct write as fallback
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
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
                self.debug_logger.debug(f"Loaded {len(self.messages)} messages from {filename}")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Failed to load conversation: {e}")
            return False
    
    def backup_memory_file(self, backup_filename: Optional[str] = None) -> bool:
        """Create backup of current memory file"""
        try:
            if not os.path.exists(self.memory_file):
                return False
            
            if not backup_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"{self.memory_file}.backup_{timestamp}"
            
            shutil.copy2(self.memory_file, backup_filename)
            
            if self.debug_logger:
                self.debug_logger.debug(f"Created backup: {backup_filename}")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Backup failed: {e}")
            return False
    
    def get_memory_file_info(self) -> Dict[str, Any]:
        """Get information about memory file"""
        try:
            if not os.path.exists(self.memory_file):
                return {"exists": False}
            
            stat = os.stat(self.memory_file)
            return {
                "exists": True,
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "filename": self.memory_file
            }
            
        except Exception:
            return {"exists": False, "error": True}
    
    # =============================================================================
    # STATE MANAGEMENT (SME INTEGRATION)
    # =============================================================================
    
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
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Return current memory statistics"""
        with self.lock:
            total_tokens = sum(msg.token_estimate for msg in self.messages)
            return {
                "message_count": len(self.messages),
                "total_tokens": total_tokens,
                "max_tokens": self.max_memory_tokens,
                "utilization": round(total_tokens / self.max_memory_tokens, 3),
                "condensations_performed": self.condensation_count,
                "oldest_message": self.messages[0].timestamp if self.messages else None,
                "newest_message": self.messages[-1].timestamp if self.messages else None,
                "auto_save_enabled": self.auto_save_enabled,
                "memory_file": self.memory_file
            }
    
    def check_condensation_needed(self) -> bool:
        """Check if condensation is needed (called by orchestrator)"""
        with self.lock:
            current_tokens = sum(msg.token_estimate for msg in self.messages)
            return current_tokens > self.max_memory_tokens
    
    def get_condensation_candidates(self, preserve_recent: int = 5) -> List[str]:
        """Get message IDs that are candidates for condensation"""
        with self.lock:
            if len(self.messages) <= preserve_recent:
                return []
            
            # Return message IDs excluding recent messages and momentum state
            candidates = []
            messages_to_check = self.messages[:-preserve_recent]
            
            for message in messages_to_check:
                if (message.message_type != MessageType.MOMENTUM_STATE and 
                    not message.condensed and
                    message.content_category in ["standard", "world_building"]):  # Lower priority categories
                    candidates.append(message.id)
            
            return candidates
    
    # =============================================================================
    # PRIVATE HELPER FUNCTIONS
    # =============================================================================
    
    def _auto_load(self) -> None:
        """Auto-load memory from file if it exists"""
        try:
            # Ensure current directory is writable
            current_dir = Path('.').resolve()
            if not os.access(current_dir, os.W_OK):
                if self.debug_logger:
                    self.debug_logger.error(f"Directory not writable: {current_dir}")
                return

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
    
    def _background_auto_save(self) -> None:
        """Handle auto-save in background thread and check for condensation"""
        try:
            # Auto-save first (file operations)
            self._auto_save()

            # Check if condensation needed - request through orchestrator
            if self.check_condensation_needed() and self.orchestrator_callback:
                if self.debug_logger:
                    current_tokens = sum(msg.token_estimate for msg in self.messages)
                    self.debug_logger.debug(f"Requesting condensation from orchestrator: {current_tokens} > {self.max_memory_tokens}")
                
                try:
                    # Request condensation through orchestrator
                    self.orchestrator_callback({
                        "request_type": "condensation",
                        "memory_stats": self.get_memory_stats(),
                        "candidates": self.get_condensation_candidates()
                    })
                except Exception as e:
                    if self.debug_logger:
                        self.debug_logger.error(f"Orchestrator condensation request failed: {e}")

        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Background auto-save failed: {e}")

# Chunk 3/3 - emm.py - Testing and Module Interface

    # =============================================================================
    # UTILITY AND DIAGNOSTIC FUNCTIONS
    # =============================================================================
    
    def clear_memory(self) -> None:
        """Clear all messages from memory"""
        with self.lock:
            self.messages.clear()
            self.condensation_count = 0
        
        if self.debug_logger:
            self.debug_logger.debug("Memory cleared")
    
    def get_messages_by_category(self, category: str) -> List[Message]:
        """Get messages filtered by content category"""
        with self.lock:
            return [msg for msg in self.messages if msg.content_category == category]
    
    def get_messages_by_type(self, message_type: MessageType) -> List[Message]:
        """Get messages filtered by message type"""
        with self.lock:
            return [msg for msg in self.messages if msg.message_type == message_type]
    
    def get_recent_conversation_summary(self, limit: int = 10) -> str:
        """Get summary of recent conversation for debugging"""
        with self.lock:
            recent_messages = self.messages[-limit:] if len(self.messages) > limit else self.messages
            
            summary_lines = []
            for msg in recent_messages:
                content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                summary_lines.append(f"[{msg.message_type.value}] {content_preview} ({msg.token_estimate} tokens)")
            
            return "\n".join(summary_lines)
    
    def validate_message_integrity(self) -> Dict[str, Any]:
        """Validate integrity of stored messages"""
        with self.lock:
            validation_results = {
                "total_messages": len(self.messages),
                "corrupted_messages": 0,
                "missing_ids": 0,
                "duplicate_ids": 0,
                "token_sum_mismatch": False,
                "issues": []
            }
            
            seen_ids = set()
            calculated_tokens = 0
            
            for i, msg in enumerate(self.messages):
                # Check for missing ID
                if not msg.id:
                    validation_results["missing_ids"] += 1
                    validation_results["issues"].append(f"Message {i} missing ID")
                
                # Check for duplicate ID
                elif msg.id in seen_ids:
                    validation_results["duplicate_ids"] += 1
                    validation_results["issues"].append(f"Message {i} duplicate ID: {msg.id}")
                else:
                    seen_ids.add(msg.id)
                
                # Check for corrupted content
                if not isinstance(msg.content, str):
                    validation_results["corrupted_messages"] += 1
                    validation_results["issues"].append(f"Message {i} corrupted content type")
                
                # Check token estimate
                actual_tokens = msg._estimate_tokens(msg.content)
                if actual_tokens != msg.token_estimate:
                    validation_results["issues"].append(f"Message {i} token mismatch: {msg.token_estimate} vs {actual_tokens}")
                
                calculated_tokens += actual_tokens
            
            # Check total token calculation
            stored_total = sum(msg.token_estimate for msg in self.messages)
            if calculated_tokens != stored_total:
                validation_results["token_sum_mismatch"] = True
                validation_results["issues"].append(f"Token sum mismatch: calculated {calculated_tokens} vs stored {stored_total}")
            
            return validation_results
    
    def export_conversation_text(self, filename: Optional[str] = None, include_metadata: bool = False) -> bool:
        """Export conversation as plain text file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"conversation_export_{timestamp}.txt"
            
            with self.lock:
                lines = []
                
                if include_metadata:
                    stats = self.get_memory_stats()
                    lines.append(f"=== CONVERSATION EXPORT ===")
                    lines.append(f"Exported: {datetime.now().isoformat()}")
                    lines.append(f"Total Messages: {stats['message_count']}")
                    lines.append(f"Total Tokens: {stats['total_tokens']}")
                    lines.append(f"Condensations: {stats['condensations_performed']}")
                    lines.append(f"=" * 50)
                    lines.append("")
                
                for msg in self.messages:
                    if msg.message_type == MessageType.MOMENTUM_STATE:
                        continue  # Skip internal state messages
                    
                    timestamp = datetime.fromisoformat(msg.timestamp).strftime("%H:%M:%S")
                    header = f"[{timestamp}] {msg.message_type.value.upper()}"
                    if msg.condensed:
                        header += " (CONDENSED)"
                    
                    lines.append(header)
                    lines.append("-" * len(header))
                    lines.append(msg.content)
                    lines.append("")
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            
            if self.debug_logger:
                self.debug_logger.debug(f"Conversation exported to {filename}")
            
            return True
            
        except Exception as e:
            if self.debug_logger:
                self.debug_logger.error(f"Export failed: {e}")
            return False

# =============================================================================
# MODULE TEST FUNCTIONALITY
# =============================================================================

def test_memory_manager():
    """Test basic memory manager functionality"""
    print("DevName RPG Client - Enhanced Memory Manager (Simplified)")
    print("Testing storage and state management functionality...")
    
    # Test with auto-save enabled
    test_emm = EnhancedMemoryManager(auto_save_enabled=True, memory_file="test_memory.json")
    
    print(f"Initial state: {test_emm.get_memory_stats()}")
    
    # Test basic message operations
    test_emm.add_message("Hello there!", MessageType.USER)
    test_emm.add_message("Greetings, traveler! Welcome to the tavern.", MessageType.ASSISTANT)
    test_emm.add_message("I look around the tavern for any interesting characters.", MessageType.USER)
    
    print(f"After adding messages: {test_emm.get_memory_stats()}")
    
    # Test message categorization (simulating orchestrator)
    messages = test_emm.get_messages()
    if messages:
        test_emm.update_message_category(messages[0].id, "character_focused")
        print("Updated first message category")
    
    # Test momentum state
    test_state = {
        "narrative_pressure": 0.6,
        "pressure_source": "social",
        "antagonist_present": False
    }
    test_emm.update_momentum_state(test_state)
    retrieved_state = test_emm.get_momentum_state()
    print(f"Momentum state test: {retrieved_state}")
    
    # Test file operations
    save_success = test_emm.save_conversation("test_export.json")
    print(f"Save test: {save_success}")
    
    # Test validation
    validation = test_emm.validate_message_integrity()
    print(f"Validation: {validation['total_messages']} messages, {len(validation['issues'])} issues")
    
    # Test export
    export_success = test_emm.export_conversation_text("test_export.txt", include_metadata=True)
    print(f"Export test: {export_success}")
    
    # Test condensation candidates
    candidates = test_emm.get_condensation_candidates()
    print(f"Condensation candidates: {len(candidates)} messages")
    
    # Cleanup test files
    try:
        import os
        for test_file in ["test_memory.json", "test_memory.json.bak", "test_export.json", "test_export.txt"]:
            if os.path.exists(test_file):
                os.remove(test_file)
        print("Test files cleaned up")
    except:
        print("Warning: Could not clean up all test files")
    
    print("Memory manager test completed successfully!")

if __name__ == "__main__":
    test_memory_manager()
