# Chunk 3/6 - nci_display.py - Message Display Module
#!/usr/bin/env python3
"""
DevName RPG Client - Message Display Module (nci_display.py)
Module architecture and interconnects documented in genai.txt
Extracted from nci.py for better separation of concerns
"""

import time
import textwrap
from typing import List

class DisplayMessage:
    """Message display with formatting"""
    
    def __init__(self, content: str, msg_type: str, timestamp: str = None):
        self.content = content
        self.msg_type = msg_type
        self.timestamp = timestamp or time.strftime("%H:%M:%S")
        self.wrapped_lines = []
    
    def format_for_display(self, max_width: int = 80) -> List[str]:
        """
        Format message with prefix system:
        - user: "You: "
        - assistant: "GM: " (changed from "AI: ")
        - system: " : " (changed from "System: " for cleaner look)
        - error: "Error: "
        """
        prefix_map = {
            'user': 'You',
            'assistant': 'GM',      # Changed from 'AI'
            'system': ' ',          # Changed from 'System' for cleaner display
            'error': 'Error'
        }
        
        prefix = prefix_map.get(self.msg_type, 'Unknown')
        header = f"[{self.timestamp}] {prefix}: "
        
        # Calculate available width for content
        content_width = max(20, max_width - len(header))
        
        # Handle empty content
        if not self.content.strip():
            return [header.rstrip()]
        
        # Wrap the content
        wrapped_content = textwrap.wrap(
            self.content, 
            width=content_width,
            break_long_words=True,
            break_on_hyphens=True,
            expand_tabs=True,
            replace_whitespace=True
        )
        
        if not wrapped_content:
            wrapped_content = [""]
        
        # Format lines
        lines = []
        for i, line in enumerate(wrapped_content):
            if i == 0:
                lines.append(header + line)
            else:
                # Indent continuation lines
                indent = " " * len(header)
                lines.append(indent + line)
        
        self.wrapped_lines = lines
        return lines

class InputValidator:
    """Input validation with multi-line support"""
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
    
    def validate(self, text: str) -> tuple[bool, str]:
        """Validate input text with specific error messages"""
        if not text.strip():
            return False, "Empty input"
        
        estimated_tokens = len(text) // 4
        if estimated_tokens > self.max_tokens:
            return False, f"Input too long: {estimated_tokens} tokens (max: {self.max_tokens})"
        
        # Check for reasonable line count
        lines = text.split('\n')
        if len(lines) > 20:
            return False, f"Too many lines: {len(lines)} (max: 20)"
        
        return True, ""
