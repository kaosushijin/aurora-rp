# Chunk 1/4 - sem.py - Header, Imports, and Core Classes

# CRITICAL: Before modifying this module, READ genai.txt for hub-and-spoke architecture rules, module interconnects, and orchestrator coordination patterns. Violating these principles will break the remodularization.

#!/usr/bin/env python3
"""
DevName RPG Client - Semantic Analysis Engine (sem.py)
Centralized semantic analysis logic extracted from emm.py and sme.py
Remodularized for hub-and-spoke architecture
FIXED: Added missing validate_input() method for orchestrator input processing
"""

import asyncio
import time
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Ensure current directory is in Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# NOTE: sem.py should NOT import mcp directly - all LLM requests go through orchestrator
# Any LLM communication should be coordinated through the orchestrator callback pattern

# =============================================================================
# SEMANTIC CATEGORIZATION SYSTEM
# =============================================================================

# Semantic importance categories with preservation ratios
SEMANTIC_CATEGORIES = {
    "story_critical": {"preservation_ratio": 0.9, "priority": 1},
    "character_focused": {"preservation_ratio": 0.8, "priority": 2}, 
    "relationship_dynamics": {"preservation_ratio": 0.7, "priority": 3},
    "emotional_significance": {"preservation_ratio": 0.6, "priority": 4},
    "world_building": {"preservation_ratio": 0.5, "priority": 5},
    "standard": {"preservation_ratio": 0.4, "priority": 6}
}

@dataclass
class SemanticAnalysisRequest:
    """Request format for semantic analysis through orchestrator"""
    analysis_type: str  # "categorization", "momentum", "antagonist", "condensation"
    context_data: Dict[str, Any]
    priority: int = 5
    timeout: int = 30

@dataclass
class SemanticAnalysisResult:
    """Result format from semantic analysis"""
    success: bool
    analysis_type: str
    data: Dict[str, Any]
    confidence: float = 0.0
    error_message: Optional[str] = None

class SemanticAnalysisEngine:
    """
    Centralized semantic analysis without direct LLM calls
    All LLM requests coordinated through orchestrator
    """
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.orchestrator_callback = None  # Set by orchestrator
    
    def set_orchestrator_callback(self, callback):
        """Set callback for orchestrator communication"""
        self.orchestrator_callback = callback

# Chunk 2/4 - sem.py - validate_input() Method (CRITICAL FIX)

    def validate_input(self, content: str) -> Dict[str, Any]:
        """
        Validate and categorize user input for orchestrator processing
        CRITICAL FIX: This method was missing and blocking the entire input pipeline
        Called by orchestrator for every user input before processing
        """
        try:
            # Basic validation first
            if not content or not isinstance(content, str):
                return {
                    "valid": False,
                    "category": None,
                    "confidence": 0.0,
                    "error": "Empty or invalid input"
                }
            
            # Clean and prepare content
            content_clean = content.strip()
            if not content_clean:
                return {
                    "valid": False, 
                    "category": None,
                    "confidence": 0.0,
                    "error": "Input contains only whitespace"
                }
            
            # Length validation
            if len(content_clean) > 4000:  # Max reasonable input length
                return {
                    "valid": False,
                    "category": None, 
                    "confidence": 0.0,
                    "error": f"Input too long ({len(content_clean)} chars, max 4000)"
                }
            
            # Category detection logic
            content_lower = content_clean.lower()
            
            # Command detection - highest priority
            if content_lower.startswith('/'):
                command_name = content_lower.split()[0] if content_lower.split() else "/"
                
                # Known commands list from ncui.py
                valid_commands = ['/help', '/clear', '/stats', '/quit', '/exit', '/theme', '/analyze']
                
                if command_name in valid_commands:
                    return {
                        "valid": True,
                        "category": "command",
                        "confidence": 1.0,
                        "error": None
                    }
                else:
                    return {
                        "valid": False,
                        "category": "command",
                        "confidence": 0.8,
                        "error": f"Unknown command: {command_name}"
                    }
            
            # Meta queries - system questions about the game/interface
            meta_indicators = [
                "help", "how do", "what is", "explain", "show me", "tell me about",
                "what can", "how to", "status", "info", "information"
            ]
            
            if any(indicator in content_lower for indicator in meta_indicators):
                return {
                    "valid": True,
                    "category": "meta",
                    "confidence": 0.7,
                    "error": None
                }
            
            # Question detection - direct questions to the system
            question_indicators = ["?", "who", "what", "when", "where", "why", "how"]
            
            if (content_clean.endswith("?") or 
                any(content_lower.startswith(q) for q in question_indicators)):
                return {
                    "valid": True,
                    "category": "query", 
                    "confidence": 0.6,
                    "error": None
                }
            
            # Default to narrative - story/roleplay content (primary use case for RPG client)
            return {
                "valid": True,
                "category": "narrative",
                "confidence": 0.8,
                "error": None
            }
            
        except Exception as e:
            self._log_debug(f"validate_input error: {e}")
            return {
                "valid": False,
                "category": None,
                "confidence": 0.0,
                "error": f"Validation error: {str(e)}"
            }

# Chunk 3/4 - sem.py - Core Semantic Analysis Methods

    def analyze_conversation(self, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conversation for semantic importance
        Coordinates with orchestrator for LLM analysis requests
        """
        if not conversation_context:
            return {"status": "no_context", "analyzed_messages": 0}
        
        try:
            self._log_debug(f"Starting semantic analysis of {len(conversation_context)} messages")
            
            # Prepare analysis results
            analysis_results = {
                "status": "completed",
                "analyzed_messages": len(conversation_context),
                "semantic_data": {},
                "timestamp": time.time()
            }
            
            # Process each message for semantic analysis
            for i, message in enumerate(conversation_context):
                message_id = message.get("id", f"msg_{i}")
                content = message.get("content", "")
                
                if not content or len(content.strip()) < 10:
                    # Skip very short messages
                    continue
                
                # Request semantic analysis through orchestrator
                semantic_result = self._analyze_message_semantic(content, message_id)
                
                if semantic_result:
                    analysis_results["semantic_data"][message_id] = semantic_result
            
            self._log_debug(f"Semantic analysis completed: {len(analysis_results['semantic_data'])} messages analyzed")
            return analysis_results
            
        except Exception as e:
            self._log_debug(f"Semantic analysis failed: {e}")
            return {"status": "error", "error": str(e), "analyzed_messages": 0}
    
    def _analyze_message_semantic(self, content: str, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Analyze single message for semantic importance
        Uses both LLM analysis and pattern-based fallbacks
        """
        try:
            # Prepare analysis request
            request = SemanticAnalysisRequest(
                analysis_type="categorization",
                context_data={
                    "message_id": message_id,
                    "content": content,
                    "analysis_prompt": self._build_semantic_prompt(content)
                }
            )
            
            # Request LLM analysis through orchestrator
            llm_result = self._request_llm_analysis(request)
            
            if llm_result and llm_result.success:
                # Parse and validate LLM response
                parsed_result = self._parse_semantic_response_robust(
                    llm_result.data.get("response", ""), attempt=1
                )
                
                if parsed_result and self._validate_semantic_data(parsed_result, 1):
                    return self._inject_missing_fields(parsed_result, 1)
            
            # Fallback to pattern-based analysis
            self._log_debug(f"Using pattern-based analysis for message {message_id}")
            return self._pattern_extract_semantic(content)
            
        except Exception as e:
            self._log_debug(f"Message semantic analysis failed for {message_id}: {e}")
            return None
    
    def _build_semantic_prompt(self, content: str) -> str:
        """Build prompt for semantic analysis"""
        return f"""
Analyze this RPG conversation message for semantic importance:

Message: "{content}"

Provide analysis in JSON format:
{{
    "importance_score": 0.0-1.0,
    "categories": ["story_critical", "character_focused", "relationship_dynamics", "emotional_significance", "world_building", "standard"]
}}

Consider:
- Story progression and plot significance
- Character development and relationships  
- World-building and lore establishment
- Emotional impact and memorable moments
- Creative or unique content
"""
    
    def _parse_semantic_response_robust(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """
        Robust parsing of semantic analysis response with multiple fallback strategies
        """
        if not response:
            return None
        
        # Try multiple parsing strategies
        strategies = [1, 2, 3, 4]  # JSON, key-value, pattern, fallback
        
        for strategy in strategies:
            try:
                if strategy == 1:
                    # JSON parsing
                    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                elif strategy == 2:
                    # Parse key-value pairs
                    return self._extract_key_values(response)
                elif strategy == 3:
                    # Pattern-based extraction
                    return self._pattern_extract_semantic(response)
                elif strategy == 4:
                    # Fallback parsing for specific attempt
                    return self._fallback_parse_semantic(response, attempt)
            except (json.JSONDecodeError, Exception):
                continue
        
        return None
    
    def _parse_momentum_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse momentum analysis response"""
        return self._parse_semantic_response_robust(response, attempt=1)
    
    def _parse_antagonist_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse antagonist generation response"""
        return self._parse_semantic_response_robust(response, attempt=1)

# Chunk 4/4 - sem.py - Helper Methods and Utilities

    def _extract_key_values(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text response"""
        result = {}
        
        # Look for score patterns
        score_match = re.search(r'importance[_\s]*score[:\s]*([\d.]+)', text, re.IGNORECASE)
        if score_match:
            result["importance_score"] = float(score_match.group(1))
        
        # Look for categories
        category_patterns = [
            r'categor(?:y|ies)[:\s]*\[([^\]]+)\]',
            r'categor(?:y|ies)[:\s]*([a-z_,\s]+)',
        ]
        
        for pattern in category_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                categories_text = match.group(1)
                categories = [cat.strip().strip('"\'') for cat in categories_text.split(',')]
                result["categories"] = categories
                break
        
        return result if result else None
    
    def _pattern_extract_semantic(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract semantic data using patterns"""
        # Default values
        result = {
            "importance_score": 0.4,
            "categories": ["standard"]
        }
        
        # Check for high importance indicators
        high_importance = any(word in text.lower() for word in [
            "critical", "important", "significant", "major", "key", "essential"
        ])
        
        if high_importance:
            result["importance_score"] = 0.7
        
        # Check for category indicators
        if any(word in text.lower() for word in ["story", "plot", "critical"]):
            result["categories"] = ["story_critical"]
        elif any(word in text.lower() for word in ["character", "personality", "development"]):
            result["categories"] = ["character_focused"]
        elif any(word in text.lower() for word in ["emotion", "feel", "heart"]):
            result["categories"] = ["emotional_significance"]
        
        return result
    
    def _fallback_parse_semantic(self, response: str, attempt: int) -> Dict[str, Any]:
        """Fallback parsing based on attempt number"""
        if attempt == 3:  # Binary decision attempt
            preserve = any(word in response.lower() for word in ["preserve", "keep", "important", "yes", "true"])
            return {
                "importance_score": 0.6 if preserve else 0.3,
                "categories": ["standard"]
            }
        
        return {
            "importance_score": 0.4,
            "categories": ["standard"]
        }
    
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
    
    def _request_llm_analysis(self, request: SemanticAnalysisRequest) -> Optional[SemanticAnalysisResult]:
        """Request LLM analysis through orchestrator callback"""
        if not self.orchestrator_callback:
            self._log_debug("No orchestrator callback available for LLM request")
            return None
        
        try:
            return self.orchestrator_callback(request)
        except Exception as e:
            self._log_debug(f"Orchestrator callback failed: {e}")
            return None
    
    def _log_debug(self, message: str):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(f"SEM: {message}")

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'SemanticAnalysisEngine',
    'SemanticAnalysisRequest', 
    'SemanticAnalysisResult',
    'SEMANTIC_CATEGORIES'
]
