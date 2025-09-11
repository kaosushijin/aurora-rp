# Chunk 1/3 - sem.py - Semantic Analysis Engine
#!/usr/bin/env python3
"""
DevName RPG Client - Semantic Analysis Engine (sem.py)
Centralizes all semantic analysis logic extracted from emm.py and sme.py
Prepares LLM requests but coordinates sending through orch.py
"""

import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

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
        """Set callback function to orchestrator for LLM requests"""
        self.orchestrator_callback = callback
    
    # =============================================================================
    # MESSAGE CATEGORIZATION FUNCTIONS
    # =============================================================================
    
    def analyze_message_semantics(self, target_message: str, context_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analyze message semantics with 3-attempt strategy
        Returns categorization results or None if all attempts fail
        """
        # Attempt 1: Full context analysis
        request = self._create_full_analysis_request(target_message, context_messages)
        result = self._request_llm_analysis(request)
        
        if result and result.success:
            parsed = self._parse_semantic_response_robust(result.data.get("response", ""), attempt=1)
            if parsed:
                return parsed
        
        # Attempt 2: Simplified analysis
        request = self._create_simple_analysis_request(target_message)
        result = self._request_llm_analysis(request)
        
        if result and result.success:
            parsed = self._parse_semantic_response_robust(result.data.get("response", ""), attempt=2)
            if parsed:
                return parsed
        
        # Attempt 3: Binary preserve/condense decision
        request = self._create_binary_analysis_request(target_message)
        result = self._request_llm_analysis(request)
        
        if result and result.success:
            parsed = self._parse_semantic_response_robust(result.data.get("response", ""), attempt=3)
            if parsed:
                return parsed
        
        # All attempts failed - return default categorization
        self._log_debug("All semantic analysis attempts failed, using default categorization")
        return {
            "importance_score": 0.4,
            "categories": ["standard"],
            "fragments": None
        }
    
    def get_highest_priority_category(self, categories: List[str]) -> str:
        """Get highest priority category from list"""
        if not categories:
            return "standard"
        
        # Find category with highest priority (lowest number)
        best_category = "standard"
        best_priority = 10
        
        for category in categories:
            if category in SEMANTIC_CATEGORIES:
                priority = SEMANTIC_CATEGORIES[category]["priority"]
                if priority < best_priority:
                    best_priority = priority
                    best_category = category
        
        return best_category
    
    def collect_preservation_candidates(self, messages: List[Any], aggressiveness_level: int = 0) -> Tuple[List[Any], List[Any]]:
        """
        Collect messages for preservation vs condensation based on semantic analysis
        Higher aggressiveness = more aggressive condensation
        """
        preserve_messages = []
        condense_candidates = []
        
        # Base preservation thresholds that get more aggressive with each pass
        base_thresholds = {
            0: 0.6,  # First pass - conservative
            1: 0.5,  # Second pass - moderate
            2: 0.4   # Third pass - aggressive
        }
        
        threshold = base_thresholds.get(aggressiveness_level, 0.4)
        
        for message in messages:
            # Always preserve recent messages (last 5)
            if len(messages) - messages.index(message) <= 5:
                preserve_messages.append(message)
                continue
            
            # Check if message should be preserved based on category
            category = getattr(message, 'content_category', 'standard')
            if category in SEMANTIC_CATEGORIES:
                preservation_ratio = SEMANTIC_CATEGORIES[category]["preservation_ratio"]
                if preservation_ratio >= threshold:
                    preserve_messages.append(message)
                else:
                    condense_candidates.append(message)
            else:
                # Unknown category - use conservative threshold
                if 0.5 >= threshold:
                    preserve_messages.append(message)
                else:
                    condense_candidates.append(message)
        
        self._log_debug(f"Preservation analysis (level {aggressiveness_level}): {len(preserve_messages)} preserve, {len(condense_candidates)} condense")
        return preserve_messages, condense_candidates

# Chunk 2/3 - sem.py - Story Momentum Analysis Functions

    # =============================================================================
    # STORY MOMENTUM ANALYSIS FUNCTIONS  
    # =============================================================================
    
    def analyze_story_momentum(self, conversation_messages: List[Dict[str, Any]], current_pressure: float, antagonist_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze story momentum and pressure from conversation context
        Returns momentum analysis results or None if analysis fails
        """
        # Prepare context within token budget
        context_messages, context_tokens = self._prepare_momentum_context(conversation_messages, max_tokens=6000)
        
        # Create momentum analysis request
        request = self._create_momentum_analysis_request(context_messages, current_pressure, antagonist_data)
        result = self._request_llm_analysis(request)
        
        if result and result.success:
            parsed = self._parse_momentum_response(result.data.get("response", ""))
            if parsed:
                return parsed
        
        # Fallback analysis using pattern detection
        self._log_debug("LLM momentum analysis failed, using pattern-based fallback")
        return self._fallback_momentum_analysis(conversation_messages, current_pressure)
    
    def generate_antagonist_data(self, story_context: Dict[str, Any], conversation_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate or enhance antagonist based on story context
        Returns antagonist data or None if generation fails
        """
        # Prepare context for antagonist generation
        context_messages, _ = self._prepare_momentum_context(conversation_messages, max_tokens=4000)
        
        # Create antagonist generation request
        request = self._create_antagonist_request(context_messages, story_context)
        result = self._request_llm_analysis(request)
        
        if result and result.success:
            parsed = self._parse_antagonist_response(result.data.get("response", ""))
            if parsed:
                return parsed
        
        # Fallback antagonist generation
        self._log_debug("LLM antagonist generation failed, using pattern-based fallback")
        return self._fallback_antagonist_generation(story_context)
    
    def detect_narrative_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Detect narrative patterns using regex and keyword analysis
        Provides fallback analysis when LLM is unavailable
        """
        patterns = {
            "conflict_indicators": [],
            "emotional_markers": [],
            "progression_signals": [],
            "character_elements": [],
            "world_building": []
        }
        
        # Conflict and tension indicators
        conflict_patterns = [
            r'\b(attack|combat|fight|battle|danger|threat|enemy|oppose|resist)\b',
            r'\b(angry|hostile|aggressive|violent|confrontation)\b',
            r'\b(chase|pursue|hunt|track|follow|escape)\b'
        ]
        
        for pattern in conflict_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns["conflict_indicators"].extend(matches)
        
        # Emotional significance markers
        emotion_patterns = [
            r'\b(love|hate|fear|anger|joy|sadness|hope|despair)\b',
            r'\b(feel|felt|emotion|heart|soul|mind)\b',
            r'\b(tear|cry|laugh|smile|frown|scream)\b'
        ]
        
        for pattern in emotion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns["emotional_markers"].extend(matches)
        
        # Story progression signals
        progression_patterns = [
            r'\b(discover|reveal|learn|find|uncover|realize)\b',
            r'\b(journey|travel|arrive|depart|enter|exit)\b',
            r'\b(begin|start|end|finish|complete|achieve)\b'
        ]
        
        for pattern in progression_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns["progression_signals"].extend(matches)
        
        # Character relationship elements
        character_patterns = [
            r'\b(friend|ally|companion|partner|enemy|rival)\b',
            r'\b(trust|betray|help|support|protect|defend)\b',
            r'\b(speak|talk|tell|ask|answer|reply)\b'
        ]
        
        for pattern in character_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns["character_elements"].extend(matches)
        
        # World building elements
        world_patterns = [
            r'\b(castle|tower|forest|mountain|river|city|village)\b',
            r'\b(magic|spell|enchant|curse|divine|sacred)\b',
            r'\b(kingdom|realm|land|territory|domain)\b'
        ]
        
        for pattern in world_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns["world_building"].extend(matches)
        
        return patterns
    
    def calculate_story_pressure(self, momentum_data: Dict[str, Any], current_pressure: float = 0.0) -> float:
        """
        Calculate narrative pressure based on momentum analysis
        Returns pressure value between 0.0 and 1.0
        """
        base_pressure = current_pressure
        
        # Factors that increase pressure
        pressure_modifiers = 0.0
        
        # Conflict indicators increase pressure
        conflict_count = len(momentum_data.get("conflict_indicators", []))
        pressure_modifiers += min(conflict_count * 0.05, 0.2)
        
        # Emotional intensity affects pressure
        emotion_count = len(momentum_data.get("emotional_markers", []))
        pressure_modifiers += min(emotion_count * 0.03, 0.15)
        
        # Story progression can increase or maintain pressure
        progression_count = len(momentum_data.get("progression_signals", []))
        pressure_modifiers += min(progression_count * 0.02, 0.1)
        
        # Antagonist presence significantly affects pressure
        if momentum_data.get("antagonist_present", False):
            pressure_modifiers += 0.15
        
        # Calculate final pressure with bounds
        new_pressure = min(1.0, max(0.0, base_pressure + pressure_modifiers))
        
        return new_pressure
    
    # =============================================================================
    # CONDENSATION SUPPORT FUNCTIONS
    # =============================================================================
    
    def create_category_aware_summary(self, messages_to_condense: List[Any]) -> Optional[str]:
        """
        Create condensed summary that preserves semantic categories
        Returns summary text or None if condensation fails
        """
        if not messages_to_condense:
            return None
        
        # Group messages by category for targeted summarization
        category_groups = {}
        for message in messages_to_condense:
            category = getattr(message, 'content_category', 'standard')
            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(message)
        
        # Create condensation request
        request = self._create_condensation_request(category_groups)
        result = self._request_llm_analysis(request)
        
        if result and result.success:
            return result.data.get("response", "")
        
        # Fallback condensation
        self._log_debug("LLM condensation failed, using extractive fallback")
        return self._fallback_condensation(messages_to_condense)
    
    # =============================================================================
    # PRIVATE HELPER FUNCTIONS
    # =============================================================================
    
    def _prepare_momentum_context(self, conversation_messages: List[Dict[str, Any]], max_tokens: int = 6000) -> Tuple[List[Dict[str, Any]], int]:
        """Prepare context for momentum analysis within token budget"""
        # Reserve 25% for analysis prompt overhead
        analysis_overhead = max_tokens // 4
        available_tokens = max_tokens - analysis_overhead
        
        # Start with recent messages and work backwards
        context_messages = []
        total_tokens = 0
        
        for message in reversed(conversation_messages):
            content = message.get("content", "")
            message_tokens = len(content) // 4  # Conservative token estimation
            
            if total_tokens + message_tokens <= available_tokens:
                context_messages.insert(0, message)
                total_tokens += message_tokens
            else:
                break
        
        self._log_debug(f"Momentum analysis context: {len(context_messages)} messages, {total_tokens} tokens")
        return context_messages, total_tokens
    
    def _fallback_momentum_analysis(self, conversation_messages: List[Dict[str, Any]], current_pressure: float) -> Dict[str, Any]:
        """Fallback momentum analysis using pattern detection"""
        # Combine recent messages for pattern analysis
        recent_text = " ".join([
            msg.get("content", "") for msg in conversation_messages[-10:]
        ])
        
        patterns = self.detect_narrative_patterns(recent_text)
        pressure = self.calculate_story_pressure(patterns, current_pressure)
        
        return {
            "pressure_level": pressure,
            "pressure_source": "pattern_analysis",
            "manifestation_type": "environmental",
            "escalation_recommended": pressure > 0.6,
            "antagonist_data": None,
            "narrative_pressure": pressure,
            "analysis_confidence": 0.3  # Lower confidence for fallback
        }
    
    def _fallback_antagonist_generation(self, story_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback antagonist generation using templates"""
        antagonist_templates = [
            {
                "name": "The Shadowed Figure",
                "motivation": "Ancient grudge against the realm",
                "methods": ["stealth", "manipulation", "dark_magic"],
                "resources": {"minions": 3, "influence": 2, "artifacts": 1},
                "commitment_level": 0.4,
                "antagonist_type": "individual"
            },
            {
                "name": "The Corrupt Order",
                "motivation": "Maintain power and control",
                "methods": ["political", "enforcement", "surveillance"],
                "resources": {"guards": 5, "influence": 4, "wealth": 3},
                "commitment_level": 0.5,
                "antagonist_type": "organization"
            }
        ]
        
        # Select template based on story context
        pressure = story_context.get("narrative_pressure", 0.5)
        template_idx = 0 if pressure < 0.5 else 1
        
        return antagonist_templates[template_idx]

# Chunk 3/3 - sem.py - LLM Request Preparation and Response Parsing

    def _fallback_condensation(self, messages_to_condense: List[Any]) -> str:
        """Fallback condensation using extractive summarization"""
        if not messages_to_condense:
            return ""
        
        # Extract key sentences from messages
        key_points = []
        for message in messages_to_condense:
            content = getattr(message, 'content', '')
            if len(content) > 50:  # Only meaningful content
                # Take first and last sentences of longer messages
                sentences = content.split('. ')
                if len(sentences) > 1:
                    key_points.append(sentences[0])
                    if len(sentences) > 2:
                        key_points.append(sentences[-1])
                else:
                    key_points.append(content[:100])  # First 100 chars
        
        return "Summary: " + " | ".join(key_points[:10])  # Limit to 10 key points
    
    # =============================================================================
    # LLM REQUEST PREPARATION FUNCTIONS
    # =============================================================================
    
    def _create_full_analysis_request(self, target_message: str, context_messages: List[Dict[str, Any]]) -> SemanticAnalysisRequest:
        """Create detailed semantic analysis request with full context"""
        context_text = "\n".join([
            f"[{i}] {msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for i, msg in enumerate(context_messages[-10:])  # Last 10 messages
        ])
        
        prompt = f"""Analyze the target message for semantic importance and categorization.

Context:
{context_text}

Target Message: {target_message}

Categories:
- story_critical: Major plot developments, key decisions, story outcomes
- character_focused: Character development, personality reveals, relationships
- relationship_dynamics: Interpersonal interactions, social developments
- emotional_significance: Strong emotions, meaningful moments, character growth
- world_building: Setting details, lore, environmental descriptions
- standard: General conversation, basic interactions

Analyze the target message and respond with JSON:
{{
    "importance_score": 0.0-1.0,
    "categories": ["primary_category", "secondary_category"],
    "reasoning": "brief explanation",
    "fragments": ["key phrase 1", "key phrase 2"] or null
}}"""

        return SemanticAnalysisRequest(
            analysis_type="categorization",
            context_data={"prompt": prompt},
            priority=3,
            timeout=20
        )
    
    def _create_simple_analysis_request(self, target_message: str) -> SemanticAnalysisRequest:
        """Create simplified semantic analysis request"""
        prompt = f"""Analyze this message for importance in an RPG story context:

Message: {target_message}

Rate importance 0.0-1.0 and categorize as one of:
story_critical, character_focused, relationship_dynamics, emotional_significance, world_building, standard

JSON response:
{{"importance_score": 0.0-1.0, "categories": ["category"]}}"""

        return SemanticAnalysisRequest(
            analysis_type="categorization", 
            context_data={"prompt": prompt},
            priority=4,
            timeout=15
        )
    
    def _create_binary_analysis_request(self, target_message: str) -> SemanticAnalysisRequest:
        """Create binary preserve/condense decision request"""
        prompt = f"""Should this RPG message be preserved or condensed?

Message: {target_message}

Consider: plot relevance, character development, world-building, emotional impact.

JSON response:
{{"preserve": true/false, "importance_score": 0.0-1.0}}"""

        return SemanticAnalysisRequest(
            analysis_type="categorization",
            context_data={"prompt": prompt}, 
            priority=5,
            timeout=10
        )
    
    def _create_momentum_analysis_request(self, context_messages: List[Dict[str, Any]], current_pressure: float, antagonist_data: Optional[Dict[str, Any]]) -> SemanticAnalysisRequest:
        """Create story momentum analysis request"""
        context_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in context_messages[-15:]  # Last 15 messages
        ])
        
        antagonist_context = ""
        if antagonist_data:
            antagonist_context = f"\nCurrent Antagonist: {antagonist_data.get('name', 'Unknown')} - {antagonist_data.get('motivation', 'Unknown motivation')}"
        
        prompt = f"""Analyze story momentum and narrative pressure in this RPG conversation.

Current Pressure Level: {current_pressure:.2f}{antagonist_context}

Recent Conversation:
{context_text}

Analyze for:
- Narrative tension and pacing
- Story progression indicators  
- Conflict escalation potential
- Character relationship dynamics
- Plot development momentum

JSON response:
{{
    "pressure_level": 0.0-1.0,
    "pressure_source": "exploration/combat/social/mystery/revelation",
    "manifestation_type": "environmental/antagonist/revelation/character",
    "escalation_recommended": true/false,
    "narrative_pressure": 0.0-1.0,
    "analysis_confidence": 0.0-1.0,
    "pressure_factors": ["factor1", "factor2"]
}}"""

        return SemanticAnalysisRequest(
            analysis_type="momentum",
            context_data={"prompt": prompt},
            priority=2,
            timeout=30
        )
    
    def _create_antagonist_request(self, context_messages: List[Dict[str, Any]], story_context: Dict[str, Any]) -> SemanticAnalysisRequest:
        """Create antagonist generation request"""
        context_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in context_messages[-10:]  # Last 10 messages
        ])
        
        pressure = story_context.get("narrative_pressure", 0.5)
        current_arc = story_context.get("story_arc", "setup")
        
        prompt = f"""Generate or enhance an antagonist for this RPG story.

Story Context:
- Current pressure: {pressure:.2f}
- Story arc: {current_arc}
- Context: {context_text}

Create an antagonist that:
- Emerges naturally from the story context
- Has believable motivation tied to recent events
- Provides appropriate challenge for current story arc
- Has specific resources and methods

JSON response:
{{
    "name": "antagonist name",
    "motivation": "clear motivation",
    "methods": ["method1", "method2", "method3"],
    "resources": {{"type1": quantity, "type2": quantity}},
    "commitment_level": 0.0-1.0,
    "antagonist_type": "individual/organization/force_of_nature",
    "current_goal": "immediate objective",
    "escalation_potential": 0.0-1.0,
    "background": "brief background"
}}"""

        return SemanticAnalysisRequest(
            analysis_type="antagonist",
            context_data={"prompt": prompt},
            priority=2,
            timeout=25
        )
    
    def _create_condensation_request(self, category_groups: Dict[str, List[Any]]) -> SemanticAnalysisRequest:
        """Create condensation request for grouped messages"""
        condensation_text = ""
        for category, messages in category_groups.items():
            condensation_text += f"\n{category.upper()}:\n"
            for msg in messages:
                content = getattr(msg, 'content', '')[:200]  # Limit length
                condensation_text += f"- {content}\n"
        
        prompt = f"""Create a condensed summary that preserves the essential semantic content of these RPG messages grouped by category:

{condensation_text}

Requirements:
- Preserve key story elements from story_critical messages
- Maintain character relationships and development
- Keep important world-building details
- Preserve emotional significance
- Create coherent narrative summary

Respond with a condensed summary that maintains semantic richness while reducing length."""

        return SemanticAnalysisRequest(
            analysis_type="condensation",
            context_data={"prompt": prompt},
            priority=3,
            timeout=25
        )
    
    # =============================================================================
    # RESPONSE PARSING FUNCTIONS
    # =============================================================================
    
    def _parse_semantic_response_robust(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Parse semantic analysis response with robust error handling"""
        if not response:
            return None
        
        # Try multiple parsing strategies
        for strategy in range(5):
            try:
                if strategy == 0:
                    # Direct JSON parse
                    return json.loads(response)
                elif strategy == 1:
                    # Extract JSON from text
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
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
