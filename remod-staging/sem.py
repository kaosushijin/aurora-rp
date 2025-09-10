# Chunk 1/4 - sem.py - Semantic Logic Module Core Components
#!/usr/bin/env python3

import json
import re
import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import httpx

class MessageType(Enum):
    """Message type enumeration for conversation tracking"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    MOMENTUM_STATE = "momentum_state"

class StoryArc(Enum):
    """Narrative progression states"""
    SETUP = "setup"
    RISING = "rising_action" 
    CLIMAX = "climax"
    RESOLUTION = "resolution"

# Semantic category preservation ratios and instructions
CONDENSATION_STRATEGIES = {
    "story_critical": {
        "threshold": 100,
        "preservation_ratio": 0.9,
        "instruction": (
            "Preserve all major plot developments, character deaths, world-changing events, "
            "key player decisions, and their consequences. Use decisive language highlighting "
            "the significance of events. Compress dialogue while maintaining essential meaning."
        )
    },
    "character_focused": {
        "threshold": 80,
        "preservation_ratio": 0.8,
        "instruction": (
            "Preserve character development moments, relationship changes, personality reveals, "
            "emotional breakthroughs, and trust/betrayal events. Maintain the emotional context "
            "and interpersonal dynamics while condensing routine interactions."
        )
    },
    "relationship_dynamics": {
        "threshold": 80,
        "preservation_ratio": 0.8,
        "instruction": (
            "Focus on alliance formations, trust building/breaking, power dynamics, conflicts "
            "between characters, and social hierarchies. Preserve the evolution of relationships "
            "while summarizing the mechanics of interactions."
        )
    },
    "emotional_significance": {
        "threshold": 70,
        "preservation_ratio": 0.75,
        "instruction": (
            "Preserve dramatic moments, emotional peaks, conflict resolution, cathartic events, "
            "and moments of tension or relief. Maintain the emotional weight and narrative impact "
            "while condensing the lead-up and aftermath."
        )
    },
    "world_building": {
        "threshold": 90,
        "preservation_ratio": 0.7,
        "instruction": (
            "Preserve new locations, cultural discoveries, lore revelations, political changes, "
            "magical/technological discoveries, and world state changes. Maintain factual accuracy "
            "and consistency while condensing exploration mechanics."
        )
    },
    "standard": {
        "threshold": 50,
        "preservation_ratio": 0.4,
        "instruction": (
            "Condense routine interactions, travel sequences, basic world interactions, and "
            "repeated activities into brief summaries. Focus on outcomes and state changes "
            "rather than detailed play-by-play descriptions."
        )
    }
}

# Story momentum pattern recognition keywords
MOMENTUM_PATTERNS = {
    "conflict": [
        "attack", "fight", "battle", "combat", "weapon", "sword", "magic", "spell",
        "defend", "block", "dodge", "strike", "hit", "damage", "hurt", "wound",
        "enemy", "foe", "opponent", "threat", "danger", "hostile", "aggressive"
    ],
    "exploration": [
        "explore", "search", "look", "examine", "investigate", "discover", "find",
        "room", "door", "path", "corridor", "chamber", "area", "location", "place",
        "hidden", "secret", "clue", "evidence", "trace", "sign", "mark"
    ],
    "social": [
        "talk", "speak", "say", "tell", "ask", "question", "answer", "reply",
        "convince", "persuade", "negotiate", "bargain", "trade", "deal", "agree",
        "friend", "ally", "companion", "trust", "help", "assist", "support"
    ],
    "mystery": [
        "strange", "odd", "unusual", "mysterious", "cryptic", "puzzle", "riddle",
        "whisper", "shadow", "darkness", "eerie", "unsettling", "disturbing",
        "clue", "evidence", "trail", "lead", "hint", "suggestion", "implication"
    ],
    "tension": [
        "tense", "nervous", "worried", "afraid", "fear", "scared", "anxious",
        "pressure", "stress", "urgent", "hurry", "rush", "quickly", "immediately",
        "danger", "threat", "risk", "warning", "alert", "caution", "careful"
    ],
    "resolution": [
        "solve", "resolved", "complete", "finish", "end", "conclude", "success",
        "victory", "win", "triumph", "achieve", "accomplish", "goal", "objective",
        "relief", "calm", "peace", "safe", "secure", "rest", "relax"
    ]
}

# Semantic analysis thresholds
SEMANTIC_THRESHOLDS = {
    "importance_score_min": 0.0,
    "importance_score_max": 1.0,
    "category_confidence_threshold": 0.6,
    "pattern_match_threshold": 2,  # Minimum keyword matches for pattern recognition
    "narrative_significance_threshold": 0.5,
    "condensation_aggressiveness_levels": 3,
    "max_analysis_attempts": 4,
    "background_analysis_interval": 15,  # Messages between comprehensive analysis
    "pressure_decay_rate": 0.02,
    "pressure_floor_increment": 0.05,
    "antagonist_commitment_thresholds": {
        "testing": 0.3,
        "engaged": 0.6,
        "desperate": 0.8,
        "cornered": 0.95
    }
}

# JSON parsing strategy configurations
JSON_PARSING_STRATEGIES = {
    "strategy_1_direct": {
        "name": "Direct JSON Parsing",
        "description": "Attempt direct json.loads() on response"
    },
    "strategy_2_substring": {
        "name": "Substring Extraction", 
        "description": "Extract JSON from within response text"
    },
    "strategy_3_pattern": {
        "name": "Pattern Matching",
        "description": "Use regex to extract specific fields"
    },
    "strategy_4_binary": {
        "name": "Binary Response",
        "description": "Simple true/false or preserve/condense parsing"
    },
    "strategy_5_fallback": {
        "name": "Complete Fallback",
        "description": "Return sensible defaults when all else fails"
    }
}

class SemanticAnalyzer:
    """Core semantic analysis functionality for message categorization and pattern recognition"""
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.analysis_cache = {}  # Cache for repeated analysis
        self.pattern_cache = {}   # Cache for pattern recognition results
    
    def get_highest_priority_category(self, categories: List[str]) -> str:
        """Return the highest priority category from a list"""
        priority_order = [
            "story_critical",
            "character_focused", 
            "relationship_dynamics",
            "emotional_significance",
            "world_building",
            "standard"
        ]
        
        for category in priority_order:
            if category in categories:
                return category
        
        return "standard"
    
    def calculate_preservation_threshold(self, category: str, aggressiveness: int = 0) -> float:
        """Calculate preservation threshold with aggressiveness adjustment"""
        base_ratio = CONDENSATION_STRATEGIES.get(category, CONDENSATION_STRATEGIES["standard"])["preservation_ratio"]
        
        # Apply aggressiveness reduction: Pass 0: base, Pass 1: -0.15, Pass 2: -0.3
        aggressiveness_reduction = aggressiveness * 0.15
        adjusted_ratio = max(0.1, base_ratio - aggressiveness_reduction)
        
        return adjusted_ratio

# Chunk 2/4 - sem.py - Pattern Recognition and Narrative Time Tracking

    def analyze_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text for story momentum patterns"""
        if not text:
            return {"patterns": {}, "dominant_pattern": None, "pattern_strength": 0.0}
        
        text_lower = text.lower()
        pattern_matches = {}
        total_matches = 0
        
        # Check each pattern category
        for pattern_type, keywords in MOMENTUM_PATTERNS.items():
            matches = [keyword for keyword in keywords if keyword in text_lower]
            pattern_matches[pattern_type] = {
                "matches": matches,
                "count": len(matches),
                "strength": len(matches) / len(keywords) if keywords else 0.0
            }
            total_matches += len(matches)
        
        # Determine dominant pattern
        dominant_pattern = None
        max_strength = 0.0
        
        for pattern_type, data in pattern_matches.items():
            if data["strength"] > max_strength and data["count"] >= SEMANTIC_THRESHOLDS["pattern_match_threshold"]:
                max_strength = data["strength"]
                dominant_pattern = pattern_type
        
        return {
            "patterns": pattern_matches,
            "dominant_pattern": dominant_pattern,
            "pattern_strength": max_strength,
            "total_keyword_matches": total_matches
        }
    
    def categorize_content_priority(self, categories: List[str], importance_score: float) -> str:
        """Determine content category with priority weighting"""
        if not categories:
            return "standard"
        
        # Weight categories by importance score
        weighted_categories = []
        for category in categories:
            if category in CONDENSATION_STRATEGIES:
                base_priority = CONDENSATION_STRATEGIES[category]["preservation_ratio"]
                weighted_priority = base_priority * importance_score
                weighted_categories.append((category, weighted_priority))
        
        if weighted_categories:
            # Sort by weighted priority and return highest
            weighted_categories.sort(key=lambda x: x[1], reverse=True)
            return weighted_categories[0][0]
        
        return self.get_highest_priority_category(categories)


class NarrativeTimeTracker:
    """Tracks narrative time progression separate from real time"""
    
    def __init__(self):
        self.total_narrative_seconds = 0.0
        self.exchange_count = 0
        self.sequence_history: List[Tuple[int, float, float]] = []  # (sequence, duration, cumulative_time)
        
        # Semantic time detection patterns
        self.quick_patterns = {
            "brief_dialogue": [r"\b(yes|no|okay|sure|maybe|perhaps)\b", r"^[^.!?]{1,20}[.!?]"],
            "immediate_action": [r"\bimmediately\b", r"\bquickly\b", r"\bright away\b", r"\bat once\b"],
            "single_action": [r"^I [a-z]+ ", r"^You [a-z]+ ", r"^[A-Z][a-z]+ [a-z]+[.!]$"]
        }
        
        self.medium_patterns = {
            "conversation": [r"\btalking\b", r"\bdiscuss\b", r"\bconversation\b", r"\bexplain\b"],
            "investigation": [r"\bsearch\b", r"\bexamine\b", r"\binvestigate\b", r"\blook around\b"],
            "planning": [r"\bplan\b", r"\bstrategy\b", r"\bthink about\b", r"\bconsider\b"]
        }
        
        self.extended_patterns = {
            "travel": [r"\btravel\b", r"\bjourney\b", r"\bwalk to\b", r"\bgo to\b"],
            "extended_action": [r"\bcarefully\b", r"\bthoroughly\b", r"\bmeticulously\b"],
            "rest": [r"\brest\b", r"\bsleep\b", r"\brelax\b", r"\bwait\b"]
        }
    
    def detect_narrative_duration(self, text: str) -> float:
        """Detect narrative time duration from text content"""
        if not text:
            return 10.0  # Default duration
        
        text_lower = text.lower()
        
        # Check for explicit time references
        time_patterns = {
            r"\b(\d+)\s*minutes?\b": lambda m: float(m.group(1)) * 60,
            r"\b(\d+)\s*hours?\b": lambda m: float(m.group(1)) * 3600,
            r"\b(\d+)\s*seconds?\b": lambda m: float(m.group(1)),
            r"\ba few minutes\b": lambda m: 180.0,
            r"\ba moment\b": lambda m: 5.0,
            r"\binstantly\b": lambda m: 1.0,
            r"\bimmediately\b": lambda m: 2.0
        }
        
        for pattern, duration_func in time_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                return duration_func(match)
        
        # Semantic duration detection
        for pattern_list in self.quick_patterns.values():
            if any(re.search(p, text_lower) for p in pattern_list):
                return 5.0  # Quick actions: 5 seconds
        
        for pattern_list in self.medium_patterns.values():
            if any(re.search(p, text_lower) for p in pattern_list):
                return 60.0  # Medium actions: 1 minute
        
        for pattern_list in self.extended_patterns.values():
            if any(re.search(p, text_lower) for p in pattern_list):
                return 300.0  # Extended actions: 5 minutes
        
        # Default duration based on text length
        word_count = len(text.split())
        if word_count < 5:
            return 5.0
        elif word_count < 20:
            return 15.0
        elif word_count < 50:
            return 30.0
        else:
            return 60.0
    
    def advance_narrative_time(self, duration: float) -> Dict[str, Any]:
        """Advance narrative time and track sequence"""
        self.exchange_count += 1
        self.total_narrative_seconds += duration
        
        # Record sequence history
        self.sequence_history.append((
            self.exchange_count,
            duration,
            self.total_narrative_seconds
        ))
        
        # Limit history size for memory efficiency
        if len(self.sequence_history) > 1000:
            self.sequence_history = self.sequence_history[-500:]
        
        return {
            "sequence": self.exchange_count,
            "duration": duration,
            "total_narrative_time": self.total_narrative_seconds,
            "formatted_time": self._format_narrative_time(self.total_narrative_seconds)
        }
    
    def get_time_since_sequence(self, sequence_number: int) -> float:
        """Get narrative time elapsed since specific sequence number"""
        if not self.sequence_history:
            return 0.0
        
        # Find the sequence in history
        for seq, duration, cumulative in self.sequence_history:
            if seq == sequence_number:
                return self.total_narrative_seconds - cumulative
        
        # If not found, estimate based on current position
        sequences_passed = max(0, self.exchange_count - sequence_number)
        return sequences_passed * 10.0  # Conservative estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get narrative time tracking statistics"""
        avg_duration = (self.total_narrative_seconds / self.exchange_count 
                       if self.exchange_count > 0 else 0.0)
        
        return {
            "total_narrative_seconds": self.total_narrative_seconds,
            "total_narrative_minutes": self.total_narrative_seconds / 60.0,
            "exchange_count": self.exchange_count,
            "average_duration_per_exchange": avg_duration,
            "narrative_time_formatted": self._format_narrative_time(self.total_narrative_seconds)
        }
    
    def _format_narrative_time(self, seconds: float) -> str:
        """Format narrative time for display"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class StoryBeatDetector:
    """Detects story beats and narrative progression markers"""
    
    def __init__(self):
        self.beat_patterns = {
            "conflict_escalation": [
                "battle intensifies", "fight escalates", "situation worsens",
                "tension rises", "pressure mounts", "stakes increase"
            ],
            "resolution_sequence": [
                "problem solved", "mystery revealed", "conflict resolved",
                "victory achieved", "goal accomplished", "quest complete"
            ],
            "transition_period": [
                "traveling to", "moving toward", "preparing for",
                "gathering supplies", "resting", "planning next"
            ],
            "stagnation_detection": [
                "nothing happens", "no progress", "still waiting",
                "same as before", "unchanged", "repetitive"
            ]
        }
    
    def detect_story_beat(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Detect what type of story beat is occurring"""
        if not text:
            return {"beat_type": None, "confidence": 0.0, "indicators": []}
        
        text_lower = text.lower()
        beat_scores = {}
        
        for beat_type, patterns in self.beat_patterns.items():
            matches = [pattern for pattern in patterns if pattern in text_lower]
            score = len(matches) / len(patterns) if patterns else 0.0
            beat_scores[beat_type] = {
                "score": score,
                "matches": matches
            }
        
        # Find dominant beat
        dominant_beat = max(beat_scores.keys(), key=lambda x: beat_scores[x]["score"])
        max_score = beat_scores[dominant_beat]["score"]
        
        return {
            "beat_type": dominant_beat if max_score > 0 else None,
            "confidence": max_score,
            "indicators": beat_scores[dominant_beat]["matches"] if max_score > 0 else [],
            "all_scores": beat_scores
        }

# Chunk 3/4 - sem.py - 5-Strategy JSON Parsing and LLM Prompt Utilities

class JSONParser:
    """Robust JSON parsing with 5-strategy fallback system"""
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.parsing_stats = {
            "strategy_1_success": 0,
            "strategy_2_success": 0, 
            "strategy_3_success": 0,
            "strategy_4_success": 0,
            "strategy_5_fallback": 0,
            "total_attempts": 0
        }
    
    def parse_semantic_response(self, response: str, attempt: int = 0) -> Optional[Dict[str, Any]]:
        """Parse LLM semantic analysis response with 5-strategy defensive handling"""
        self.parsing_stats["total_attempts"] += 1
        
        # Strategy 1: Direct JSON parsing
        result = self._strategy_1_direct_parse(response, attempt)
        if result is not None:
            self.parsing_stats["strategy_1_success"] += 1
            return result
        
        # Strategy 2: Substring extraction
        result = self._strategy_2_substring_parse(response, attempt)
        if result is not None:
            self.parsing_stats["strategy_2_success"] += 1
            return result
        
        # Strategy 3: Pattern matching and field extraction
        result = self._strategy_3_pattern_parse(response, attempt)
        if result is not None:
            self.parsing_stats["strategy_3_success"] += 1
            return result
        
        # Strategy 4: Binary response handling
        result = self._strategy_4_binary_parse(response, attempt)
        if result is not None:
            self.parsing_stats["strategy_4_success"] += 1
            return result
        
        # Strategy 5: Complete fallback
        self.parsing_stats["strategy_5_fallback"] += 1
        return self._strategy_5_fallback(attempt)
    
    def _strategy_1_direct_parse(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Strategy 1: Direct JSON parsing"""
        try:
            data = json.loads(response.strip())
            if self._validate_semantic_data(data, attempt):
                return self._inject_missing_fields(data, attempt)
        except json.JSONDecodeError:
            pass
        return None
    
    def _strategy_2_substring_parse(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Strategy 2: Extract JSON from within response text"""
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
        return None
    
    def _strategy_3_pattern_parse(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Strategy 3: Use regex to extract specific fields"""
        try:
            # Handle binary response for attempt 3
            if attempt == 3:
                if "true" in response.lower() or "preserve" in response.lower():
                    return {"importance_score": 0.8, "categories": ["story_critical"], "fragments": None}
                else:
                    return {"importance_score": 0.2, "categories": ["standard"], "fragments": None}
            
            # Try to extract numeric importance
            importance_match = re.search(r'"?importance_score"?\s*:\s*([0-9.]+)', response)
            if importance_match:
                importance = float(importance_match.group(1))
                
                # Try to extract categories
                categories_match = re.search(r'"?categories"?\s*:\s*\[(.*?)\]', response)
                categories = ["standard"]  # Default
                if categories_match:
                    category_text = categories_match.group(1)
                    # Extract quoted strings
                    category_matches = re.findall(r'"([^"]+)"', category_text)
                    if category_matches:
                        categories = category_matches
                
                return {
                    "importance_score": importance,
                    "categories": categories,
                    "fragments": None
                }
        except (ValueError, AttributeError):
            pass
        return None
    
    def _strategy_4_binary_parse(self, response: str, attempt: int) -> Optional[Dict[str, Any]]:
        """Strategy 4: Simple binary response parsing"""
        if attempt == 3:  # Binary response expected
            response_lower = response.lower()
            if any(word in response_lower for word in ["true", "preserve", "keep", "important"]):
                return {"importance_score": 0.8, "categories": ["story_critical"], "fragments": None}
            elif any(word in response_lower for word in ["false", "condense", "remove", "unimportant"]):
                return {"importance_score": 0.2, "categories": ["standard"], "fragments": None}
        
        # Default injection based on attempt type
        if attempt == 2:  # Simple response
            return {"importance_score": 0.4, "categories": ["standard"], "fragments": None}
        
        return None
    
    def _strategy_5_fallback(self, attempt: int) -> Dict[str, Any]:
        """Strategy 5: Complete fallback with sensible defaults"""
        return {
            "importance_score": 0.4,
            "categories": ["standard"],
            "fragments": None,
            "fallback_used": True
        }
    
    def _validate_semantic_data(self, data: Dict[str, Any], attempt: int) -> bool:
        """Validate that semantic analysis data has required fields"""
        if not isinstance(data, dict):
            return False
        
        if attempt == 3:  # Binary response
            return "preserve" in data or "importance_score" in data
        
        required_fields = ["importance_score", "categories"]
        return all(field in data for field in required_fields)
    
    def _inject_missing_fields(self, data: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """Inject missing fields with sensible defaults"""
        if attempt == 3:  # Binary response
            preserve = data.get("preserve", False)
            if "importance_score" not in data:
                data["importance_score"] = 0.8 if preserve else 0.2
            if "categories" not in data:
                data["categories"] = ["story_critical"] if preserve else ["standard"]
        
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
    
    def get_parsing_stats(self) -> Dict[str, Any]:
        """Get statistics on parsing strategy usage"""
        total = self.parsing_stats["total_attempts"]
        if total == 0:
            return self.parsing_stats
        
        stats_with_percentages = dict(self.parsing_stats)
        for key, value in self.parsing_stats.items():
            if key != "total_attempts":
                percentage = (value / total) * 100
                stats_with_percentages[f"{key}_percentage"] = percentage
        
        return stats_with_percentages


class LLMPromptGenerator:
    """Generate semantic analysis prompts for different analysis strategies"""
    
    def __init__(self):
        self.prompt_templates = {
            "comprehensive": self._create_comprehensive_prompt,
            "simple": self._create_simple_prompt,
            "binary": self._create_binary_prompt,
            "pattern_analysis": self._create_pattern_analysis_prompt,
            "condensation": self._create_condensation_prompt
        }
    
    def generate_semantic_analysis_prompt(self, content: str, context: Optional[Dict[str, Any]] = None, strategy: str = "comprehensive") -> str:
        """Generate semantic analysis prompt based on strategy"""
        if strategy not in self.prompt_templates:
            strategy = "comprehensive"
        
        return self.prompt_templates[strategy](content, context)
    
    def _create_comprehensive_prompt(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create comprehensive semantic analysis prompt"""
        base_prompt = f"""Analyze this RPG conversation message for semantic significance and categorization.

Message: {content}

Evaluate across these dimensions:
1. **Story Impact**: Major plot developments, character deaths, world changes, key decisions
2. **Character Development**: Relationship changes, personality reveals, emotional moments
3. **Relationship Dynamics**: Trust/betrayal, alliances, conflicts, power dynamics
4. **Emotional Weight**: Dramatic moments, tension, relief, cathartic events
5. **World Building**: New locations, lore, cultural discoveries, political changes
6. **Routine Content**: Standard interactions, travel, basic activities

Categories: story_critical, character_focused, relationship_dynamics, emotional_significance, world_building, standard

Return JSON format:
{{
  "importance_score": 0.0-1.0,
  "categories": ["primary_category", "secondary_category"],
  "narrative_significance": "brief explanation of why this matters to the story"
}}"""
        
        if context:
            base_prompt += f"\n\nAdditional Context: {context.get('narrative_context', '')}"
        
        return base_prompt
    
    def _create_simple_prompt(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create simplified analysis prompt"""
        return f"""Categorize this RPG message and rate its story importance:

Message: {content}

Categories: story_critical, character_focused, relationship_dynamics, emotional_significance, world_building, standard

Return JSON:
{{
  "importance_score": 0.0-1.0,
  "categories": ["primary_category"]
}}"""
    
    def _create_binary_prompt(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create binary preserve/condense prompt"""
        return f"""Should this RPG message be preserved or condensed for story continuity?

Message: {content}

Return JSON:
{{
  "preserve": true/false
}}"""
    
    def _create_pattern_analysis_prompt(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create pattern analysis prompt for story momentum"""
        return f"""Analyze this text for RPG story momentum patterns:

Text: {content}

Identify dominant patterns from: conflict, exploration, social, mystery, tension, resolution

Return JSON:
{{
  "dominant_pattern": "pattern_name",
  "pattern_strength": 0.0-1.0,
  "narrative_pressure_change": -0.5 to +0.5
}}"""
    
    def _create_condensation_prompt(self, category: str, messages: List[str]) -> str:
        """Create category-aware condensation prompt"""
        strategy = CONDENSATION_STRATEGIES.get(category, CONDENSATION_STRATEGIES["standard"])
        instruction = strategy["instruction"]
        
        content_block = "\n".join(messages)
        
        return f"""Condense the following {category} conversation content according to these guidelines:

{instruction}

Content to condense:
{content_block}

Return a concise summary that preserves the essential elements for this category while minimizing length. Maintain narrative continuity and emotional context."""

# Chunk 4/4 - sem.py - Background Processing Helpers and Main Semantic Processor

class BackgroundAnalysisManager:
    """Manages background semantic analysis tasks and threading"""
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.active_tasks = {}
        self.task_results = {}
        self.analysis_queue = asyncio.Queue()
        self.processing = False
    
    async def queue_analysis(self, task_id: str, analysis_func, *args, **kwargs) -> str:
        """Queue background analysis task"""
        task_data = {
            "id": task_id,
            "function": analysis_func,
            "args": args,
            "kwargs": kwargs,
            "timestamp": time.time()
        }
        
        await self.analysis_queue.put(task_data)
        
        if self.debug_logger:
            self.debug_logger.debug(f"Queued background analysis: {task_id}")
        
        return task_id
    
    async def start_processing(self):
        """Start background processing loop"""
        self.processing = True
        
        while self.processing:
            try:
                # Wait for task with timeout
                task_data = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)
                
                if task_data:
                    await self._process_task(task_data)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.error(f"Background processing error: {e}")
    
    async def _process_task(self, task_data: Dict[str, Any]):
        """Process individual analysis task"""
        task_id = task_data["id"]
        
        try:
            self.active_tasks[task_id] = task_data
            
            # Execute analysis function
            result = await task_data["function"](*task_data["args"], **task_data["kwargs"])
            
            # Store result
            self.task_results[task_id] = {
                "result": result,
                "completed_at": time.time(),
                "success": True
            }
            
            if self.debug_logger:
                self.debug_logger.debug(f"Completed background analysis: {task_id}")
                
        except Exception as e:
            self.task_results[task_id] = {
                "result": None,
                "error": str(e),
                "completed_at": time.time(),
                "success": False
            }
            
            if self.debug_logger:
                self.debug_logger.error(f"Background analysis failed {task_id}: {e}")
        
        finally:
            # Clean up active task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of completed analysis task"""
        return self.task_results.get(task_id)
    
    def is_task_complete(self, task_id: str) -> bool:
        """Check if analysis task is complete"""
        return task_id in self.task_results
    
    def cleanup_old_results(self, max_age_seconds: int = 3600):
        """Clean up old task results to prevent memory leaks"""
        current_time = time.time()
        expired_tasks = []
        
        for task_id, result_data in self.task_results.items():
            if current_time - result_data["completed_at"] > max_age_seconds:
                expired_tasks.append(task_id)
        
        for task_id in expired_tasks:
            del self.task_results[task_id]
        
        if expired_tasks and self.debug_logger:
            self.debug_logger.debug(f"Cleaned up {len(expired_tasks)} expired analysis results")
    
    def stop_processing(self):
        """Stop background processing"""
        self.processing = False


class SemanticProcessor:
    """Main semantic processing interface that coordinates all semantic analysis"""
    
    def __init__(self, debug_logger=None, mcp_client=None):
        self.debug_logger = debug_logger
        self.mcp_client = mcp_client
        
        # Initialize components
        self.analyzer = SemanticAnalyzer(debug_logger)
        self.json_parser = JSONParser(debug_logger)
        self.prompt_generator = LLMPromptGenerator()
        self.narrative_timer = NarrativeTimeTracker()
        self.story_beat_detector = StoryBeatDetector()
        self.background_manager = BackgroundAnalysisManager(debug_logger)
        
        # Analysis statistics
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "background_analyses": 0,
            "cached_results": 0
        }
    
    async def analyze_message_semantics(self, content: str, context: Optional[Dict[str, Any]] = None, 
                                      strategy: str = "comprehensive") -> Optional[Dict[str, Any]]:
        """Main entry point for semantic message analysis"""
        if not content or not self.mcp_client:
            return None
        
        self.analysis_stats["total_analyses"] += 1
        
        # Check cache first
        cache_key = f"{content[:100]}_{strategy}"
        if hasattr(self, '_analysis_cache') and cache_key in self._analysis_cache:
            self.analysis_stats["cached_results"] += 1
            return self._analysis_cache[cache_key]
        
        # Analyze text patterns immediately
        pattern_analysis = self.analyzer.analyze_text_patterns(content)
        
        # Detect narrative duration
        narrative_duration = self.narrative_timer.detect_narrative_duration(content)
        
        # Detect story beat
        story_beat = self.story_beat_detector.detect_story_beat(content, context)
        
        # Generate LLM analysis prompt
        prompt = self.prompt_generator.generate_semantic_analysis_prompt(content, context, strategy)
        
        # Attempt LLM analysis with multiple strategies
        llm_analysis = await self._attempt_llm_analysis(prompt, strategy)
        
        if llm_analysis:
            self.analysis_stats["successful_analyses"] += 1
            
            # Combine all analysis results
            result = {
                "importance_score": llm_analysis.get("importance_score", 0.4),
                "categories": llm_analysis.get("categories", ["standard"]),
                "pattern_analysis": pattern_analysis,
                "narrative_duration": narrative_duration,
                "story_beat": story_beat,
                "narrative_significance": llm_analysis.get("narrative_significance", ""),
                "analysis_strategy": strategy,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            if not hasattr(self, '_analysis_cache'):
                self._analysis_cache = {}
            self._analysis_cache[cache_key] = result
            
            # Limit cache size
            if len(self._analysis_cache) > 100:
                # Remove oldest entries
                oldest_keys = list(self._analysis_cache.keys())[:20]
                for key in oldest_keys:
                    del self._analysis_cache[key]
            
            return result
        else:
            self.analysis_stats["failed_analyses"] += 1
            return None
    
    async def _attempt_llm_analysis(self, prompt: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Attempt LLM analysis with multiple fallback strategies"""
        strategies = ["comprehensive", "simple", "binary"]
        
        # If specific strategy requested, try it first
        if strategy in strategies:
            strategies.remove(strategy)
            strategies.insert(0, strategy)
        
        for attempt, current_strategy in enumerate(strategies):
            try:
                # Send prompt to LLM via MCP
                messages = [{"role": "system", "content": prompt}]
                response = await self.mcp_client.send_request(messages)
                
                if response:
                    # Parse response using robust JSON parser
                    analysis = self.json_parser.parse_semantic_response(response, attempt)
                    if analysis and not analysis.get("fallback_used", False):
                        return analysis
                        
            except Exception as e:
                if self.debug_logger:
                    self.debug_logger.debug(f"LLM analysis attempt {attempt + 1} failed: {e}")
                continue
        
        # Return fallback result if all strategies failed
        return {
            "importance_score": 0.4,
            "categories": ["standard"],
            "analysis_failed": True
        }
    
    async def analyze_message_background(self, content: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Queue message analysis for background processing"""
        task_id = f"analysis_{int(time.time() * 1000)}"
        
        await self.background_manager.queue_analysis(
            task_id,
            self.analyze_message_semantics,
            content,
            context,
            "comprehensive"
        )
        
        self.analysis_stats["background_analyses"] += 1
        return task_id
    
    def advance_narrative_time(self, content: str) -> Dict[str, Any]:
        """Advance narrative time based on content analysis"""
        duration = self.narrative_timer.detect_narrative_duration(content)
        return self.narrative_timer.advance_narrative_time(duration)
    
    def should_preserve_message(self, analysis: Dict[str, Any], aggressiveness: int = 0) -> bool:
        """Determine if message should be preserved during condensation"""
        if not analysis:
            return False
        
        categories = analysis.get("categories", ["standard"])
        importance = analysis.get("importance_score", 0.4)
        
        # Get highest priority category
        highest_category = self.analyzer.get_highest_priority_category(categories)
        
        # Calculate preservation threshold
        threshold = self.analyzer.calculate_preservation_threshold(highest_category, aggressiveness)
        
        return importance >= threshold
    
    def get_condensation_instruction(self, category: str) -> str:
        """Get condensation instruction for specific category"""
        strategy = CONDENSATION_STRATEGIES.get(category, CONDENSATION_STRATEGIES["standard"])
        return strategy["instruction"]
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all semantic components"""
        return {
            "analysis_stats": self.analysis_stats,
            "parsing_stats": self.json_parser.get_parsing_stats(),
            "narrative_time_stats": self.narrative_timer.get_stats(),
            "background_tasks": {
                "active": len(self.background_manager.active_tasks),
                "completed": len(self.background_manager.task_results)
            },
            "cache_size": len(getattr(self, '_analysis_cache', {}))
        }
    
    async def start_background_processing(self):
        """Start background analysis processing"""
        await self.background_manager.start_processing()
    
    def stop_background_processing(self):
        """Stop background analysis processing"""
        self.background_manager.stop_processing()


# Standalone utility functions for testing and integration
def create_semantic_processor(debug_logger=None, mcp_client=None) -> SemanticProcessor:
    """Factory function to create configured semantic processor"""
    return SemanticProcessor(debug_logger, mcp_client)

def get_default_semantic_thresholds() -> Dict[str, Any]:
    """Get copy of default semantic thresholds for configuration"""
    return dict(SEMANTIC_THRESHOLDS)

def get_condensation_strategies() -> Dict[str, Any]:
    """Get copy of condensation strategies for reference"""
    return dict(CONDENSATION_STRATEGIES)

def get_momentum_patterns() -> Dict[str, List[str]]:
    """Get copy of momentum patterns for reference"""
    return dict(MOMENTUM_PATTERNS)


# Main entry point for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_semantic_processor():
        processor = SemanticProcessor()
        
        test_messages = [
            "I draw my sword and attack the dragon fiercely!",
            "I carefully examine the ancient runes on the wall.",
            "The mysterious figure whispers something unsettling.",
            "I talk to the shopkeeper about buying supplies."
        ]
        
        print("Semantic Processor Test:")
        for msg in test_messages:
            pattern_analysis = processor.analyzer.analyze_text_patterns(msg)
            narrative_time = processor.advance_narrative_time(msg)
            story_beat = processor.story_beat_detector.detect_story_beat(msg)
            
            print(f"\nMessage: {msg}")
            print(f"Dominant Pattern: {pattern_analysis['dominant_pattern']}")
            print(f"Narrative Duration: {narrative_time['duration']}s")
            print(f"Story Beat: {story_beat['beat_type']}")
        
        print(f"\nStats: {processor.get_comprehensive_stats()}")
    
    asyncio.run(test_semantic_processor())
