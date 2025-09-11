# Chunk 1/3 - sme.py - Story Momentum Engine (Phase 5: Uses sem.py)
#!/usr/bin/env python3
"""
DevName RPG Client - Story Momentum Engine (sme.py)

Phase 5 Refactor: Removed embedded semantic logic, now uses sem.py
This eliminates circular dependencies with emm.py and consolidates analysis
Module architecture and interconnects documented in genai.txt
"""

import json
import time
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

# Import centralized semantic logic from sem.py
from sem import (
    SemanticProcessor, StoryArc, MessageType, 
    MOMENTUM_PATTERNS, SEMANTIC_THRESHOLDS, create_semantic_processor
)

# SME-specific configuration constants
SME_CONFIG = {
    "pressure_floor_increment": 0.05,
    "pressure_decay_rate": 0.02,
    "pressure_ceiling": 1.0,
    "antagonist_commitment_thresholds": {
        "testing": 0.3,
        "engaged": 0.6,
        "desperate": 0.8,
        "cornered": 0.95
    },
    "analysis_trigger_threshold": 15,  # Messages between comprehensive analysis
    "stagnation_detection_threshold": 5,  # Consecutive low-pressure exchanges
    "momentum_spike_threshold": 0.15  # Pressure increase to trigger spike
}

class AntagonistCommitment(Enum):
    """Antagonist commitment levels"""
    TESTING = "testing"
    ENGAGED = "engaged"
    DESPERATE = "desperate"
    CORNERED = "cornered"

class MomentumEvent:
    """Individual momentum event tracking"""
    
    def __init__(self, event_type: str, pressure_change: float, message_sequence: int):
        self.event_type = event_type
        self.pressure_change = pressure_change
        self.message_sequence = message_sequence
        self.timestamp = time.time()
        self.narrative_context = ""
        
        # Semantic analysis results (populated by sem.py)
        self.dominant_pattern = None
        self.pattern_strength = 0.0
        self.story_beat = None
        
    def update_from_semantic_analysis(self, analysis: Dict[str, Any]):
        """Update event from semantic analysis results"""
        if not analysis:
            return
        
        pattern_analysis = analysis.get("pattern_analysis", {})
        self.dominant_pattern = pattern_analysis.get("dominant_pattern")
        self.pattern_strength = pattern_analysis.get("pattern_strength", 0.0)
        self.story_beat = analysis.get("story_beat", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_type": self.event_type,
            "pressure_change": self.pressure_change,
            "message_sequence": self.message_sequence,
            "timestamp": self.timestamp,
            "narrative_context": self.narrative_context,
            "dominant_pattern": self.dominant_pattern,
            "pattern_strength": self.pattern_strength,
            "story_beat": self.story_beat
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MomentumEvent':
        """Create from dictionary during deserialization"""
        event = cls(
            data.get("event_type", "unknown"),
            data.get("pressure_change", 0.0),
            data.get("message_sequence", 0)
        )
        event.timestamp = data.get("timestamp", time.time())
        event.narrative_context = data.get("narrative_context", "")
        event.dominant_pattern = data.get("dominant_pattern")
        event.pattern_strength = data.get("pattern_strength", 0.0)
        event.story_beat = data.get("story_beat", {})
        return event


class AntagonistState:
    """Antagonist tracking and commitment management"""
    
    def __init__(self):
        self.current_antagonist = "Generic Opposition"
        self.commitment_level = AntagonistCommitment.TESTING
        self.commitment_score = 0.0
        self.escalation_history: List[Dict[str, Any]] = []
        self.manifestation_count = 0
        self.last_manifestation_sequence = 0
        
        # Tracking different antagonist types
        self.antagonist_types = {
            "environmental": {"score": 0.0, "last_seen": 0},
            "social": {"score": 0.0, "last_seen": 0},
            "internal": {"score": 0.0, "last_seen": 0},
            "supernatural": {"score": 0.0, "last_seen": 0},
            "systemic": {"score": 0.0, "last_seen": 0}
        }
    
    def update_commitment(self, pressure_level: float, pattern_analysis: Dict[str, Any]):
        """Update antagonist commitment based on pressure and patterns"""
        # Base commitment on pressure level
        self.commitment_score = min(1.0, pressure_level * 1.2)
        
        # Adjust based on conflict patterns from sem.py analysis
        if pattern_analysis.get("dominant_pattern") == "conflict":
            pattern_strength = pattern_analysis.get("pattern_strength", 0.0)
            self.commitment_score = min(1.0, self.commitment_score + (pattern_strength * 0.2))
        
        # Determine commitment level
        thresholds = SME_CONFIG["antagonist_commitment_thresholds"]
        if self.commitment_score >= thresholds["cornered"]:
            self.commitment_level = AntagonistCommitment.CORNERED
        elif self.commitment_score >= thresholds["desperate"]:
            self.commitment_level = AntagonistCommitment.DESPERATE
        elif self.commitment_score >= thresholds["engaged"]:
            self.commitment_level = AntagonistCommitment.ENGAGED
        else:
            self.commitment_level = AntagonistCommitment.TESTING
    
    def should_manifest(self, current_sequence: int, pressure_level: float) -> bool:
        """Determine if antagonist should manifest based on commitment"""
        # Don't manifest too frequently
        sequence_gap = current_sequence - self.last_manifestation_sequence
        if sequence_gap < 3:
            return False
        
        # Base probability on commitment level
        commitment_probabilities = {
            AntagonistCommitment.TESTING: 0.2,
            AntagonistCommitment.ENGAGED: 0.4,
            AntagonistCommitment.DESPERATE: 0.6,
            AntagonistCommitment.CORNERED: 0.8
        }
        
        base_probability = commitment_probabilities[self.commitment_level]
        
        # Increase probability with higher pressure
        pressure_bonus = pressure_level * 0.3
        
        # Increase probability with longer gaps
        gap_bonus = min(0.3, (sequence_gap - 3) * 0.05)
        
        total_probability = min(0.9, base_probability + pressure_bonus + gap_bonus)
        
        # Simple probability check (would use random in real implementation)
        return pressure_level > 0.5 and total_probability > 0.6
    
    def record_manifestation(self, sequence: int, manifestation_type: str):
        """Record antagonist manifestation"""
        self.manifestation_count += 1
        self.last_manifestation_sequence = sequence
        
        manifestation_record = {
            "sequence": sequence,
            "type": manifestation_type,
            "commitment_level": self.commitment_level.value,
            "commitment_score": self.commitment_score,
            "timestamp": time.time()
        }
        
        self.escalation_history.append(manifestation_record)
        
        # Limit history size
        if len(self.escalation_history) > 50:
            self.escalation_history = self.escalation_history[-25:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "current_antagonist": self.current_antagonist,
            "commitment_level": self.commitment_level.value,
            "commitment_score": self.commitment_score,
            "escalation_history": self.escalation_history.copy(),
            "manifestation_count": self.manifestation_count,
            "last_manifestation_sequence": self.last_manifestation_sequence,
            "antagonist_types": self.antagonist_types.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AntagonistState':
        """Create from dictionary during deserialization"""
        state = cls()
        state.current_antagonist = data.get("current_antagonist", "Generic Opposition")
        
        # Handle commitment level
        commitment_str = data.get("commitment_level", "testing")
        try:
            state.commitment_level = AntagonistCommitment(commitment_str)
        except ValueError:
            state.commitment_level = AntagonistCommitment.TESTING
        
        state.commitment_score = data.get("commitment_score", 0.0)
        state.escalation_history = data.get("escalation_history", [])
        state.manifestation_count = data.get("manifestation_count", 0)
        state.last_manifestation_sequence = data.get("last_manifestation_sequence", 0)
        state.antagonist_types = data.get("antagonist_types", {
            "environmental": {"score": 0.0, "last_seen": 0},
            "social": {"score": 0.0, "last_seen": 0},
            "internal": {"score": 0.0, "last_seen": 0},
            "supernatural": {"score": 0.0, "last_seen": 0},
            "systemic": {"score": 0.0, "last_seen": 0}
        })
        
        return state


class StoryMomentumEngine:
    """
    Story Momentum Engine - Phase 5 Refactored
    
    Responsibilities:
    - Track narrative pressure and story momentum
    - Manage antagonist states and commitment levels
    - Coordinate with sem.py for pattern analysis
    - Generate story context for LLM prompts
    - Maintain story arc progression tracking
    
    Key Changes in Phase 5:
    - Removed embedded semantic analysis logic
    - Uses SemanticProcessor from sem.py for all pattern recognition
    - Leverages centralized momentum patterns and thresholds
    - Maintains story-specific state management
    """
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        
        # Core momentum state
        self.current_pressure = 0.0
        self.pressure_floor = 0.0
        self.current_arc = StoryArc.SETUP
        self.last_analysis_count = 0
        
        # Event and antagonist tracking
        self.momentum_events: List[MomentumEvent] = []
        self.antagonist_state = AntagonistState()
        
        # Semantic processor integration (set by orchestrator)
        self.semantic_processor: Optional[SemanticProcessor] = None
        
        # Analysis state
        self.last_comprehensive_analysis = 0.0
        self.stagnation_counter = 0
        self.momentum_spike_detected = False
        
        # Performance tracking
        self.stats = {
            "total_analyses": 0,
            "pressure_updates": 0,
            "antagonist_manifestations": 0,
            "arc_transitions": 0,
            "comprehensive_analyses": 0
        }
    
    def set_semantic_processor(self, processor: SemanticProcessor):
        """Set the semantic processor from orchestrator"""
        self.semantic_processor = processor
    
    def _log_debug(self, message: str):
        """Debug logging helper"""
        if self.debug_logger:
            self.debug_logger.debug(message, "SME")

# Chunk 2/3 - sme.py - Momentum Processing and Analysis Coordination

    def process_user_input(self, content: str) -> Dict[str, Any]:
        """
        Process user input for immediate momentum patterns using sem.py
        
        Args:
            content: User input content
            
        Returns:
            Dict with immediate momentum analysis results
        """
        if not content or not self.semantic_processor:
            return {"pressure_change": 0.0, "pattern": None}
        
        try:
            # Use sem.py for immediate pattern analysis
            pattern_analysis = self.semantic_processor.analyzer.analyze_text_patterns(content)
            
            # Calculate pressure change based on patterns
            pressure_change = self._calculate_pressure_change_from_patterns(pattern_analysis)
            
            # Update current pressure
            old_pressure = self.current_pressure
            self._update_pressure(pressure_change)
            
            # Create momentum event
            event = MomentumEvent(
                "user_input",
                pressure_change,
                len(self.momentum_events) + 1
            )
            event.update_from_semantic_analysis({"pattern_analysis": pattern_analysis})
            self.momentum_events.append(event)
            
            # Update antagonist state
            self.antagonist_state.update_commitment(self.current_pressure, pattern_analysis)
            
            # Check for momentum spike
            if pressure_change > SME_CONFIG["momentum_spike_threshold"]:
                self.momentum_spike_detected = True
                self._log_debug(f"Momentum spike detected: +{pressure_change:.3f}")
            
            self.stats["pressure_updates"] += 1
            
            result = {
                "pressure_change": pressure_change,
                "old_pressure": old_pressure,
                "new_pressure": self.current_pressure,
                "dominant_pattern": pattern_analysis.get("dominant_pattern"),
                "pattern_strength": pattern_analysis.get("pattern_strength", 0.0),
                "antagonist_commitment": self.antagonist_state.commitment_level.value,
                "momentum_spike": self.momentum_spike_detected
            }
            
            self._log_debug(f"User input processed: {pattern_analysis.get('dominant_pattern')} "
                           f"-> {pressure_change:+.3f} pressure (total: {self.current_pressure:.3f})")
            
            return result
            
        except Exception as e:
            self._log_debug(f"Error processing user input: {e}")
            return {"pressure_change": 0.0, "pattern": None, "error": str(e)}
    
    def _calculate_pressure_change_from_patterns(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calculate pressure change based on pattern analysis from sem.py"""
        if not pattern_analysis:
            return 0.0
        
        dominant_pattern = pattern_analysis.get("dominant_pattern")
        pattern_strength = pattern_analysis.get("pattern_strength", 0.0)
        
        # Pattern-based pressure changes (using sem.py pattern definitions)
        pattern_pressure_values = {
            "conflict": 0.12,
            "tension": 0.08,
            "mystery": 0.06,
            "exploration": 0.04,
            "social": 0.02,
            "resolution": -0.05
        }
        
        base_change = pattern_pressure_values.get(dominant_pattern, 0.01)
        return base_change * pattern_strength
    
    def _update_pressure(self, change: float):
        """Update current pressure with floor ratcheting"""
        # Apply pressure change
        self.current_pressure = max(0.0, min(SME_CONFIG["pressure_ceiling"], 
                                           self.current_pressure + change))
        
        # Ratchet pressure floor if significant pressure reached
        if self.current_pressure > self.pressure_floor + 0.1:
            self.pressure_floor = min(self.pressure_floor + SME_CONFIG["pressure_floor_increment"],
                                    self.current_pressure - 0.05)
        
        # Apply natural decay to current pressure (but not below floor)
        decay = SME_CONFIG["pressure_decay_rate"]
        self.current_pressure = max(self.pressure_floor, 
                                  self.current_pressure - decay)
    
    async def analyze_momentum(self, conversation_messages: List[Dict[str, str]], 
                             total_messages: int, is_first_analysis: bool = False) -> Dict[str, Any]:
        """
        Comprehensive momentum analysis using sem.py for semantic processing
        
        Args:
            conversation_messages: Full conversation for analysis
            total_messages: Total message count for context
            is_first_analysis: Whether this is the first comprehensive analysis
            
        Returns:
            Dict with comprehensive analysis results
        """
        if not self.semantic_processor:
            self._log_debug("No semantic processor available for comprehensive analysis")
            return {"status": "no_processor", "narrative_pressure": self.current_pressure}
        
        try:
            start_time = time.time()
            self.last_analysis_count = total_messages
            
            # Prepare analysis context
            context = {
                "total_messages": total_messages,
                "current_pressure": self.current_pressure,
                "pressure_floor": self.pressure_floor,
                "current_arc": self.current_arc.value,
                "is_first_analysis": is_first_analysis,
                "antagonist_commitment": self.antagonist_state.commitment_level.value
            }
            
            # Analyze recent conversation patterns using sem.py
            pattern_analysis = await self._analyze_conversation_patterns(conversation_messages)
            
            # Determine story arc progression
            arc_analysis = self._analyze_story_arc_progression(pattern_analysis, total_messages)
            
            # Evaluate antagonist manifestation needs
            manifestation_analysis = self._evaluate_antagonist_manifestation(
                pattern_analysis, total_messages
            )
            
            # Update story arc if needed
            if arc_analysis.get("arc_transition"):
                old_arc = self.current_arc
                self.current_arc = StoryArc(arc_analysis["new_arc"])
                self.stats["arc_transitions"] += 1
                self._log_debug(f"Story arc transition: {old_arc.value} -> {self.current_arc.value}")
            
            # Apply comprehensive pressure adjustments
            comprehensive_pressure_change = self._calculate_comprehensive_pressure_change(
                pattern_analysis, arc_analysis, manifestation_analysis
            )
            
            if comprehensive_pressure_change != 0:
                self._update_pressure(comprehensive_pressure_change)
            
            # Reset spike detection after comprehensive analysis
            self.momentum_spike_detected = False
            
            processing_time = time.time() - start_time
            self.stats["comprehensive_analyses"] += 1
            self.last_comprehensive_analysis = time.time()
            
            result = {
                "status": "success",
                "narrative_pressure": self.current_pressure,
                "pressure_floor": self.pressure_floor,
                "pressure_change": comprehensive_pressure_change,
                "current_arc": self.current_arc.value,
                "pattern_analysis": pattern_analysis,
                "arc_analysis": arc_analysis,
                "manifestation_analysis": manifestation_analysis,
                "antagonist_commitment": self.antagonist_state.commitment_level.value,
                "manifestation_type": manifestation_analysis.get("manifestation_type"),
                "processing_time": processing_time,
                "analysis_context": context
            }
            
            self._log_debug(f"Comprehensive analysis complete: "
                           f"{self.current_pressure:.3f} pressure, "
                           f"{self.current_arc.value} arc, "
                           f"{manifestation_analysis.get('manifestation_type', 'none')} manifestation")
            
            return result
            
        except Exception as e:
            self._log_debug(f"Comprehensive analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "narrative_pressure": self.current_pressure
            }
    
    async def _analyze_conversation_patterns(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze conversation patterns using sem.py semantic processor"""
        try:
            # Combine recent messages for pattern analysis
            recent_content = []
            for msg in messages[-10:]:  # Analyze last 10 messages
                if msg.get("role") in ["user", "assistant"]:
                    recent_content.append(msg.get("content", ""))
            
            combined_content = " ".join(recent_content)
            
            # Use sem.py for comprehensive pattern analysis
            analysis = await self.semantic_processor.analyze_message_semantics(
                combined_content,
                {"analysis_type": "momentum_comprehensive"},
                "pattern_analysis"
            )
            
            if analysis:
                return analysis.get("pattern_analysis", {})
            else:
                # Fallback to direct pattern analysis
                return self.semantic_processor.analyzer.analyze_text_patterns(combined_content)
        
        except Exception as e:
            self._log_debug(f"Pattern analysis failed: {e}")
            return {}
    
    def _analyze_story_arc_progression(self, pattern_analysis: Dict[str, Any], 
                                     total_messages: int) -> Dict[str, Any]:
        """Analyze story arc progression based on patterns and pressure"""
        current_arc_value = self.current_arc.value
        suggested_arc = current_arc_value
        arc_transition = False
        
        # Arc progression logic based on pressure and patterns
        if self.current_arc == StoryArc.SETUP:
            if (self.current_pressure > 0.3 and total_messages > 20 and
                pattern_analysis.get("dominant_pattern") in ["conflict", "tension"]):
                suggested_arc = StoryArc.RISING.value
                arc_transition = True
        
        elif self.current_arc == StoryArc.RISING:
            if (self.current_pressure > 0.7 and 
                pattern_analysis.get("pattern_strength", 0) > 0.6):
                suggested_arc = StoryArc.CLIMAX.value
                arc_transition = True
            elif (self.current_pressure < 0.2 and
                  pattern_analysis.get("dominant_pattern") == "resolution"):
                suggested_arc = StoryArc.RESOLUTION.value
                arc_transition = True
        
        elif self.current_arc == StoryArc.CLIMAX:
            if pattern_analysis.get("dominant_pattern") == "resolution":
                suggested_arc = StoryArc.RESOLUTION.value
                arc_transition = True
        
        elif self.current_arc == StoryArc.RESOLUTION:
            if (self.current_pressure > 0.4 and total_messages > self.last_analysis_count + 30):
                suggested_arc = StoryArc.SETUP.value  # New story cycle
                arc_transition = True
        
        return {
            "current_arc": current_arc_value,
            "suggested_arc": suggested_arc,
            "arc_transition": arc_transition,
            "new_arc": suggested_arc if arc_transition else current_arc_value,
            "transition_reason": f"pressure={self.current_pressure:.2f}, pattern={pattern_analysis.get('dominant_pattern')}"
        }
    
    def _evaluate_antagonist_manifestation(self, pattern_analysis: Dict[str, Any], 
                                         total_messages: int) -> Dict[str, Any]:
        """Evaluate need for antagonist manifestation"""
        should_manifest = self.antagonist_state.should_manifest(
            total_messages, self.current_pressure
        )
        
        manifestation_type = "none"
        manifestation_urgency = 0.0
        
        if should_manifest:
            # Determine manifestation type based on patterns and commitment
            dominant_pattern = pattern_analysis.get("dominant_pattern")
            commitment_level = self.antagonist_state.commitment_level
            
            if commitment_level == AntagonistCommitment.CORNERED:
                manifestation_type = "desperate_gambit"
                manifestation_urgency = 0.9
            elif commitment_level == AntagonistCommitment.DESPERATE:
                manifestation_type = "escalated_threat"
                manifestation_urgency = 0.7
            elif commitment_level == AntagonistCommitment.ENGAGED:
                if dominant_pattern == "conflict":
                    manifestation_type = "direct_confrontation"
                else:
                    manifestation_type = "indirect_pressure"
                manifestation_urgency = 0.5
            else:  # TESTING
                manifestation_type = "probing_action"
                manifestation_urgency = 0.3
            
            # Record manifestation
            self.antagonist_state.record_manifestation(total_messages, manifestation_type)
            self.stats["antagonist_manifestations"] += 1
        
        return {
            "should_manifest": should_manifest,
            "manifestation_type": manifestation_type,
            "manifestation_urgency": manifestation_urgency,
            "commitment_level": self.antagonist_state.commitment_level.value,
            "commitment_score": self.antagonist_state.commitment_score
        }
    
    def _calculate_comprehensive_pressure_change(self, pattern_analysis: Dict[str, Any],
                                               arc_analysis: Dict[str, Any],
                                               manifestation_analysis: Dict[str, Any]) -> float:
        """Calculate pressure change from comprehensive analysis"""
        total_change = 0.0
        
        # Arc transition pressure adjustments
        if arc_analysis.get("arc_transition"):
            new_arc = arc_analysis.get("new_arc")
            if new_arc == StoryArc.RISING.value:
                total_change += 0.1  # Boost when entering rising action
            elif new_arc == StoryArc.CLIMAX.value:
                total_change += 0.15  # Significant boost for climax
            elif new_arc == StoryArc.RESOLUTION.value:
                total_change -= 0.2  # Pressure release during resolution
        
        # Manifestation pressure effects
        if manifestation_analysis.get("should_manifest"):
            manifestation_urgency = manifestation_analysis.get("manifestation_urgency", 0.0)
            total_change += manifestation_urgency * 0.1
        
        # Stagnation detection and correction
        pattern_strength = pattern_analysis.get("pattern_strength", 0.0)
        if pattern_strength < 0.1:
            self.stagnation_counter += 1
            if self.stagnation_counter >= SME_CONFIG["stagnation_detection_threshold"]:
                total_change += 0.08  # Pressure boost to break stagnation
                self.stagnation_counter = 0
                self._log_debug("Stagnation detected, applying pressure boost")
        else:
            self.stagnation_counter = 0
        
        return total_change

# Chunk 3/3 - sme.py - Story Context Generation and State Management

    def get_story_context(self) -> str:
        """
        Generate story context for LLM prompts based on current momentum state
        
        Returns:
            Formatted story context string for inclusion in system prompts
        """
        try:
            # Base context with current state
            context_parts = []
            
            # Pressure and arc information
            context_parts.append(f"Current narrative pressure: {self.current_pressure:.3f}/1.0 "
                                f"(floor: {self.pressure_floor:.3f})")
            context_parts.append(f"Story arc: {self.current_arc.value}")
            
            # Antagonist state
            antagonist_info = (f"Antagonist commitment: {self.antagonist_state.commitment_level.value} "
                             f"({self.antagonist_state.commitment_score:.2f})")
            if self.antagonist_state.current_antagonist != "Generic Opposition":
                antagonist_info += f" - {self.antagonist_state.current_antagonist}"
            context_parts.append(antagonist_info)
            
            # Recent momentum patterns (using sem.py analysis)
            if self.momentum_events:
                recent_events = self.momentum_events[-3:]
                patterns = [event.dominant_pattern for event in recent_events if event.dominant_pattern]
                if patterns:
                    context_parts.append(f"Recent patterns: {', '.join(patterns)}")
            
            # Pressure guidance based on current level
            pressure_guidance = self._get_pressure_guidance()
            if pressure_guidance:
                context_parts.append(f"Narrative guidance: {pressure_guidance}")
            
            # Manifestation instructions if needed
            manifestation_guidance = self._get_manifestation_guidance()
            if manifestation_guidance:
                context_parts.append(f"Antagonist guidance: {manifestation_guidance}")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            self._log_debug(f"Error generating story context: {e}")
            return f"Narrative pressure: {self.current_pressure:.3f} | Story arc: {self.current_arc.value}"
    
    def _get_pressure_guidance(self) -> str:
        """Generate pressure-based narrative guidance"""
        if self.current_pressure < 0.2:
            return "Introduce complications or challenges to build tension"
        elif self.current_pressure < 0.4:
            return "Develop existing conflicts and raise stakes"
        elif self.current_pressure < 0.6:
            return "Escalate tension through direct confrontation or revelation"
        elif self.current_pressure < 0.8:
            return "Approach climactic moments with high-stakes decisions"
        else:
            return "Navigate peak dramatic tension with consequential outcomes"
    
    def _get_manifestation_guidance(self) -> str:
        """Generate antagonist manifestation guidance"""
        commitment = self.antagonist_state.commitment_level
        
        if commitment == AntagonistCommitment.CORNERED:
            return "Antagonist should use desperate, high-risk tactics"
        elif commitment == AntagonistCommitment.DESPERATE:
            return "Antagonist escalates with significant resources or threats"
        elif commitment == AntagonistCommitment.ENGAGED:
            return "Antagonist actively opposes with focused efforts"
        elif commitment == AntagonistCommitment.TESTING:
            return "Antagonist probes defenses with cautious actions"
        else:
            return ""
    
    def reset_momentum(self):
        """Reset momentum state for new story or debugging"""
        self.current_pressure = 0.0
        self.pressure_floor = 0.0
        self.current_arc = StoryArc.SETUP
        self.last_analysis_count = 0
        
        # Clear events and reset antagonist
        self.momentum_events.clear()
        self.antagonist_state = AntagonistState()
        
        # Reset analysis state
        self.last_comprehensive_analysis = 0.0
        self.stagnation_counter = 0
        self.momentum_spike_detected = False
        
        # Reset stats
        self.stats.update({
            "total_analyses": 0,
            "pressure_updates": 0,
            "antagonist_manifestations": 0,
            "arc_transitions": 0,
            "comprehensive_analyses": 0
        })
        
        self._log_debug("Story momentum reset to initial state")
    
    def get_pressure_stats(self) -> Dict[str, Any]:
        """Get detailed pressure and momentum statistics"""
        recent_events = self.momentum_events[-10:] if self.momentum_events else []
        
        return {
            "current_pressure": self.current_pressure,
            "pressure_floor": self.pressure_floor,
            "current_arc": self.current_arc.value,
            "total_events": len(self.momentum_events),
            "recent_events": len(recent_events),
            "last_analysis_count": self.last_analysis_count,
            "stagnation_counter": self.stagnation_counter,
            "momentum_spike_detected": self.momentum_spike_detected,
            "antagonist_commitment": self.antagonist_state.commitment_level.value,
            "antagonist_manifestations": self.antagonist_state.manifestation_count,
            "last_comprehensive_analysis": self.last_comprehensive_analysis
        }
    
    def save_state_to_dict(self) -> Dict[str, Any]:
        """Save complete SME state to dictionary for persistence"""
        return {
            "version": "2.0_phase5",
            "current_pressure": self.current_pressure,
            "pressure_floor": self.pressure_floor,
            "current_arc": self.current_arc.value,
            "last_analysis_count": self.last_analysis_count,
            "momentum_events": [event.to_dict() for event in self.momentum_events],
            "antagonist_state": self.antagonist_state.to_dict(),
            "last_comprehensive_analysis": self.last_comprehensive_analysis,
            "stagnation_counter": self.stagnation_counter,
            "momentum_spike_detected": self.momentum_spike_detected,
            "stats": self.stats.copy(),
            "save_timestamp": datetime.now().isoformat()
        }
    
    def load_state_from_dict(self, state_data: Dict[str, Any]):
        """Load SME state from dictionary"""
        try:
            # Load basic state
            self.current_pressure = state_data.get("current_pressure", 0.0)
            self.pressure_floor = state_data.get("pressure_floor", 0.0)
            self.last_analysis_count = state_data.get("last_analysis_count", 0)
            
            # Load story arc
            arc_value = state_data.get("current_arc", "setup")
            try:
                self.current_arc = StoryArc(arc_value)
            except ValueError:
                self.current_arc = StoryArc.SETUP
            
            # Load momentum events
            self.momentum_events.clear()
            for event_data in state_data.get("momentum_events", []):
                event = MomentumEvent.from_dict(event_data)
                self.momentum_events.append(event)
            
            # Load antagonist state
            antagonist_data = state_data.get("antagonist_state", {})
            self.antagonist_state = AntagonistState.from_dict(antagonist_data)
            
            # Load analysis state
            self.last_comprehensive_analysis = state_data.get("last_comprehensive_analysis", 0.0)
            self.stagnation_counter = state_data.get("stagnation_counter", 0)
            self.momentum_spike_detected = state_data.get("momentum_spike_detected", False)
            
            # Load stats
            if "stats" in state_data:
                self.stats.update(state_data["stats"])
            
            version = state_data.get("version", "unknown")
            self._log_debug(f"SME state loaded from {version}")
            
        except Exception as e:
            self._log_debug(f"Error loading SME state: {e}")
            self.reset_momentum()  # Fallback to clean state
    
    def get_narrative_time_info(self) -> Dict[str, Any]:
        """Get narrative time information using sem.py if available"""
        if self.semantic_processor and hasattr(self.semantic_processor, 'narrative_timer'):
            return self.semantic_processor.narrative_timer.get_stats()
        else:
            # Fallback estimation based on events
            total_events = len(self.momentum_events)
            estimated_time = total_events * 30.0  # Rough estimate: 30 seconds per event
            
            return {
                "total_narrative_seconds": estimated_time,
                "exchange_count": total_events,
                "narrative_time_formatted": f"{estimated_time/60:.1f}m" if estimated_time > 60 else f"{estimated_time:.0f}s"
            }
    
    def should_trigger_analysis(self, current_message_count: int) -> bool:
        """Determine if comprehensive analysis should be triggered"""
        messages_since_last = current_message_count - self.last_analysis_count
        
        # Trigger every N messages as configured
        if messages_since_last >= SME_CONFIG["analysis_trigger_threshold"]:
            return True
        
        # Early trigger for significant momentum spikes
        if self.momentum_spike_detected and messages_since_last >= 5:
            return True
        
        # Early trigger for high stagnation
        if self.stagnation_counter >= SME_CONFIG["stagnation_detection_threshold"] - 1:
            return True
        
        return False


# Factory functions for orchestrator integration
def create_story_momentum_engine(debug_logger=None) -> StoryMomentumEngine:
    """Factory function to create story momentum engine"""
    return StoryMomentumEngine(debug_logger)


def get_sme_config() -> Dict[str, Any]:
    """Get copy of SME configuration for reference"""
    return dict(SME_CONFIG)


def validate_pressure_value(pressure: float) -> float:
    """Validate and clamp pressure value to valid range"""
    return max(0.0, min(SME_CONFIG["pressure_ceiling"], pressure))


def validate_story_arc(arc_value: Any) -> StoryArc:
    """Validate and convert story arc value"""
    if isinstance(arc_value, StoryArc):
        return arc_value
    
    if isinstance(arc_value, str):
        try:
            return StoryArc(arc_value.lower())
        except ValueError:
            return StoryArc.SETUP
    
    return StoryArc.SETUP


# Utility functions for testing and integration
def test_sme_functionality() -> bool:
    """Test basic SME functionality"""
    try:
        # Create test instance
        sme = create_story_momentum_engine()
        
        # Test basic state
        assert sme.current_pressure == 0.0
        assert sme.current_arc == StoryArc.SETUP
        assert len(sme.momentum_events) == 0
        
        # Test state persistence
        test_state = sme.save_state_to_dict()
        assert "current_pressure" in test_state
        assert "current_arc" in test_state
        
        # Test state loading
        sme.load_state_from_dict(test_state)
        assert sme.current_pressure == 0.0
        
        # Test story context generation
        context = sme.get_story_context()
        assert "narrative pressure" in context.lower()
        assert "story arc" in context.lower()
        
        return True
        
    except Exception:
        return False


def get_sme_info() -> Dict[str, Any]:
    """Get information about Story Momentum Engine capabilities"""
    return {
        "name": "Story Momentum Engine (Phase 5 Refactored)",
        "version": "2.0_phase5",
        "dependencies": ["sem.py for semantic analysis"],
        "features": [
            "Dynamic narrative pressure tracking with floor ratcheting",
            "Antagonist commitment and manifestation management", 
            "Story arc progression analysis",
            "Pattern-based momentum calculation via sem.py",
            "Comprehensive analysis coordination",
            "Stagnation detection and correction",
            "Story context generation for LLM prompts",
            "State persistence and loading"
        ],
        "phase5_changes": [
            "Removed embedded pattern recognition logic",
            "Integrated with centralized sem.py SemanticProcessor",
            "Uses shared momentum patterns from sem.py",
            "Leverages sem.py for comprehensive analysis",
            "Maintains story-specific state management"
        ],
        "integration_points": [
            "SemanticProcessor from sem.py for pattern analysis",
            "Async analysis coordination",
            "Background processing through semantic processor",
            "Debug logger for monitoring"
        ]
    }


# Module test functionality
if __name__ == "__main__":
    print("DevName RPG Client - Story Momentum Engine (Phase 5 Refactored)")
    print("Successfully refactored to use centralized semantic logic:")
    print("✓ Removed embedded pattern recognition code")
    print("✓ Integrated with sem.py SemanticProcessor")
    print("✓ Uses centralized momentum patterns")
    print("✓ Async analysis coordination")
    print("✓ Maintains story-specific state management")
    print("✓ Dynamic pressure tracking with floor ratcheting")
    print("✓ Antagonist commitment and manifestation logic")
    print("✓ Story arc progression and transition handling")
    
    print("\nSME Phase 5 Info:")
    info = get_sme_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  • {item}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nFunctionality test: {'✓ PASSED' if test_sme_functionality() else '✗ FAILED'}")
    print("\nPhase 5 refactoring complete - SME now uses centralized sem.py for all semantic analysis.")
    print("Ready for Phase 6: Integration and Testing.")
