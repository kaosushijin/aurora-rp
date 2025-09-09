#!/usr/bin/env python3
"""
Aurora RPG Client - Story Momentum Engine Module (sme_nc5.py) - Chunk 1/2

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

4. emm_nc5.py: Enhanced Memory Manager Module
   - Conversation history storage with semantic condensation
   - Token estimation and memory optimization
   - Called by nci_nc5.py for message storage/retrieval
   - Provides conversation context to mcp_nc5.py

5. sme_nc5.py (THIS FILE): Story Momentum Engine Module
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
- Dynamic pressure level calculation based on conversation flow
- Antagonist selection and management for story arcs
- Narrative pacing and story momentum tracking
- Context enhancement for MCP requests
- Story arc progression and climax management
- Real-time analysis of user input and assistant responses
"""

import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class StoryArc(Enum):
    """Story progression phases"""
    BEGINNING = "beginning"
    RISING_ACTION = "rising_action" 
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"

class PressureLevel(Enum):
    """Narrative pressure classifications"""
    CALM = "calm"
    BUILDING = "building"
    TENSE = "tense"
    CRITICAL = "critical"
    CLIMACTIC = "climactic"

@dataclass
class Antagonist:
    """Antagonist character definition"""
    name: str
    description: str
    motivation: str
    power_level: float
    introduction_threshold: float
    active: bool = False

class StoryMomentumEngine:
    """
    Dynamic Story Momentum Engine for narrative pressure management.
    
    This system analyzes conversation patterns to maintain optimal story pacing
    and introduces narrative elements to enhance the RPG experience.
    """
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.active = False
        
        # Core SME state
        self.pressure_level = 0.0  # 0.0 to 1.0
        self.last_update = datetime.now()
        self.story_arc = StoryArc.BEGINNING
        self.current_antagonist = None
        self.session_start = datetime.now()
        
        # Tracking variables
        self.user_input_count = 0
        self.assistant_response_count = 0
        self.last_pressure_change = 0.0
        self.pressure_history = []
        self.narrative_events = []
        
        # Define available antagonists
        self.antagonists = [
            Antagonist(
                name="The Shadow Weaver",
                description="A mysterious figure who manipulates darkness and illusion",
                motivation="Seeks to unravel the fabric of reality itself",
                power_level=0.7,
                introduction_threshold=0.3
            ),
            Antagonist(
                name="Lord Malachar",
                description="An ancient lich with dominion over the undead",
                motivation="Desires to merge the realm of the living with the dead",
                power_level=0.9,
                introduction_threshold=0.5
            ),
            Antagonist(
                name="The Void Caller",
                description="A cosmic entity that devours worlds and memories",
                motivation="Hungers for the complete erasure of existence",
                power_level=1.0,
                introduction_threshold=0.7
            ),
            Antagonist(
                name="Corrupted Nature Spirit",
                description="A once-benevolent forest guardian twisted by dark magic",
                motivation="Seeks to reclaim the natural world through destruction",
                power_level=0.6,
                introduction_threshold=0.2
            ),
            Antagonist(
                name="The Mind Flenser",
                description="A psychic predator that feeds on thoughts and dreams",
                motivation="Wishes to absorb all consciousness into itself",
                power_level=0.8,
                introduction_threshold=0.4
            )
        ]
        
        if self.debug_logger:
            self.debug_logger.debug("Story Momentum Engine initialized", "SME")
    
    def activate(self):
        """Activate the Story Momentum Engine"""
        self.active = True
        self.session_start = datetime.now()
        
        if self.debug_logger:
            self.debug_logger.debug("SME activated", "SME")
    
    def deactivate(self):
        """Deactivate the Story Momentum Engine"""
        self.active = False
        
        if self.debug_logger:
            self.debug_logger.debug("SME deactivated", "SME")
    
    def reset(self):
        """Reset SME state to beginning"""
        self.pressure_level = 0.0
        self.story_arc = StoryArc.BEGINNING
        self.current_antagonist = None
        self.user_input_count = 0
        self.assistant_response_count = 0
        self.pressure_history.clear()
        self.narrative_events.clear()
        self.session_start = datetime.now()
        
        # Reset antagonist states
        for antagonist in self.antagonists:
            antagonist.active = False
        
        if self.debug_logger:
            self.debug_logger.debug("SME state reset", "SME")
    
    def process_user_input(self, user_input: str):
        """Process user input for story momentum analysis"""
        if not self.active:
            return
        
        self.user_input_count += 1
        
        # Analyze input for story elements
        pressure_delta = self._analyze_user_input(user_input)
        self._update_pressure(pressure_delta)
        
        # Check for antagonist introduction
        self._check_antagonist_introduction()
        
        # Update story arc based on pressure and time
        self._update_story_arc()
        
        if self.debug_logger:
            self.debug_logger.debug(f"User input processed: pressure={self.pressure_level:.3f}", "SME")
    
    def process_assistant_response(self, response: str):
        """Process assistant response for story momentum analysis"""
        if not self.active:
            return
        
        self.assistant_response_count += 1
        
        # Analyze response for narrative elements
        pressure_delta = self._analyze_assistant_response(response)
        self._update_pressure(pressure_delta)
        
        if self.debug_logger:
            self.debug_logger.debug(f"Assistant response processed: pressure={self.pressure_level:.3f}", "SME")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive SME status"""
        return {
            "active": self.active,
            "pressure_level": self.pressure_level,
            "pressure_name": self._get_pressure_name(),
            "story_arc": self.story_arc.value,
            "antagonist_name": self.current_antagonist.name if self.current_antagonist else "None",
            "user_inputs": self.user_input_count,
            "assistant_responses": self.assistant_response_count,
            "session_duration": str(datetime.now() - self.session_start).split('.')[0],
            "last_update": self.last_update.strftime("%H:%M:%S"),
            "narrative_events": len(self.narrative_events)
        }
    
    def get_context_for_mcp(self) -> Dict[str, Any]:
        """Get SME context for enhanced MCP requests"""
        if not self.active:
            return {}
        
        return {
            "pressure_level": self.pressure_level,
            "pressure_name": self._get_pressure_name(),
            "story_arc": self.story_arc.value,
            "antagonist_name": self.current_antagonist.name if self.current_antagonist else "None",
            "antagonist_description": self.current_antagonist.description if self.current_antagonist else "",
            "recent_pressure_trend": self._get_pressure_trend()
        }
    
    def _analyze_user_input(self, user_input: str) -> float:
        """
        Analyze user input for story momentum impact.
        
        This analyzes the semantic content of user input to determine
        how it should affect the narrative pressure.
        """
        pressure_delta = 0.0
        input_lower = user_input.lower()
        
        # Action keywords increase pressure
        action_keywords = [
            "attack", "fight", "battle", "kill", "destroy", "break", "smash",
            "run", "flee", "escape", "hide", "sneak", "steal", "take",
            "explore", "investigate", "search", "look", "examine",
            "cast", "spell", "magic", "summon", "invoke"
        ]
        
        action_count = sum(1 for keyword in action_keywords if keyword in input_lower)
        pressure_delta += action_count * 0.02
        
        # Danger keywords increase pressure more
        danger_keywords = [
            "danger", "threat", "enemy", "monster", "death", "dark", "evil",
            "shadow", "blood", "wound", "pain", "fear", "terror", "nightmare"
        ]
        
        danger_count = sum(1 for keyword in danger_keywords if keyword in input_lower)
        pressure_delta += danger_count * 0.05
        
        # Social keywords decrease pressure
        social_keywords = [
            "talk", "speak", "conversation", "friend", "help", "assist",
            "peaceful", "calm", "rest", "sleep", "eat", "drink", "relax"
        ]
        
        social_count = sum(1 for keyword in social_keywords if keyword in input_lower)
        pressure_delta -= social_count * 0.01
        
        # Question marks suggest uncertainty, slight pressure increase
        if "?" in user_input:
            pressure_delta += 0.005
        
        # Exclamation marks suggest intensity
        exclamation_count = user_input.count("!")
        pressure_delta += exclamation_count * 0.01
        
        # Long inputs suggest engagement, slight pressure increase
        if len(user_input) > 100:
            pressure_delta += 0.01
        
        return pressure_delta
    
    def _analyze_assistant_response(self, response: str) -> float:
        """
        Analyze assistant response for story momentum impact.
        
        Responses can maintain, increase, or decrease pressure based on content.
        """
        pressure_delta = 0.0
        response_lower = response.lower()
        
        # Conflict description increases pressure
        conflict_keywords = [
            "battle", "fight", "attack", "enemy", "danger", "threat",
            "darkness", "shadow", "evil", "monster", "death", "blood"
        ]
        
        conflict_count = sum(1 for keyword in conflict_keywords if keyword in response_lower)
        pressure_delta += conflict_count * 0.01
        
        # Peaceful descriptions decrease pressure
        peace_keywords = [
            "peaceful", "calm", "serene", "quiet", "gentle", "warm",
            "safe", "comfort", "rest", "beauty", "light", "hope"
        ]
        
        peace_count = sum(1 for keyword in peace_keywords if keyword in response_lower)
        pressure_delta -= peace_count * 0.01
        
        # Mysterious elements maintain tension
        mystery_keywords = [
            "mysterious", "strange", "unknown", "hidden", "secret",
            "whisper", "shadow", "ancient", "forgotten", "lost"
        ]
        
        mystery_count = sum(1 for keyword in mystery_keywords if keyword in response_lower)
        pressure_delta += mystery_count * 0.005
        
        return pressure_delta

# sme_nc5.py - Chunk 2/2
# Pressure Management and Antagonist System

    def _update_pressure(self, delta: float):
        """Update pressure level with bounds checking and history tracking"""
        old_pressure = self.pressure_level
        
        # Apply natural decay over time
        time_since_last = (datetime.now() - self.last_update).total_seconds()
        natural_decay = time_since_last * 0.0001  # Very slow natural decay
        
        # Update pressure with delta and decay
        self.pressure_level = max(0.0, min(1.0, self.pressure_level + delta - natural_decay))
        
        # Track pressure changes
        if abs(self.pressure_level - old_pressure) > 0.001:
            self.pressure_history.append({
                "timestamp": datetime.now().isoformat(),
                "old_pressure": old_pressure,
                "new_pressure": self.pressure_level,
                "delta": delta
            })
            
            # Keep only recent history (last 20 changes)
            if len(self.pressure_history) > 20:
                self.pressure_history.pop(0)
        
        self.last_update = datetime.now()
        self.last_pressure_change = delta
    
    def _check_antagonist_introduction(self):
        """Check if an antagonist should be introduced based on pressure level"""
        if self.current_antagonist:
            return  # Already have an antagonist
        
        # Find suitable antagonists for current pressure level
        suitable_antagonists = [
            ant for ant in self.antagonists 
            if not ant.active and ant.introduction_threshold <= self.pressure_level
        ]
        
        if suitable_antagonists:
            # Choose antagonist based on pressure level and randomness
            weights = [ant.introduction_threshold for ant in suitable_antagonists]
            chosen_antagonist = random.choices(suitable_antagonists, weights=weights)[0]
            
            self.current_antagonist = chosen_antagonist
            chosen_antagonist.active = True
            
            # Record narrative event
            self.narrative_events.append({
                "timestamp": datetime.now().isoformat(),
                "event": "antagonist_introduction",
                "antagonist": chosen_antagonist.name,
                "pressure_level": self.pressure_level
            })
            
            if self.debug_logger:
                self.debug_logger.debug(f"Antagonist introduced: {chosen_antagonist.name}", "SME")
    
    def _update_story_arc(self):
        """Update story arc based on pressure level and session duration"""
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60  # minutes
        
        # Story arc progression based on pressure and time
        if self.pressure_level < 0.2:
            new_arc = StoryArc.BEGINNING
        elif self.pressure_level < 0.4:
            new_arc = StoryArc.RISING_ACTION
        elif self.pressure_level < 0.7:
            new_arc = StoryArc.CLIMAX
        elif self.pressure_level < 0.9:
            new_arc = StoryArc.FALLING_ACTION
        else:
            new_arc = StoryArc.RESOLUTION
        
        # Consider session duration for natural progression
        if session_duration > 30:  # After 30 minutes, tend toward resolution
            if self.story_arc == StoryArc.CLIMAX and self.pressure_level < 0.6:
                new_arc = StoryArc.FALLING_ACTION
            elif self.story_arc == StoryArc.FALLING_ACTION and self.pressure_level < 0.4:
                new_arc = StoryArc.RESOLUTION
        
        if new_arc != self.story_arc:
            old_arc = self.story_arc
            self.story_arc = new_arc
            
            # Record narrative event
            self.narrative_events.append({
                "timestamp": datetime.now().isoformat(),
                "event": "story_arc_change",
                "old_arc": old_arc.value,
                "new_arc": new_arc.value,
                "pressure_level": self.pressure_level
            })
            
            if self.debug_logger:
                self.debug_logger.debug(f"Story arc changed: {old_arc.value} -> {new_arc.value}", "SME")
    
    def _get_pressure_name(self) -> str:
        """Get descriptive name for current pressure level"""
        if self.pressure_level < 0.2:
            return "Calm"
        elif self.pressure_level < 0.4:
            return "Building"
        elif self.pressure_level < 0.6:
            return "Tense"
        elif self.pressure_level < 0.8:
            return "Critical"
        else:
            return "Climactic"
    
    def _get_pressure_trend(self) -> str:
        """Get recent pressure trend direction"""
        if len(self.pressure_history) < 3:
            return "stable"
        
        recent_changes = [entry["delta"] for entry in self.pressure_history[-3:]]
        avg_change = sum(recent_changes) / len(recent_changes)
        
        if avg_change > 0.01:
            return "rising"
        elif avg_change < -0.01:
            return "falling"
        else:
            return "stable"
    
    def get_antagonist_list(self) -> List[Dict[str, Any]]:
        """Get list of all available antagonists"""
        return [
            {
                "name": ant.name,
                "description": ant.description,
                "motivation": ant.motivation,
                "power_level": ant.power_level,
                "introduction_threshold": ant.introduction_threshold,
                "active": ant.active
            }
            for ant in self.antagonists
        ]
    
    def force_antagonist_introduction(self, antagonist_name: str) -> bool:
        """Force introduction of specific antagonist (for testing/debugging)"""
        for antagonist in self.antagonists:
            if antagonist.name == antagonist_name and not antagonist.active:
                if self.current_antagonist:
                    self.current_antagonist.active = False
                
                self.current_antagonist = antagonist
                antagonist.active = True
                
                self.narrative_events.append({
                    "timestamp": datetime.now().isoformat(),
                    "event": "forced_antagonist_introduction",
                    "antagonist": antagonist.name,
                    "pressure_level": self.pressure_level
                })
                
                if self.debug_logger:
                    self.debug_logger.debug(f"Forced antagonist introduction: {antagonist.name}", "SME")
                
                return True
        
        return False
    
    def export_sme_data(self) -> Dict[str, Any]:
        """Export comprehensive SME data for analysis"""
        return {
            "timestamp": datetime.now().isoformat(),
            "session_start": self.session_start.isoformat(),
            "status": self.get_status(),
            "pressure_history": self.pressure_history.copy(),
            "narrative_events": self.narrative_events.copy(),
            "antagonist_list": self.get_antagonist_list(),
            "context": self.get_context_for_mcp()
        }
    
    def get_debug_content(self) -> List[str]:
        """Get debug information about SME state"""
        status = self.get_status()
        
        debug_lines = [
            "Story Momentum Engine Debug Information",
            "=" * 50,
            f"Active: {status['active']}",
            f"Pressure Level: {status['pressure_level']:.3f} ({status['pressure_name']})",
            f"Story Arc: {status['story_arc']}",
            f"Current Antagonist: {status['antagonist_name']}",
            f"Session Duration: {status['session_duration']}",
            f"User Inputs: {status['user_inputs']}",
            f"Assistant Responses: {status['assistant_responses']}",
            f"Narrative Events: {status['narrative_events']}",
            f"Last Update: {status['last_update']}",
            f"Pressure Trend: {self._get_pressure_trend()}",
            "",
            "Available Antagonists:",
        ]
        
        for ant in self.antagonists:
            status_str = "ACTIVE" if ant.active else f"threshold: {ant.introduction_threshold:.1f}"
            debug_lines.append(f"  {ant.name}: {status_str}")
        
        debug_lines.extend([
            "",
            "Recent Pressure History:",
        ])
        
        recent_history = self.pressure_history[-5:] if len(self.pressure_history) >= 5 else self.pressure_history
        for entry in recent_history:
            timestamp = entry["timestamp"].split("T")[1][:8]  # Extract time portion
            debug_lines.append(f"  [{timestamp}] {entry['old_pressure']:.3f} -> {entry['new_pressure']:.3f} (Δ{entry['delta']:+.3f})")
        
        debug_lines.extend([
            "",
            "Recent Narrative Events:",
        ])
        
        recent_events = self.narrative_events[-3:] if len(self.narrative_events) >= 3 else self.narrative_events
        for event in recent_events:
            timestamp = event["timestamp"].split("T")[1][:8]
            debug_lines.append(f"  [{timestamp}] {event['event']}: {event.get('antagonist', event.get('new_arc', 'N/A'))}")
        
        return debug_lines

# Utility functions for SME analysis
def analyze_conversation_pressure(messages: List[str]) -> float:
    """Analyze overall pressure level from conversation messages"""
    if not messages:
        return 0.0
    
    total_pressure = 0.0
    
    for message in messages:
        message_lower = message.lower()
        
        # Count action and danger keywords
        pressure_keywords = [
            "attack", "fight", "danger", "threat", "enemy", "monster",
            "death", "dark", "evil", "shadow", "blood", "fear"
        ]
        
        keyword_count = sum(1 for keyword in pressure_keywords if keyword in message_lower)
        total_pressure += keyword_count * 0.1
    
    # Average pressure across messages
    return min(1.0, total_pressure / len(messages))

def create_sme_report(sme: StoryMomentumEngine) -> str:
    """Create a human-readable SME status report"""
    status = sme.get_status()
    
    report_lines = [
        "Story Momentum Engine Report",
        "=" * 30,
        f"Status: {'Active' if status['active'] else 'Inactive'}",
        f"Pressure Level: {status['pressure_level']:.3f} ({status['pressure_name']})",
        f"Story Arc: {status['story_arc'].title().replace('_', ' ')}",
        f"Current Antagonist: {status['antagonist_name']}",
        f"Session Duration: {status['session_duration']}",
        f"User Inputs: {status['user_inputs']}",
        f"Assistant Responses: {status['assistant_responses']}",
        f"Narrative Events: {status['narrative_events']}",
        f"Last Update: {status['last_update']}"
    ]
    
    return "\n".join(report_lines)

def simulate_pressure_scenario(sme: StoryMomentumEngine, scenario: str) -> Dict[str, Any]:
    """Simulate different pressure scenarios for testing"""
    initial_pressure = sme.pressure_level
    
    scenarios = {
        "combat": ["I attack the orc!", "I cast fireball!", "The enemy strikes back!"],
        "exploration": ["I examine the room", "I look around carefully", "I search for clues"],
        "social": ["I talk to the merchant", "I negotiate peacefully", "I make friends"],
        "mystery": ["Something seems strange", "I investigate the whispers", "The shadows move oddly"]
    }
    
    if scenario not in scenarios:
        return {"error": f"Unknown scenario: {scenario}"}
    
    # Process scenario inputs
    for input_text in scenarios[scenario]:
        sme.process_user_input(input_text)
    
    final_pressure = sme.pressure_level
    pressure_change = final_pressure - initial_pressure
    
    return {
        "scenario": scenario,
        "initial_pressure": initial_pressure,
        "final_pressure": final_pressure,
        "pressure_change": pressure_change,
        "current_arc": sme.story_arc.value,
        "antagonist": sme.current_antagonist.name if sme.current_antagonist else "None"
    }

# Performance monitoring for SME
class SMEPerformanceMonitor:
    """Monitor SME performance and effectiveness"""
    
    def __init__(self, debug_logger=None):
        self.debug_logger = debug_logger
        self.pressure_updates = 0
        self.antagonist_introductions = 0
        self.arc_changes = 0
        self.analysis_times = []
    
    def record_analysis(self, analysis_time: float):
        """Record analysis performance"""
        self.analysis_times.append(analysis_time)
        if len(self.analysis_times) > 100:
            self.analysis_times.pop(0)
    
    def record_pressure_update(self):
        """Record pressure update"""
        self.pressure_updates += 1
    
    def record_antagonist_introduction(self):
        """Record antagonist introduction"""
        self.antagonist_introductions += 1
    
    def record_arc_change(self):
        """Record story arc change"""
        self.arc_changes += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_analysis_time = sum(self.analysis_times) / len(self.analysis_times) if self.analysis_times else 0.0
        
        return {
            "pressure_updates": self.pressure_updates,
            "antagonist_introductions": self.antagonist_introductions,
            "arc_changes": self.arc_changes,
            "avg_analysis_time": avg_analysis_time,
            "total_analyses": len(self.analysis_times)
        }

# Enhanced SME with performance monitoring
class EnhancedStoryMomentumEngine(StoryMomentumEngine):
    """SME with built-in performance monitoring"""
    
    def __init__(self, debug_logger=None):
        super().__init__(debug_logger)
        self.performance_monitor = SMEPerformanceMonitor(debug_logger)
    
    def process_user_input(self, user_input: str):
        """Process user input with performance monitoring"""
        start_time = time.time()
        super().process_user_input(user_input)
        analysis_time = time.time() - start_time
        
        self.performance_monitor.record_analysis(analysis_time)
        self.performance_monitor.record_pressure_update()
    
    def process_assistant_response(self, response: str):
        """Process assistant response with performance monitoring"""
        start_time = time.time()
        super().process_assistant_response(response)
        analysis_time = time.time() - start_time
        
        self.performance_monitor.record_analysis(analysis_time)
        self.performance_monitor.record_pressure_update()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_monitor.get_performance_stats()

# Module test when run directly
if __name__ == "__main__":
    print("Aurora RPG Client - Story Momentum Engine Module")
    print("Testing SME functionality...")
    
    # Create test SME
    sme = StoryMomentumEngine()
    sme.activate()
    
    # Test input processing
    sme.process_user_input("I draw my sword and attack the monster!")
    sme.process_assistant_response("The creature roars in fury and lunges at you!")
    
    # Test status
    status = sme.get_status()
    print(f"Pressure level: {status['pressure_level']:.3f}")
    print(f"Story arc: {status['story_arc']}")
    print(f"Antagonist: {status['antagonist_name']}")
    
    # Test MCP context
    context = sme.get_context_for_mcp()
    print(f"MCP context: {len(context)} elements")
    
    # Test scenario simulation
    scenario_result = simulate_pressure_scenario(sme, "combat")
    print(f"Combat scenario result: {scenario_result['pressure_change']:.3f} pressure change")
    
    print("Story Momentum Engine module test completed.")

# End of sme_nc5.py - Aurora RPG Client Story Momentum Engine Module
