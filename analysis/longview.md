# Longview.md - Comprehensive Logic Flow and Token Budget Strategy

## Executive Summary

This document outlines the complete logic flow from user input through validation, memory storage, semantic analysis, condensation, momentum tracking, and antagonist generation. It identifies critical token budget vulnerabilities and proposes a centralized Token Budget Manager to maintain our 32K context window limit while preserving narrative quality.

---

## PART 1: COMPLETE LOGIC FLOW ARCHITECTURE

### 1. INPUT → VALIDATION FLOW

```
User Input (ncui.py)
    ↓
Orchestrator Callback
    ↓
Semantic Validation (sem.py::validate_input)
    ├─ Length Check (≤4000 chars)
    ├─ Token Estimation (~1000 tokens)
    ├─ Category Classification
    │   ├─ command (starts with /)
    │   ├─ action (I/me statements)
    │   ├─ dialogue ("quoted text")
    │   ├─ query (questions)
    │   └─ narrative (default)
    └─ Confidence Score (0.0-1.0)
```

**Intended Outcome**: Ensure input is valid, categorized, and within token limits before processing.

### 2. VALIDATION → MEMORY FLOW

```
Valid Input
    ↓
EMM Storage (emm.py::add_message)
    ├─ Thread-Safe Addition (RLock)
    ├─ Token Estimation (chars/4)
    ├─ Message ID Generation (UUID)
    ├─ Timestamp Recording
    └─ Background Auto-Save Thread
        └─ Atomic File Write (temp→rename)
```

**Intended Outcome**: Persistent, thread-safe storage with immediate availability and crash recovery.

### 3. MEMORY → SEMANTIC ANALYSIS FLOW

```
Stored Message
    ↓
Background Semantic Categorization (30s cooldown)
    ↓
Orchestrator → sem.py Analysis Request
    ↓
Build Semantic Prompt:
    "Analyze this RPG conversation message for semantic importance:
     Message: {content}
     Provide analysis in JSON format:
     {
         'importance_score': 0.0-1.0,
         'categories': [...]
     }"
    ↓
MCP LLM Call (via orchestrator)
    ↓
5-Strategy Response Parsing:
    1. Direct JSON parse
    2. Regex JSON extraction
    3. Binary preserve/discard
    4. Keyword pattern matching
    5. Default fallback (standard)
    ↓
Update Message Category in EMM
```

**Categories & Preservation Ratios**:
- `story_critical`: 90% preservation
- `character_focused`: 80% preservation
- `relationship_dynamics`: 70% preservation
- `emotional_significance`: 60% preservation
- `world_building`: 50% preservation
- `standard`: 40% preservation

**Intended Outcome**: Every message gets semantic importance rating for intelligent condensation.

### 4. MEMORY → CONDENSATION FLOW

```
Token Limit Check (25,000 tokens)
    ↓
If Exceeded → Trigger Condensation
    ↓
Multi-Pass Condensation (up to 3 passes):
    
Pass 1: Gentle (preserve 60%+)
    ├─ Categorize all messages
    ├─ Group by semantic importance
    └─ Condense lowest-priority groups
    
Pass 2: Moderate (preserve 40%+)
    ├─ Re-evaluate preservation
    ├─ Merge similar messages
    └─ Compress dialogue
    
Pass 3: Aggressive (preserve 20%+)
    ├─ Keep only critical events
    ├─ Summarize entire scenes
    └─ Preserve plot essentials
    ↓
Replace Original Messages with Condensed Summaries
    ↓
Auto-Save Updated Memory
```

**Intended Outcome**: Maintain conversation within token budget while preserving narrative essence.

### 5. INPUT → MOMENTUM ANALYSIS FLOW

```
Every User Input
    ↓
SME Pattern Analysis (sme.py::process_user_input)
    ├─ Duration Detection (time patterns)
    ├─ Activity Classification
    ├─ Keyword Pattern Matching:
    │   ├─ exploration_patterns
    │   ├─ tension_patterns
    │   ├─ conflict_patterns
    │   └─ resolution_patterns
    └─ Pressure Calculation
        ├─ Current Level (0.0-1.0)
        ├─ Decay Over Time (-0.01/min)
        └─ Floor Ratcheting (prevents regression)
    
Every 15 Messages → Background LLM Analysis
    ↓
Build Momentum Prompt:
    "Analyze narrative pressure (0.0-1.0)
     Current Arc: {SETUP|RISING|CLIMAX|RESOLUTION}
     Recent Events: {detected_patterns}
     Determine pressure changes and manifestation type"
    ↓
Update Story Arc Transitions:
    SETUP (0.0-0.3) → RISING (0.3-0.7) → CLIMAX (0.7-1.0) → RESOLUTION
```

**Intended Outcome**: Dynamic story pacing that responds to player actions and prevents stagnation.

### 6. MOMENTUM → ANTAGONIST GENERATION FLOW

```
Pressure Threshold Exceeded (>0.5)
    ↓
Check Current Antagonist Status
    ├─ None → Generate New
    ├─ Inactive → Reactivate
    └─ Active → Escalate Commitment
    
Antagonist Generation (simplified in current):
    Basic Threshold Detection
    └─ Create "Unknown Threat" placeholder
    
Antagonist Generation (legacy/intended):
    ↓
Build Generation Prompt:
    "Generate antagonist for pressure level {0.0-1.0}
     Story Arc: {current_arc}
     Create: name, motivation, threat_level, introduction"
    ↓
Parse Response → Create Antagonist Object
    ↓
Track Commitment Levels:
    testing → engaged → desperate → cornered
```

**Intended Outcome**: Dynamic opposition that scales with narrative tension and player behavior.

---

## PART 2: CRITICAL TOKEN BUDGET ISSUES

### Current Token Overflow Scenarios

1. **Uncontrolled Conversation History**
   - MCP requests "last 10 messages" without token counting
   - 10 long messages could be 15,000+ tokens
   - No truncation based on actual token usage

2. **Unbounded Story Context**
   - SME generates context without size limits
   - Complex world state could exceed 5,000 tokens
   - Added to every LLM request

3. **Concurrent Request Collision**
   - Main conversation: 20K tokens
   - Background semantic analysis: 6K tokens
   - Background momentum analysis: 6K tokens
   - Total: 32K+ tokens = OVERFLOW

4. **No Central Coordination**
   - EMM tracks 25K limit independently
   - MCP builds requests without checking EMM's budget
   - SME adds context without knowing remaining space

---

## PART 3: PROPOSED TOKEN BUDGET MANAGER

### Centralized Token Budget Architecture

```python
class TokenBudgetManager:
    """
    Central authority for all token allocation decisions.
    Enforces hard limits and provides dynamic allocation.
    """
    
    # Fixed Allocations (Reserved)
    CONTEXT_WINDOW = 32000
    SYSTEM_PROMPTS = 4000      # Reduced from 5000
    USER_INPUT_MAX = 2000       # Hard enforced
    LLM_RESPONSE_RESERVE = 5000 # Space for response
    SAFETY_BUFFER = 1500        # Overflow protection
    
    # Dynamic Pool (Available for conversation + context)
    DYNAMIC_POOL = CONTEXT_WINDOW - (
        SYSTEM_PROMPTS + USER_INPUT_MAX + 
        LLM_RESPONSE_RESERVE + SAFETY_BUFFER
    )  # = 19,500 tokens
    
    def allocate_for_request(self, request_type: str) -> Dict[str, int]:
        """
        Allocate tokens based on request type and current state
        """
        if request_type == "main_conversation":
            return {
                "system_prompts": 4000,
                "story_context": min(1500, self.get_available()),
                "conversation": min(14000, self.get_available()),
                "user_input": 2000,
                "reserved_response": 5000
            }
        elif request_type == "semantic_analysis":
            return {
                "analysis_prompt": 500,
                "message_content": 1000,
                "context_samples": 2000,
                "reserved_response": 1000
            }
        elif request_type == "momentum_analysis":
            return {
                "analysis_prompt": 1000,
                "recent_messages": 4000,
                "story_state": 500,
                "reserved_response": 1500
            }
```

### Integration Points

#### 1. MCP Request Building (mcp.py)
```python
def send_message(self, user_input: str, ...):
    # Get token allocation from budget manager
    budget = self.token_budget_manager.allocate_for_request("main_conversation")
    
    # Build request within budget
    messages = []
    token_count = 0
    
    # Add system prompts (guaranteed to fit)
    messages.append({"role": "system", "content": self.system_prompt})
    token_count += budget["system_prompts"]
    
    # Add story context (truncated if needed)
    if story_context:
        truncated_context = self.truncate_to_tokens(
            story_context, 
            budget["story_context"]
        )
        messages.append({"role": "system", "content": truncated_context})
        token_count += self.count_tokens(truncated_context)
    
    # Add conversation (smart truncation)
    conversation_budget = budget["conversation"]
    selected_messages = self.select_messages_within_budget(
        conversation_history,
        conversation_budget,
        strategy="sliding_window_with_importance"
    )
    messages.extend(selected_messages)
    
    # Add user input (pre-validated)
    messages.append({"role": "user", "content": user_input})
    
    # Verify we're within limits
    assert token_count <= self.token_budget_manager.CONTEXT_WINDOW - budget["reserved_response"]
```

#### 2. EMM Conversation Retrieval (emm.py)
```python
def get_conversation_for_mcp(self, token_budget: int) -> List[Dict[str, str]]:
    """
    Return conversation that fits within token budget.
    Prioritize recent and important messages.
    """
    with self.lock:
        selected = []
        remaining_tokens = token_budget
        
        # Start with most recent, work backwards
        for msg in reversed(self.messages):
            if msg.message_type == MessageType.MOMENTUM_STATE:
                continue  # Skip state messages
                
            msg_tokens = msg.token_estimate
            
            # Include if it fits
            if msg_tokens <= remaining_tokens:
                selected.insert(0, {
                    "role": msg.message_type.value,
                    "content": msg.content
                })
                remaining_tokens -= msg_tokens
            
            # Stop when budget exhausted
            if remaining_tokens < 100:  # Min useful message size
                break
        
        return selected
```

#### 3. SME Story Context (sme.py)
```python
def get_story_context(self, max_tokens: int = 1500) -> str:
    """
    Generate story context within token limit.
    """
    context_parts = []
    token_count = 0
    
    # Add critical state first
    critical = f"Arc: {self.story_arc.value}, Pressure: {self.pressure_level:.2f}"
    context_parts.append(critical)
    token_count += self.estimate_tokens(critical)
    
    # Add antagonist if space available
    if self.current_antagonist and token_count < max_tokens - 200:
        antagonist_info = f"Antagonist: {self.current_antagonist.name}"
        context_parts.append(antagonist_info)
        token_count += self.estimate_tokens(antagonist_info)
    
    # Add recent events if space available
    if token_count < max_tokens - 300:
        recent_events = self.get_recent_events_summary(max_tokens - token_count)
        context_parts.append(recent_events)
    
    return " | ".join(context_parts)
```

### Smart Message Selection Strategies

#### 1. Sliding Window with Importance
```python
def select_messages_with_importance(messages, token_budget):
    """
    Select messages balancing recency and importance.
    """
    # Always include last 3 messages (context continuity)
    essential = messages[-3:]
    remaining = messages[:-3]
    
    # Sort remaining by importance score
    sorted_msgs = sorted(
        remaining, 
        key=lambda m: m.importance_score,
        reverse=True
    )
    
    # Add important messages until budget exhausted
    selected = essential[:]
    used_tokens = sum(m.token_estimate for m in essential)
    
    for msg in sorted_msgs:
        if used_tokens + msg.token_estimate <= token_budget:
            selected.append(msg)
            used_tokens += msg.token_estimate
    
    # Return in chronological order
    return sorted(selected, key=lambda m: m.timestamp)
```

#### 2. Cluster-Based Selection
```python
def select_message_clusters(messages, token_budget):
    """
    Select complete conversation clusters to maintain coherence.
    """
    clusters = identify_conversation_clusters(messages)
    selected_clusters = []
    used_tokens = 0
    
    # Prioritize recent clusters
    for cluster in reversed(clusters):
        cluster_tokens = sum(m.token_estimate for m in cluster)
        if used_tokens + cluster_tokens <= token_budget:
            selected_clusters.insert(0, cluster)
            used_tokens += cluster_tokens
    
    return flatten(selected_clusters)
```

---

## PART 4: IMPLEMENTATION PRIORITIES

### Immediate (Prevent Overflow)
1. **Add token counting to MCP request building**
2. **Implement hard limit on user input (2000 tokens)**
3. **Create basic TokenBudgetManager class**
4. **Add token_budget parameter to EMM.get_conversation_for_mcp()**

### Short-term (Improve Quality)
1. **Implement sliding window with importance selection**
2. **Add story context truncation to SME**
3. **Create token usage statistics in /stats command**
4. **Add overflow warnings to debug log**

### Long-term (Optimize Experience)
1. **Implement cluster-based message selection**
2. **Add adaptive prompt sizing based on conversation length**
3. **Create tiered condensation strategies by genre**
4. **Implement predictive token allocation**

---

## PART 5: EXPECTED OUTCOMES

### With Token Budget Manager

1. **Guaranteed Context Window Compliance**
   - Never exceed 32K tokens
   - Automatic truncation when needed
   - Graceful degradation of older content

2. **Preserved Narrative Quality**
   - Important messages prioritized
   - Story continuity maintained
   - Critical events never lost

3. **Improved Performance**
   - Fewer LLM rejections
   - Consistent response times
   - Predictable behavior

4. **Enhanced Debugging**
   - Real-time token tracking
   - Budget violation alerts
   - Usage statistics

### Token Allocation Strategy (32K Total)

```
Fixed Allocations (12,500 tokens):
├─ System Prompts:     4,000
├─ User Input:         2,000  
├─ LLM Response:       5,000
└─ Safety Buffer:      1,500

Dynamic Pool (19,500 tokens):
├─ Recent Conversation: 14,000 (high priority)
├─ Story Context:        1,500 (medium priority)
├─ Semantic Analysis:    2,000 (background)
└─ Momentum Analysis:    2,000 (background)
```

---

## CONCLUSION

The current implementation has strong logic flows but lacks critical token budget coordination. The proposed TokenBudgetManager provides centralized control over token allocation, ensuring we never exceed the 32K context window while intelligently preserving the most important narrative content.

Key improvements:
- **Centralized token accounting** prevents overflow
- **Smart truncation strategies** preserve story quality
- **Priority-based allocation** optimizes token usage
- **Real-time monitoring** enables proactive management

This approach maintains the sophisticated semantic analysis and momentum tracking from the legacy implementation while adding the crucial token management layer missing from the current system.

---

*Status: Complete architectural analysis with actionable token budget solution*