# Memory Management & Semantic Alignment Plan
*DevName RPG Client - Architectural Alignment Document*

## Executive Summary
Transform the current pattern-based semantic system into a fully LLM-driven semantic tagging and condensation pipeline that preserves narrative clarity while managing token budgets effectively.

## Current State Analysis

### What's Working
- ✅ Serial LLM queue prevents race conditions
- ✅ Immediate pattern-based tagging provides placeholder categorization
- ✅ Memory persistence with auto-save
- ✅ Basic pressure tracking for narrative momentum

### What's Broken
- ❌ **Condensation Never Triggers**: 25,000 token threshold too high
- ❌ **No Actual Summarization**: Only placeholder text "[Condensed summary...]"
- ❌ **Categories Unused**: Semantic tags assigned but not applied to preservation
- ❌ **Whole-Memory Condensation**: Should only condense oldest 10-20%
- ❌ **Pattern Matching**: Using keywords instead of LLM understanding
- ❌ **Inconsistent JSON**: LLM responses vary between models

## Phase 1: Fix Condensation Window & Triggering
*Goal: Make condensation actually trigger and target only old messages*

### 1.1 Adjust Condensation Parameters
```python
# In emm.py
class EnhancedMemoryManager:
    def __init__(self):
        self.max_memory_tokens = 5000  # Reduced from 25000 for testing
        self.condensation_window = 0.2  # Only condense oldest 20%
        self.preserve_recent_count = 20  # Always keep last 20 messages
```

### 1.2 Implement Window-Based Condensation
```python
def get_condensation_candidates(self, preserve_recent: int = 20) -> List[str]:
    """Get ONLY oldest 20% of messages for condensation"""
    if len(self.messages) <= preserve_recent:
        return []
    
    # Calculate window size (oldest 20%)
    total_eligible = len(self.messages) - preserve_recent
    window_size = max(5, int(total_eligible * self.condensation_window))
    
    # Get oldest messages in window
    candidates = []
    for msg in self.messages[:window_size]:
        if (msg.message_type != MessageType.MOMENTUM_STATE and 
            not msg.condensed):
            candidates.append(msg.id)
    
    return candidates
```

### 1.3 Testing Checkpoint
- Set `max_memory_tokens = 500` for rapid testing
- Generate 30-40 messages in conversation
- Verify condensation triggers on oldest 6-8 messages only
- Check that recent 20 messages remain untouched

**Success Criteria**: Condensation triggers reliably at threshold and only affects old messages

---

## Phase 2: Implement Actual Summarization
*Goal: Replace placeholder with real content preservation*

### 2.1 Create Summarization Prompt Template
```python
CONDENSATION_PROMPT = """
Summarize these RPG narrative messages while preserving critical information.

Messages to condense:
{messages_text}

Create a condensed summary that:
1. Preserves all story-critical events and decisions
2. Maintains character names and relationships
3. Keeps important world-building details
4. Reduces redundant descriptions and small talk

Output format: A narrative paragraph that reads naturally as a story summary.
Maximum length: {target_tokens} tokens (approximately {target_chars} characters)
"""
```

### 2.2 Implement Condensation Request
```python
def _process_condensation_request(self, request: LLMRequest):
    """Actually summarize messages using LLM"""
    candidates = request.context_data.get("candidates", [])
    
    # Get actual message content
    messages_to_condense = []
    for msg_id in candidates:
        message = self.memory_manager.get_message_by_id(msg_id)
        if message:
            messages_to_condense.append(
                f"[{message.message_type.value}]: {message.content}"
            )
    
    # Build condensation prompt
    messages_text = "\n".join(messages_to_condense)
    target_tokens = len(messages_text) // 10  # Aim for 10x compression
    
    prompt = CONDENSATION_PROMPT.format(
        messages_text=messages_text,
        target_tokens=target_tokens,
        target_chars=target_tokens * 4
    )
    
    # Make LLM request
    llm_response = self._make_llm_request_for_analysis(prompt, "condense")
    
    if llm_response.get("success"):
        condensed_text = llm_response.get("response", "")
        self.memory_manager.replace_messages_with_condensed(
            candidates, condensed_text
        )
```

### 2.3 Testing Checkpoint
- Trigger condensation with meaningful conversation
- Verify condensed content preserves key information
- Check token reduction (aim for 5-10x compression)
- Ensure narrative flow remains coherent

**Success Criteria**: Condensed summaries are coherent and preserve critical information

---

## Phase 3: Standardize LLM JSON Responses
*Goal: Ensure consistent semantic analysis across different models*

### 3.1 Create Explicit JSON Schema Prompt
```python
SEMANTIC_ANALYSIS_PROMPT = """
Analyze this RPG message for narrative importance.

Message: "{content}"

Respond with EXACTLY this JSON structure (no other text):
{{
    "importance_score": <decimal between 0.0 and 1.0>,
    "category": <exactly one of: "story_critical", "character_focused", "relationship_dynamics", "emotional_significance", "world_building", "standard">,
    "preserve": <boolean true or false>,
    "reason": <string explaining the categorization>
}}

Scoring guide:
- 0.9-1.0: Story critical (main quest, major plot points)
- 0.7-0.8: Character development or relationships
- 0.5-0.6: Emotional moments or world building
- 0.3-0.4: Standard dialogue and actions
- 0.0-0.2: Trivial or redundant content

IMPORTANT: Output ONLY the JSON object, no markdown, no explanation.
"""
```

### 3.2 Implement Robust JSON Parser
```python
def _parse_semantic_json(self, response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON with validation and type coercion"""
    try:
        # Strip any markdown or extra text
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response[json_start:json_end]
            data = json.loads(json_text)
            
            # Validate and coerce types
            result = {
                "importance_score": float(data.get("importance_score", 0.4)),
                "category": str(data.get("category", "standard")),
                "preserve": bool(data.get("preserve", False)),
                "reason": str(data.get("reason", ""))
            }
            
            # Validate category
            valid_categories = ["story_critical", "character_focused", 
                              "relationship_dynamics", "emotional_significance",
                              "world_building", "standard"]
            if result["category"] not in valid_categories:
                result["category"] = "standard"
            
            # Clamp importance score
            result["importance_score"] = max(0.0, min(1.0, result["importance_score"]))
            
            return result
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        self._log_debug(f"JSON parse failed: {e}")
    
    return None  # Fallback to pattern matching
```

### 3.3 Testing Checkpoint
- Test with multiple qwen3 model variants
- Verify consistent JSON structure returned
- Check type consistency (decimals not strings)
- Validate category assignments match content

**Success Criteria**: 95%+ successful JSON parsing across model variants

---

## Phase 4: Replace Pattern Matching with LLM Analysis
*Goal: Full LLM semantic tagging for all messages*

### 4.1 Update Immediate Analysis Pipeline
```python
def _perform_immediate_semantic_analysis(self, message) -> None:
    """Queue LLM semantic analysis instead of patterns"""
    # Queue for async LLM analysis
    request_id = self._queue_semantic_analysis(
        message.id, 
        message.content, 
        message.message_type.value
    )
    
    # Set temporary defaults (will be updated async)
    message.content_category = "pending"
    message.importance_score = 0.5
    message.semantic_metadata = {
        "analysis_request_id": request_id,
        "analysis_queued_at": time.time()
    }
```

### 4.2 Process Semantic Results
```python
def _process_semantic_analysis_result(self, message_id: str, result: Dict):
    """Update message with LLM analysis results"""
    message = self.memory_manager.get_message_by_id(message_id)
    if message:
        message.content_category = result["category"]
        message.importance_score = result["importance_score"]
        message.semantic_metadata.update({
            "analysis_completed_at": time.time(),
            "preserve": result["preserve"],
            "reason": result["reason"]
        })
        
        # Trigger save
        self.memory_manager._auto_save()
```

### 4.3 Testing Checkpoint
- Monitor LLM queue for semantic requests
- Verify all messages get analyzed within 2-3 seconds
- Check categories align with content semantically
- Validate importance scores are reasonable

**Success Criteria**: All messages tagged by LLM within 5 seconds of creation

---

## Phase 5: Apply Semantic Categories to Condensation
*Goal: Use importance scores to guide preservation*

### 5.1 Implement Category-Aware Selection
```python
def select_messages_for_condensation(self, candidates: List[str]) -> List[str]:
    """Select which candidates to actually condense based on importance"""
    messages_by_importance = []
    
    for msg_id in candidates:
        message = self.get_message_by_id(msg_id)
        if message:
            messages_by_importance.append({
                "id": msg_id,
                "importance": message.importance_score,
                "category": message.content_category,
                "tokens": message.token_estimate
            })
    
    # Sort by importance (lowest first)
    messages_by_importance.sort(key=lambda x: x["importance"])
    
    # Select messages to condense until we free enough tokens
    selected = []
    tokens_to_free = self._calculate_tokens_to_free()
    tokens_freed = 0
    
    for msg_info in messages_by_importance:
        # Never condense story_critical (importance >= 0.9)
        if msg_info["importance"] >= 0.9:
            continue
            
        selected.append(msg_info["id"])
        tokens_freed += msg_info["tokens"]
        
        if tokens_freed >= tokens_to_free:
            break
    
    return selected
```

### 5.2 Group by Category for Condensation
```python
def create_condensation_groups(self, selected_ids: List[str]) -> List[List[str]]:
    """Group messages by category for coherent summaries"""
    groups = {}
    
    for msg_id in selected_ids:
        message = self.get_message_by_id(msg_id)
        if message:
            category = message.content_category
            if category not in groups:
                groups[category] = []
            groups[category].append(msg_id)
    
    # Convert to list of groups
    return list(groups.values())
```

### 5.3 Testing Checkpoint
- Generate messages with varied importance
- Verify low-importance messages condense first
- Check story_critical messages never condense
- Validate grouped summaries maintain coherence

**Success Criteria**: Condensation preserves high-importance content while reducing tokens by 50%+

---

## Phase 6: Memory Splitting for Mixed Categories
*Goal: Split messages with multiple importance levels*

### 6.1 Detect Mixed-Importance Content
```python
SPLIT_DETECTION_PROMPT = """
Analyze if this message contains multiple distinct narrative elements with different importance levels.

Message: "{content}"

Identify if this should be split into separate memories. If yes, provide splits.

Respond with JSON:
{{
    "should_split": <boolean>,
    "segments": [
        {{
            "content": <segment text>,
            "category": <category name>,
            "importance": <0.0-1.0>
        }}
    ]
}}

Split only if segments have importance differences > 0.3.
"""
```

### 6.2 Implement Message Splitting
```python
def split_message_if_needed(self, message: Message) -> List[Message]:
    """Split message into multiple memories if mixed importance"""
    # Check for splitting via LLM
    split_analysis = self._analyze_for_splitting(message.content)
    
    if split_analysis["should_split"] and len(split_analysis["segments"]) > 1:
        split_messages = []
        for segment in split_analysis["segments"]:
            new_msg = Message(
                content=segment["content"],
                message_type=message.message_type,
                timestamp=message.timestamp
            )
            new_msg.content_category = segment["category"]
            new_msg.importance_score = segment["importance"]
            split_messages.append(new_msg)
        
        return split_messages
    
    return [message]  # No split needed
```

### 6.3 Testing Checkpoint
- Test with mixed-content messages
- Verify appropriate splitting decisions
- Check split segments maintain context
- Validate no over-splitting of coherent content

**Success Criteria**: Mixed messages split intelligently without fragmenting narrative

---

## Testing Methodology

### Test Configuration
```python
# In config for testing
TEST_CONFIG = {
    "memory": {
        "max_tokens": 500,  # Low for rapid testing
        "condensation_window": 0.2,
        "preserve_recent": 10
    },
    "semantic": {
        "use_llm": True,  # Toggle for A/B testing
        "queue_priority": 3,
        "timeout": 5.0
    }
}
```

### Test Scenarios
1. **Rapid Conversation**: 50 messages in 5 minutes
2. **Mixed Content**: Combat, dialogue, exploration
3. **Story Critical**: Main quest revelations
4. **Trivial Content**: Weather comments, small talk
5. **Long Messages**: Multi-paragraph descriptions

### Success Metrics
- Condensation triggers at ~500 tokens
- 5-10x token reduction post-condensation
- 95%+ JSON parse success rate
- <5 second semantic tagging latency
- Story-critical content 100% preserved
- Narrative coherence maintained

---

## Implementation Timeline

### Week 1: Foundation (Phases 1-2)
- Day 1-2: Fix condensation window and triggering
- Day 3-4: Implement actual summarization
- Day 5: Integration testing

### Week 2: LLM Integration (Phases 3-4)
- Day 1-2: Standardize JSON responses
- Day 3-4: Replace pattern matching
- Day 5: Performance testing

### Week 3: Advanced Features (Phases 5-6)
- Day 1-2: Category-aware condensation
- Day 3-4: Message splitting logic
- Day 5: Full system testing

---

## Risk Mitigation

### Fallback Strategies
1. **Pattern matching fallback** if LLM times out
2. **Default categorization** for parse failures
3. **Skip condensation** if errors exceed threshold
4. **Manual triggers** for testing/debugging

### Performance Safeguards
- Queue timeout: 5 seconds max per request
- Batch semantic analysis for efficiency
- Cache recent analyses to avoid duplicates
- Background processing for non-critical tags

---

## Configuration Flags

```python
# Feature flags for progressive rollout
FEATURE_FLAGS = {
    "use_llm_semantic": False,  # Start with patterns
    "enable_condensation": False,  # Enable when ready
    "allow_message_splitting": False,  # Phase 6 only
    "condensation_debug": True,  # Verbose logging
}
```

Toggle flags as each phase completes successfully.

---

## Notes for Claude/Sonnet

When implementing:
1. **Always preserve method signatures** - Don't rename existing methods
2. **Test with low token limits** - Set max_tokens to 500 for rapid testing
3. **Log extensively** - Add debug logs at each decision point
4. **Fail gracefully** - Pattern fallback if LLM fails
5. **Monitor queue depth** - Ensure semantic queue doesn't backup

Remember: The goal is narrative preservation, not just token reduction. Every condensation should maintain story coherence and emotional impact.