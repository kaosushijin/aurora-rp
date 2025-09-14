## Remodularization Testing Plan

### Overview
This testing plan evaluates the semantic analysis pipeline, memory condensation system, antagonist generation, and story momentum tracking in the DevName RPG Client's remodularized architecture.

### Test Environment Setup

1. **Required Files**
   - All `.py` modules in root directory
   - `critrules.prompt` (required)
   - `companion.prompt` and `lowrules.prompt` (optional)
   - Clean `memory.json` for each test run

2. **Debug Mode Execution**
   ```bash
   python main.py --debug
   ```

3. **Output Artifacts**
   - `debug.log` - Complete execution trace
   - `memory.json` - Conversation persistence with semantic metadata

### Phase 1: Input Validation & Categorization Testing

#### Test Cases for `sem.py::validate_input()`

**1.1 Command Detection**
- Input: `/help`, `/stats`, `/analyze`, `/theme dark`
- Expected: `category: "command"`, `confidence: 1.0`
- Validation: Check debug.log for "SEM: validate_input" entries

**1.2 Narrative Detection**
- Input: "I draw my sword and approach the shadowy figure"
- Expected: `category: "narrative"`, `confidence: 0.8`

**1.3 Meta Queries**
- Input: "How do I cast spells?" / "What can I do here?"
- Expected: `category: "meta"`, `confidence: 0.7`

**1.4 Question Detection**
- Input: "What's behind the iron door?" / "Who is the innkeeper?"
- Expected: `category: "query"`, `confidence: 0.6`

**1.5 Edge Cases**
- Empty input: Expected validation failure
- 5000+ character input: Expected "Input too long" error
- Unicode/special characters: Should handle gracefully

### Phase 2: Semantic Analysis Quality Testing

#### Test Scenarios for Message Categorization

**2.1 Story Critical Events**
```
User: "I confront the dark lord in his throne room"
Expected: importance_score > 0.8, categories: ["story_critical"]
Preservation: 90%
```

**2.2 Character Development**
```
User: "I tell the orphan about my tragic past"
Expected: importance_score > 0.7, categories: ["character_focused"]
Preservation: 80%
```

**2.3 World Building**
```
Assistant: "The ancient ruins bear inscriptions in Elder Script..."
Expected: importance_score > 0.5, categories: ["world_building"]
Preservation: 50%
```

**2.4 Standard Interactions**
```
User: "I look around the room"
Expected: importance_score < 0.5, categories: ["standard"]
Preservation: 40%
```

### Phase 3: Memory Condensation Testing

#### Test Protocol

1. **Generate 50+ messages** mixing categories
2. **Trigger condensation** at 25,000 token threshold
3. **Verify in memory.json:**
   - `condensed: true` flags on summaries
   - `content_category: "condensed_summary"`
   - Original message IDs replaced
   - Token count reduced below threshold

#### Expected Behavior
- High-importance messages preserved verbatim
- Low-importance messages grouped and summarized
- Semantic ordering maintained
- No story-critical information lost

### Phase 4: Antagonist Generation & Tracking

#### Test Sequence

**4.1 Pressure Buildup**
```
Sequence:
1. Exploration phase (pressure < 0.3)
2. Insert tension: "I hear growling from the shadows"
3. Escalation: "The creature emerges, eyes glowing red"
4. Expected: Antagonist object created with threat_level > 0.5
```

**4.2 Antagonist Persistence**
Check in `memory.json` for `MOMENTUM_STATE` messages:
```json
{
  "antagonist": {
    "name": "Shadow Beast",
    "threat_level": 0.7,
    "active": true,
    "last_mention": "..."
  }
}
```

**4.3 Commitment Levels**
Monitor state transitions:
- `testing` → `engaged` (pressure 0.5-0.7)
- `engaged` → `desperate` (pressure 0.7-0.9)
- `desperate` → `cornered` (pressure > 0.9)

### Phase 5: Story Momentum Engine Validation

#### Narrative Time Tracking

**5.1 Duration Detection**
```
Input: "I rest for an hour by the campfire"
Expected duration: 3600 seconds
Input: "I quickly glance around"
Expected duration: 5 seconds
```

**5.2 Pressure Decay**
- Verify pressure decreases by -0.02 per exchange without conflict
- Confirm floor ratcheting prevents regression below established minimums

**5.3 Arc Progression**
```
SETUP (0.0-0.3) → RISING (0.3-0.7) → CLIMAX (0.7-1.0) → RESOLUTION
```

### Phase 6: LLM Integration Testing

#### Prompt Injection Validation

**6.1 Check System Prompts**
In debug.log, verify concatenation:
```
MCP: System prompt configured: X chars from 3 sections
- critrules.prompt included
- companion.prompt included (if present)
- lowrules.prompt included (if present)
```

**6.2 Antagonist Instructions**
When pressure > 0.5, verify LLM receives:
```
Story Context: pressure_level: 0.65, antagonist_status: active (Shadow Beast)
```

### Phase 7: Integration Testing Checklist

- [ ] User input immediately echoes before LLM response
- [ ] Background semantic analysis triggers every 30 seconds
- [ ] Momentum analysis triggers every 15 messages
- [ ] Auto-save creates `.bak` files atomically
- [ ] Condensation preserves story continuity
- [ ] Antagonist appears when pressure exceeds 0.5
- [ ] Pressure ratcheting prevents stagnation
- [ ] Narrative time accumulates correctly
- [ ] Token budget stays under 25,000

### Testing Workflow

1. **Start fresh session** with debug mode
2. **Generate 30-40 message conversation** following test scenarios
3. **Force analysis** with `/analyze` at key points
4. **Monitor debug.log** for:
   - "SEM: Starting semantic analysis"
   - "SME: Processed input: pressure="
   - "EMM: Requesting condensation"
   - "MCP: Request:" showing context injection

5. **Export artifacts** after 40+ messages:
   - `debug.log`
   - `memory.json`
   - Screenshot of terminal showing antagonist involvement

### Sonnet's Analysis Responsibilities

When provided with `debug.log` and `memory.json`:

1. **Semantic Analysis Quality**
   - Count messages by category
   - Calculate preservation ratios
   - Identify miscategorized content
   - Evaluate condensation quality

2. **Antagonist Tracking**
   - Verify creation trigger at pressure > 0.5
   - Track threat_level progression
   - Confirm persistence across analyses
   - Evaluate LLM compliance with antagonist instructions

3. **Momentum Analysis**
   - Plot pressure curve over time
   - Verify narrative time accumulation
   - Check ratcheting behavior
   - Confirm arc transitions

4. **System Integration**
   - Identify any orchestration failures
   - Check for memory leaks or thread issues
   - Verify token budget compliance
   - Assess overall narrative coherence

5. **Recommendations**
   - Identify specific methods needing adjustment
   - Suggest prompt refinements
   - Highlight architectural issues
   - Propose optimization strategies
