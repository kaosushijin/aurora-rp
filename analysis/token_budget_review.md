# Token Budget Review

## **Critical Finding: Potential Token Budget Overflow**

After analyzing the actual code flow, I've identified **several critical token allocation issues** that could cause context window overflow beyond the 32K limit.

---

## **Documented vs. Actual Token Allocation**

### **Official Allocation (from main.py)**
- **Total Context Window**: 32,000 tokens
- **System Prompts Budget**: 5,000 tokens (with auto-condensation)
- **Max User Input**: 2,000 tokens (documented but not enforced)
- **Remaining for Conversation**: ~25,000 tokens

### **Actual Implementation Problems**
The code doesn't properly track or enforce these limits in the MCP request building process.

---

## **Critical Overflow Points Identified**

### **1. MCP Request Building (mcp.py) - HIGH RISK**

**Current Flow in `send_message()`:**
```python
messages = [{"role": "system", "content": self.system_prompt}]

# Add story context from SME if provided
if story_context:
    messages.append({"role": "system", "content": f"Story Context: {story_context}"})

# Add conversation history from EMM (last 10 messages for efficiency)
if conversation_history:
    for msg in conversation_history[-10:]:
        messages.append(msg)

# Add current user input
messages.append({"role": "user", "content": user_input})
```

**PROBLEMS:**
- **No token counting** during message building
- **"Last 10 messages"** could easily exceed remaining budget
- **Story context** addition has no size limits
- **System prompt** not validated against 5K budget before use

**OVERFLOW SCENARIO:**
- System prompts: 5,000 tokens
- Story context: 2,000 tokens (uncontrolled)
- 10 conversation messages: 15,000 tokens (possible with long conversations)
- User input: 2,000 tokens
- **Total: 24,000 tokens** - but this doesn't include LLM response space!

### **2. EMM Memory Limits (emm.py) - MEDIUM RISK**

**Current Configuration Mismatch:**
- **EMM max_memory_tokens**: 25,000 tokens
- **MCP conversation retrieval**: "last 10 messages" (no token limit)
- **Context window available**: ~27,000 tokens after system prompts

**PROBLEMS:**
- EMM can store 25K tokens, but MCP may try to send ALL of it
- No coordination between EMM limits and MCP request building
- `get_conversation_for_mcp()` doesn't enforce token budget

### **3. System Prompt Condensation (main.py) - LOW RISK**

**Current Implementation:**
- Target: 5,000 tokens for all prompts combined
- Auto-condensation if exceeded
- Validation happens at startup

**POTENTIAL ISSUE:**
- Condensation uses **temporary MCP client** which could overflow during condensation process
- No guarantee condensed prompts stay under 5K over time

### **4. User Input Validation - HIGH RISK**

**Current Status**: NO ENFORCEMENT
- Documentation says "2,000 tokens max"
- `sem.py` validation checks "4,000 characters" but doesn't convert to tokens
- `uilib.py` has MAX_USER_INPUT_TOKENS = 2000 but not consistently used

**OVERFLOW SCENARIO:**
User inputs 4,000 characters = ~1,000 tokens (safe)
BUT: User inputs very token-dense content (special chars, code, etc.) = could be 2,000+ tokens

---

## **Token Calculation Inconsistencies**

### **Multiple Token Estimation Methods**
1. **EMM**: `len(text) // 4` (conservative)
2. **Main.py**: `estimate_tokens()` function exists but may use different logic
3. **SME**: Various pattern-based estimates
4. **MCP**: No token counting during request building

### **No Centralized Token Accounting**
- Each module estimates independently
- No real-time tracking of total context usage
- No enforcement of budget limits during request building

---

## **Specific Overflow Scenarios**

### **Scenario 1: Long Conversation History**
1. EMM stores 25K tokens of conversation
2. MCP requests "last 10 messages" = potentially 15K tokens
3. Add system prompts (5K) + story context (2K) + user input (2K)
4. **Total: 24K tokens** + LLM response needs 3-5K = **27-29K tokens**
5. **OVERFLOW RISK: HIGH**

### **Scenario 2: Large Story Context**
1. SME generates detailed story context (no size limit)
2. Complex world state could be 5K+ tokens
3. Add normal conversation (10K) + prompts (5K) + user input (2K)
4. **Total: 22K tokens** before LLM response
5. **OVERFLOW RISK: MEDIUM**

### **Scenario 3: Semantic Analysis Requests**
1. Background semantic analysis requests include conversation context
2. SME momentum analysis uses 6K token budget
3. But this is ADDITIONAL to normal MCP requests
4. Could cause concurrent overflow if timing aligns badly
5. **OVERFLOW RISK: MEDIUM**

---

## **Immediate Fixes Required**

### **Priority 1: MCP Request Building**
- Add **real-time token counting** in `send_message()`
- Implement **conversation truncation** based on token budget, not message count
- Add **story context size limiting**
- Validate total request size before sending

### **Priority 2: User Input Validation**
- Enforce **2K token limit** in UI layer (`uilib.py`)
- Use **consistent token estimation** across all modules
- Reject oversized input at validation time

### **Priority 3: EMM-MCP Coordination**
- Modify `get_conversation_for_mcp()` to accept **token budget parameter**
- Return conversation that fits within specified token limit
- Consider **message importance** during truncation

### **Priority 4: Centralized Token Accounting**
- Create **TokenBudgetManager** class
- Track **real-time context usage**
- Provide **budget enforcement** across all modules

---

## **Recommended Token Allocation Strategy**

### **Safe Budget Allocation (32K total):**
- **System Prompts**: 4,000 tokens (reduced from 5K for safety)
- **Story Context**: 1,500 tokens (new limit)
- **Conversation History**: 18,000 tokens (managed by smart truncation)
- **User Input**: 2,000 tokens (enforced)
- **LLM Response**: 5,000 tokens (reserved)
- **Safety Buffer**: 1,500 tokens

### **Dynamic Allocation Logic:**
1. **Reserve fixed amounts** for system prompts and user input
2. **Calculate available space** for conversation + story context
3. **Prioritize recent conversation** over older messages
4. **Truncate story context** if conversation is more important
5. **Always reserve space** for LLM response

---

## **Long-term Improvements**

### **Intelligent Context Management**
- **Semantic importance-based** conversation truncation
- **Sliding window** with smart boundary detection
- **Compression of old content** while preserving key details

### **Budget Monitoring**
- **Real-time budget tracking** in debug mode
- **Token usage statistics** in `/stats` command
- **Overflow warnings** before they happen

### **Context Optimization**
- **Story context summarization** when it grows too large
- **Conversation clustering** to maintain narrative coherence
- **Adaptive prompt sizing** based on available budget

---

## **Critical Action Items**

1. **Immediate**: Add token counting to MCP request building
2. **Immediate**: Enforce user input token limits in UI
3. **Short-term**: Implement conversation truncation in EMM
4. **Short-term**: Add story context size limiting in SME
5. **Medium-term**: Create centralized token budget management

**Without these fixes, the system WILL overflow the 32K context window during normal usage with longer conversations.**

---

*Status: Critical token budget overflow vulnerabilities identified - immediate fixes required*