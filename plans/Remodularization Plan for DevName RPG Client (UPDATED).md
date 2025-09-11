# DevName RPG Client - Remodularization Fix Plan (UPDATED)

## Executive Summary
Progress has been made on the remodularization. The AsyncSyncBridge is implemented and working. The main remaining issue is MCP communication - the sync_message_processor in orch.py is having trouble communicating with the MCP server. This updated plan reflects what's been accomplished and what still needs fixing.

## Current Status

### ✅ COMPLETED FIXES

#### 1. AsyncSyncBridge Implementation (Priority 2) - DONE
- `AsyncSyncBridge` class successfully created in `orch.py`
- Dedicated event loop running in separate thread
- Bridge starts/stops properly with orchestrator lifecycle
- `run_async_safely()` method provides timeout and error handling

#### 2. Orchestration Flow Fixed (Priority 1) - PARTIALLY DONE
- Orchestrator no longer shuts down prematurely
- Modules remain available throughout UI lifecycle
- UI runs with live orchestrator reference
- **Issue**: Message processors still having communication issues

#### 3. Module Dependencies (Priority 3) - MOSTLY DONE
- `sem.py` created with consolidated semantic logic
- `uilib.py` created with consolidated UI utilities
- `ui.py` refactored from `nci.py`
- `emm.py` and `sme.py` using `sem.py`
- Clean dependency hierarchy established

#### 4. Configuration Standardized (Priority 4) - DONE
- Prompt files using `.prompt` extension consistently
- Configuration centralized in orchestrator
- Default prompts in place

### ❌ REMAINING ISSUES

#### 1. MCP Communication Failure (NEW CRITICAL ISSUE)
**Problem**: The sync_message_processor can't successfully communicate with MCP server
- Location: `orch.py` - sync_message_processor function
- Symptoms: Debug shows payload being sent but response handling fails
- Impact: No AI responses, core functionality broken

#### 2. Payload Format Mismatch
**Problem**: Inconsistent payload formats between test and actual messages
- Test payload: `{"message": user_message, "model": "..."}`
- Actual payload: `{"messages": [...], "model": "...", "stream": false}`
- The MCP server expects the OpenAI-compatible format with `messages` array

#### 3. Response Processing Issues
**Problem**: Response from MCP server not being properly extracted
- Server returns: `{"message": {"content": "..."}}`
- Code expects different format at different points
- Inconsistent response handling between phases

## Updated Fix Implementation Plan

### Priority 1: Fix MCP Communication (CRITICAL - NEW)
**Severity**: CRITICAL - Blocks all AI functionality

#### Root Cause Analysis:
1. The test MCP call in Phase 2 uses a simplified payload format
2. The actual message processor needs the full OpenAI-compatible format
3. Response extraction logic is inconsistent

#### Required Changes in `orch.py`:

```python
# In sync_message_processor function
async def async_mcp_call():
    import httpx
    
    # Build proper message array for OpenAI format
    messages = []
    
    # Add system prompts if available
    if orchestrator.loaded_prompts.get('critrules'):
        messages.append({
            "role": "system", 
            "content": orchestrator.loaded_prompts['critrules']
        })
    
    # Add conversation history from memory_manager
    if memory_manager:
        history = memory_manager.get_conversation_for_mcp()
        messages.extend(history[-10:])  # Last 10 messages
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Proper OpenAI-compatible payload
    payload = {
        "model": "qwen2.5:14b-instruct-q4_k_m",
        "messages": messages,
        "stream": False  # Ensure not streaming
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://127.0.0.1:3456/chat",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Handle MCP server response format
            if "message" in data and "content" in data["message"]:
                ai_response = data["message"]["content"]
                
                # Store in memory manager
                if memory_manager:
                    from emm import MessageType
                    memory_manager.add_message(ai_response, MessageType.GM)
                
                return {"success": True, "response": ai_response}
            else:
                return {"error": f"Invalid response format: {data}"}
        else:
            return {"error": f"MCP error: {response.status_code}"}
```

### Priority 2: Consolidate MCP Communication
**Severity**: HIGH - Reduces complexity and potential for errors

#### Solution:
Create a unified MCP communication method that both test and actual calls can use.

```python
# Add to DevNameOrchestrator class
async def call_mcp_server(self, messages: list, timeout: float = 30.0) -> dict:
    """Unified MCP server communication"""
    import httpx
    
    payload = {
        "model": "qwen2.5:14b-instruct-q4_k_m",
        "messages": messages,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "http://127.0.0.1:3456/chat",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                if "message" in data and "content" in data["message"]:
                    return {"success": True, "content": data["message"]["content"]}
                else:
                    return {"success": False, "error": "Invalid response format"}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Priority 3: Fix Memory Manager Integration
**Severity**: MEDIUM - Needed for conversation continuity

#### Issue:
The memory manager's `get_conversation_for_mcp()` method needs to return properly formatted messages.

#### Required Changes in `emm.py`:
```python
def get_conversation_for_mcp(self) -> List[Dict[str, str]]:
    """Get conversation history in OpenAI message format"""
    messages = []
    for msg in self.messages[-20:]:  # Last 20 messages
        role = "user" if msg.msg_type == MessageType.USER else "assistant"
        messages.append({
            "role": role,
            "content": msg.content
        })
    return messages
```

### Priority 4: Simplify Debug Output
**Severity**: LOW - Improves troubleshooting

Remove excessive debug print statements and use proper logging:

```python
# Replace print() statements with debug logger
if self.debug_logger:
    self.debug_logger.debug(f"MCP request with {len(messages)} messages", "ORCHESTRATOR")
    self.debug_logger.debug(f"MCP response received: {len(content)} chars", "ORCHESTRATOR")
```

## Implementation Steps (Updated)

### Step 1: Fix MCP Communication (1-2 hours)
1. Update sync_message_processor to use correct payload format
2. Fix response extraction to handle server's actual format
3. Test with simple message exchange

### Step 2: Consolidate MCP Calls (1 hour)
1. Create unified `call_mcp_server()` method
2. Update all MCP calls to use unified method
3. Remove duplicate code

### Step 3: Fix Memory Integration (30 minutes)
1. Update `get_conversation_for_mcp()` format
2. Ensure proper role assignment
3. Test conversation continuity

### Step 4: Clean Up Debug Code (30 minutes)
1. Remove print() statements
2. Use debug logger consistently
3. Add meaningful log messages

## Testing Checklist (Updated)

### Immediate Tests:
- [x] Application starts without errors
- [x] UI displays properly
- [x] Can type and submit messages
- [ ] **MCP server responds to messages** ← Current blocker
- [ ] AI responses appear in display
- [x] Clean shutdown with Ctrl+C

### Integration Tests (After MCP Fix):
- [ ] Conversation history maintained
- [ ] Background semantic analysis runs
- [ ] Story momentum updates
- [ ] Memory persistence works
- [ ] Theme switching works
- [ ] Terminal resize handling works

## Key Files Requiring Changes

### `orch.py` (HIGH PRIORITY)
- Fix sync_message_processor payload format
- Fix response extraction logic
- Add unified MCP communication method
- Clean up debug prints

### `emm.py` (MEDIUM PRIORITY)
- Update `get_conversation_for_mcp()` return format
- Ensure MessageType mapping is correct

### `mcp.py` (LOW PRIORITY - Already Working)
- Module is properly structured with httpx
- Consider adding sync wrapper methods if needed

## Success Metrics

### Immediate Success:
- User types message → MCP server receives it → Response displayed

### Full Success:
- Complete conversation flow working
- All background processes active
- State persistence functional
- Clean modular architecture maintained

## Debug Strategy

### Current Debug Points:
1. **Payload Construction**: Log exact payload being sent
2. **HTTP Response**: Log status code and raw response
3. **Response Parsing**: Log extracted content
4. **Memory Storage**: Log successful storage

### Debug Commands to Add:
```python
# In sync_message_processor
self.debug_logger.debug(f"Payload: {json.dumps(payload, indent=2)}", "MCP_DEBUG")
self.debug_logger.debug(f"Response: {response.status_code} - {response.text[:200]}", "MCP_DEBUG")
self.debug_logger.debug(f"Extracted: {ai_response[:100]}", "MCP_DEBUG")
```

## Timeline Estimate (Updated)

- **MCP Communication Fix**: 1-2 hours (CRITICAL)
- **Consolidation**: 1 hour
- **Memory Integration**: 30 minutes
- **Cleanup**: 30 minutes
- **Testing**: 1 hour

**Total Remaining**: 4-5 hours of focused work

## Notes for Implementation

### Critical Focus Areas:
1. **Payload Format**: Must match MCP server expectations exactly
2. **Response Handling**: Must extract content from correct JSON path
3. **Error Handling**: Must gracefully handle connection failures
4. **Memory Integration**: Must maintain conversation context

### What's Working Well:
- AsyncSyncBridge is functional
- Module architecture is clean
- UI is responsive
- Dependency hierarchy is correct

### Next Steps After MCP Fix:
1. Verify background semantic analysis
2. Test story momentum engine
3. Confirm state persistence
4. Performance optimization

## Conclusion

The remodularization has made significant progress. The architectural improvements are in place, and the main remaining issue is the MCP communication format mismatch. Once this is fixed, the system should be fully functional with a much cleaner, more maintainable architecture than the original.