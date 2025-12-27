TASK: Debug and fix conversation context memory loss in a LangChain + LangGraph Node.js agent.

PROBLEM:
Conversation context (productType/style) is lost between turns.
Example:
Turn 1: "I want a dining table japandi" → context = { productType: "meja makan", style: "japandi" }
Turn 2: "yes i need for a small room" → context becomes {}

EXPECTED BEHAVIOR:
Context must persist across turns for the same thread_id.
Modifiers like room size or style-only queries must reuse previous productType.

TECH STACK:
- Node.js / TypeScript
- LangChain
- LangGraph StateGraph
- MongoDBSaver (checkpoint)
- Custom in-memory Map for context memory

ROOT CAUSE TO INVESTIGATE:
- thread_id inconsistency across requests
- context overwritten with empty object
- unsafe defaulting (e.g. `|| {}`)
- context not re-saved on every turn
- context not merged, but replaced

REQUIREMENTS:
1. Ensure thread_id is stable and logged on every request.
2. Prevent context from being reset or replaced with `{}`.
3. Context updates must be MERGED, not overwritten.
4. Never nullify productType unless explicitly changed by user.
5. Style-only or modifier-only queries must reuse previous productType.
6. Persist context safely (in-memory or MongoDB).
7. Add defensive logs:
   - thread_id
   - context BEFORE update
   - context AFTER update
8. Provide corrected TypeScript code snippets:
   - context retrieval
   - context update
   - contextual query building

DO NOT:
- Change business logic
- Rewrite the whole agent
- Suggest theoretical explanations

OUTPUT:
- Root cause summary
- Exact code fixes (minimal diff)
- Final corrected logic flow