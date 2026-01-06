/**
 * DEKO - ReAct Design Partner Agent
 * 
 * Architecture: Cyclic Graph with Tool-Calling LLM
 * 
 * Flow: Start → Enricher → Agent ←→ Tools → End
 * 
 * The agent is an interior design consultant that:
 * 1. Analyzes user context (room size, style, budget)
 * 2. Provides design consultation and advice
 * 3. Searches products when relevant
 * 4. Never hallucinates - only mentions products from search results
 */

import { StateGraph, Annotation, END, START } from "@langchain/langgraph"
import { ToolNode } from "@langchain/langgraph/prebuilt"
import { ChatOpenAI } from "@langchain/openai"
import { AIMessage, BaseMessage, HumanMessage, ToolMessage } from "@langchain/core/messages"
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb"
import { MongoClient } from "mongodb"
import "dotenv/config"

import { createProductSearchTool, type ProductItem } from "./services/tools/productSearchTool"

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface UserProfile {
  stylePreference?: string[]      // ["Minimalist", "Japandi"]
  budgetRange?: { min?: number; max?: number }
  roomDimensions?: string         // "3x3 meters"
  constraints?: string[]          // ["cat owner", "small apartment"]
  preferredMaterials?: string[]
  preferredColors?: string[]
}

export interface AgentResponse {
  intro: string
  products: ProductItem[]
  followUp: string
  meta?: {
    hasProducts: boolean
    searchType?: "vector" | "text" | "none"
    productCount: number
    intent?: string
    detectedLanguage?: string
  }
}

// ============================================================================
// STATE DEFINITION
// ============================================================================

const AgentState = Annotation.Root({
  // Message history (accumulates)
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y)
  }),

  // User profile with smart merge for arrays
  userProfile: Annotation<UserProfile>({
    reducer: (current, update) => {
      if (!update || Object.keys(update).length === 0) return current || {}
      return {
        ...current,
        ...update,
        // Merge arrays (deduplicated)
        stylePreference: [...new Set([
          ...(current?.stylePreference || []),
          ...(update.stylePreference || [])
        ])],
        constraints: [...new Set([
          ...(current?.constraints || []),
          ...(update.constraints || [])
        ])],
        preferredMaterials: [...new Set([
          ...(current?.preferredMaterials || []),
          ...(update.preferredMaterials || [])
        ])],
        preferredColors: [...new Set([
          ...(current?.preferredColors || []),
          ...(update.preferredColors || [])
        ])]
      }
    }
  }),

  // Active products from last search (for UI carousel)
  activeProducts: Annotation<ProductItem[]>({
    reducer: (current, newProducts) => newProducts ?? current ?? []
  }),

  // Language preference
  language: Annotation<"id" | "en">({
    reducer: (current, update) => update ?? current ?? "en"
  })
})

type AgentStateType = typeof AgentState.State

// ============================================================================
// SYSTEM PROMPT - DEKO PERSONA
// ============================================================================

const SYSTEM_PROMPT = `You are "Deko", a Senior Interior Designer and furniture consultant at Home Decor Indonesia.

CORE IDENTITY:
- Warm, approachable, and knowledgeable
- You help customers find the perfect furniture for their homes
- You give honest, practical design advice

BEHAVIOR RULES:
1. **Analyze First**: When a customer describes their needs, understand their:
   - Room size and layout
   - Style preferences (Japandi, Minimalist, Modern, etc.)
   - Budget constraints
   - Any special needs (kids, pets, small apartment)

2. **Consult + Recommend**: Never just dump product links. Explain WHY something fits.
   Example: "For small rooms, I recommend raised-leg sofas to create a sense of space. Let me find some options..."

3. **ALWAYS Use Tools for Products**: 
   - Call 'search_products' before mentioning ANY specific products
   - NEVER invent or hallucinate product names/prices
   - If search returns empty, admit it honestly: "Sorry, we don't have that in stock right now."

4. **Be Conversational**: 
   - Ask clarifying questions when needed
   - Remember context from earlier in the conversation
   - It's okay to just chat if the user wants to discuss ideas

DESIGN KNOWLEDGE:
- **Japandi**: Blend of Japanese minimalism and Scandinavian warmth. Natural wood + neutral colors.
- **Minimalist**: Clean lines, uncluttered, functional. Less is more.
- **Scandinavian**: Light wood, cozy textiles, bright and airy.
- **Industrial**: Metal accents, raw textures, exposed elements.
- **Classic**: Timeless elegance, rich woods, traditional craftsmanship.

- **Teak wood**: Excellent for Indonesian climate (humidity resistant)
- **Leather**: Durable, gets better with age, easy to clean
- **Fabric**: Comfortable, more color options, needs more care

RESPONSE FORMATTING:
- Keep responses concise but helpful (2-3 paragraphs max)
- Be persuasive but honest
- When showing products, explain briefly why each one fits
- Use Indonesian Rupiah for prices (Rp)

CURRENT USER PROFILE:
{userProfile}
`

// ============================================================================
// CONTEXT ENRICHER NODE
// ============================================================================

async function enricherNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const lastMessage = state.messages[state.messages.length - 1]

  // Only process human messages
  if (lastMessage._getType() !== "human") {
    return {}
  }

  const content = typeof lastMessage.content === "string"
    ? lastMessage.content
    : JSON.stringify(lastMessage.content)

  const msgLower = content.toLowerCase()

  console.log(`[Enricher] Processing: "${content}"`)

  // Extract profile updates (rule-based for speed)
  const updates: UserProfile = {}

  // Style detection
  const styles: Record<string, string> = {
    "japandi": "Japandi",
    "minimalist": "Minimalist",
    "minimalis": "Minimalist",
    "scandinavian": "Scandinavian",
    "industrial": "Industrial",
    "klasik": "Classic",
    "classic": "Classic",
    "modern": "Modern",
    "contemporary": "Contemporary"
  }

  for (const [key, value] of Object.entries(styles)) {
    if (msgLower.includes(key)) {
      updates.stylePreference = [value]
      console.log(`[Enricher] Detected style: ${value}`)
      break
    }
  }

  // Room dimension detection
  const dimensionMatch = content.match(/(\d+)\s*[xX×]\s*(\d+)\s*(m|meter|meters)?/)
  if (dimensionMatch) {
    updates.roomDimensions = `${dimensionMatch[1]}x${dimensionMatch[2]}m`
    console.log(`[Enricher] Detected dimensions: ${updates.roomDimensions}`)
  }

  // Budget detection (IDR)
  const budgetMatch = content.match(/(?:budget|harga|maksimal?|under|dibawah|sekitar)\s*(?:rp\.?|rupiah)?\s*(\d+(?:[.,]\d+)?)\s*(jt|juta|rb|ribu|k|m)?/i)
  if (budgetMatch) {
    let amount = parseFloat(budgetMatch[1].replace(',', '.'))
    const unit = budgetMatch[2]?.toLowerCase()
    if (unit === 'jt' || unit === 'juta' || unit === 'm') amount *= 1000000
    if (unit === 'rb' || unit === 'ribu' || unit === 'k') amount *= 1000
    updates.budgetRange = { max: amount }
    console.log(`[Enricher] Detected budget max: ${amount}`)
  }

  // Color preferences
  const colors = ["putih", "white", "hitam", "black", "coklat", "brown", "abu", "grey", "gray",
    "cream", "beige", "biru", "blue", "hijau", "green", "merah", "red"]
  for (const color of colors) {
    if (msgLower.includes(color)) {
      updates.preferredColors = [color]
      console.log(`[Enricher] Detected color: ${color}`)
      break
    }
  }

  // Material preferences
  const materials: Record<string, string> = {
    "kayu": "wood", "teak": "teak", "jati": "teak",
    "kulit": "leather", "leather": "leather",
    "kain": "fabric", "fabric": "fabric",
    "rotan": "rattan", "rattan": "rattan",
    "metal": "metal", "besi": "metal"
  }

  for (const [key, value] of Object.entries(materials)) {
    if (msgLower.includes(key)) {
      updates.preferredMaterials = [value]
      console.log(`[Enricher] Detected material: ${value}`)
      break
    }
  }

  // Constraints detection
  if (msgLower.includes("kecil") || msgLower.includes("small") || msgLower.includes("sempit")) {
    updates.constraints = ["small space"]
  }
  if (msgLower.includes("kucing") || msgLower.includes("cat") || msgLower.includes("anjing") || msgLower.includes("dog")) {
    updates.constraints = [...(updates.constraints || []), "pet owner"]
  }
  if (msgLower.includes("anak") || msgLower.includes("kid") || msgLower.includes("child")) {
    updates.constraints = [...(updates.constraints || []), "has children"]
  }

  // Language detection
  const isIndonesian = /\b(apa|ada|saya|mau|cari|tolong|halo|hai|boleh|bisa|bagus|mencari|gimana|dong)\b/i.test(content)

  if (Object.keys(updates).length > 0) {
    console.log(`[Enricher] Profile updates:`, updates)
  }

  return {
    userProfile: Object.keys(updates).length > 0 ? updates : undefined,
    language: isIndonesian ? "id" : "en"
  }
}


// ============================================================================
// ROUTER FUNCTION
// ============================================================================

function routeAfterAgent(state: AgentStateType): string {
  const lastMessage = state.messages[state.messages.length - 1]

  // Check if the last message has tool calls
  if (lastMessage._getType() === "ai") {
    const aiMessage = lastMessage as AIMessage
    if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
      console.log(`[Router] Agent requested tool calls, routing to tools`)
      return "tools"
    }
  }

  console.log(`[Router] No tool calls, ending`)
  return END
}

// ============================================================================
// MAIN AGENT FUNCTION
// ============================================================================

// Global tool instance (will be initialized on first call)
let productSearchToolInstance: ReturnType<typeof createProductSearchTool> | null = null

export async function callAgent(
  client: MongoClient,
  query: string,
  thread_id: string
): Promise<AgentResponse> {
  console.log(`\n[Deko] ============= NEW REQUEST =============`)
  console.log(`[Deko] Thread: ${thread_id}`)
  console.log(`[Deko] Query: "${query}"`)

  try {
    const dbName = "admin_db"

    // Initialize product search tool (with side-effect storage)
    if (!productSearchToolInstance) {
      productSearchToolInstance = createProductSearchTool(client)
    }
    productSearchToolInstance.clearResults()

    const tools = [productSearchToolInstance.tool]

    // Create model with tools bound
    const model = new ChatOpenAI({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama-3.3-70b-versatile",
      temperature: 0.7,
      configuration: {
        baseURL: "https://api.groq.com/openai/v1",
      },
    }).bindTools(tools)

    // Create tool node
    const toolNode = new ToolNode(tools)

    // Local agent node that captures model in closure
    const localAgentNode = async (state: AgentStateType): Promise<Partial<AgentStateType>> => {
      console.log(`[Agent] Processing, message count: ${state.messages.length}`)

      const profileStr = JSON.stringify(state.userProfile || {}, null, 2)
      const systemPrompt = SYSTEM_PROMPT.replace("{userProfile}", profileStr)

      const response = await model.invoke([
        { role: "system", content: systemPrompt },
        ...state.messages
      ])

      console.log(`[Agent] Response type: ${response._getType()}`)
      console.log(`[Agent] Has tool calls: ${(response as any).tool_calls?.length > 0}`)

      return { messages: [response] }
    }

    // Build the cyclic graph
    const workflow = new StateGraph(AgentState)
      // Nodes
      .addNode("enricher", enricherNode)
      .addNode("agent", localAgentNode)
      .addNode("tools", toolNode)

      // Entry point
      .addEdge(START, "enricher")

      // Enricher -> Agent
      .addEdge("enricher", "agent")

      // Agent -> Tools or End (conditional)
      .addConditionalEdges("agent", routeAfterAgent, {
        tools: "tools",
        [END]: END
      })

      // Tools -> Agent (loop back for re-reasoning)
      .addEdge("tools", "agent")

    // Initialize checkpointer for conversation memory
    const checkpointer = new MongoDBSaver({ client, dbName })

    // Compile the graph
    const app = workflow.compile({ checkpointer })

    // Execute the graph
    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(query)],
        userProfile: {},
        activeProducts: []
      },
      {
        configurable: { thread_id }
      }
    )

    console.log(`[Deko] Final state - messages: ${finalState.messages.length}`)
    console.log(`[Deko] User profile:`, finalState.userProfile)

    // Extract the final AI response
    const lastAiMessage = [...finalState.messages]
      .reverse()
      .find(m => m._getType() === "ai" && !(m as any).tool_calls?.length)

    const responseText = lastAiMessage?.content?.toString() || "I apologize, could you please rephrase that?"

    // Get products from tool side-effect
    const products = productSearchToolInstance.getLastResults()

    console.log(`[Deko] Products found: ${products.length}`)

    // Parse response into intro and followUp
    const { intro, followUp } = parseResponse(responseText)

    return {
      intro,
      products,
      followUp,
      meta: {
        hasProducts: products.length > 0,
        searchType: products.length > 0 ? "vector" : "none",
        productCount: products.length,
        intent: "react_agent",
        detectedLanguage: finalState.language
      }
    }

  } catch (error: any) {
    console.error("[Deko] Error:", error.message)

    if (error.status === 429) {
      throw new Error("Service temporarily unavailable. Please try again in a minute.")
    }

    throw new Error(`Agent failed: ${error.message}`)
  }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

function parseResponse(text: string): { intro: string; followUp: string } {
  // Try to split response into intro and follow-up question
  const lines = text.split('\n').filter(l => l.trim())

  if (lines.length >= 2) {
    // Check if last line is a question
    const lastLine = lines[lines.length - 1]
    if (lastLine.includes('?') || /^(would|could|can|shall|do|may|what|how|which|are|is)/i.test(lastLine)) {
      return {
        intro: lines.slice(0, -1).join('\n'),
        followUp: lastLine
      }
    }
  }

  // Fallback: whole text is intro, generic follow-up
  return {
    intro: text,
    followUp: "Is there anything else I can help you with?"
  }
}

// ============================================================================
// UTILITY EXPORTS (for API compatibility)
// ============================================================================

export function getStateSummary(threadId: string): string {
  return `ReAct agent for thread ${threadId}`
}

export function resetConversation(threadId: string): void {
  console.log(`[Deko] Conversation reset requested for thread ${threadId}`)
  // Checkpointer in MongoDB handles this - state will be fresh on next call
}