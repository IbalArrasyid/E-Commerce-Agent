/**
 * LANGGRAPH AGENT WITH AGENTIC REASONING
 *
 * Multi-node graph architecture for:
 * 1. Intent extraction + preference detection
 * 2. Preference management across turns
 * 3. RAG-based product search
 * 4. Context-aware response generation
 *
 * Flow: User Message → Intent Node → [Preference Node] → Search Node → Response Node
 */

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { ChatOpenAI } from "@langchain/openai"
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages"
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts"
import { StateGraph, Annotation, END } from "@langchain/langgraph"
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb"
import { MongoClient } from "mongodb"
import "dotenv/config"

// Import services
import { intentExtractor, type Intent } from "./services/IntentExtractor"
import {
  preferenceManager,
  type UserPreferences,
  type ConversationContext,
  type PreferenceUpdate
} from "./services/PreferenceManager"
import { createProductSearchService } from "./services/ProductSearch"
import { responseGenerator } from "./services/ResponseGenerator"

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface ProductItem {
  item_id: string
  item_name: string
  item_description: string
  brand: string
  prices: Array<{ variant?: string; price: number; currency?: string }>
  user_reviews?: { rating?: number; count?: number }
  categories: string[]
  images: string[]
  slug?: string
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
    reasoning?: string  // Agent's reasoning trace
  }
}

// ============================================================================
// STATE DEFINITION (LangGraph Annotation)
// ============================================================================

const AgentState = Annotation.Root({
  // Message history
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y)
  }),

  // Current user message
  userMessage: Annotation<string>({
    reducer: (_, y) => y
  }),

  // Extracted intent
  intent: Annotation<Intent | null>({
    reducer: (_, y) => y
  }),

  // User preferences (persisted across turns)
  preferences: Annotation<UserPreferences>({
    reducer: (existing, updates) => {
      if (!updates || Object.keys(updates).length === 0) return existing || {}
      // Merge updates into existing preferences
      return { ...(existing || {}), ...updates }
    }
  }),

  // Conversation context (for "this one" references)
  context: Annotation<ConversationContext>({
    reducer: (_, y) => y
  }),

  // Search results
  searchResults: Annotation<ProductItem[]>({
    reducer: (_, y) => y
  }),

  // Response (final output)
  response: Annotation<AgentResponse | null>({
    reducer: (_, y) => y
  }),

  // Language
  language: Annotation<"id" | "en">({
    reducer: (existing, y) => y ?? existing ?? "en"
  }),

  // Reasoning trace
  reasoning: Annotation<string[]>({
    reducer: (x, y) => [...(x || []), ...(y || [])]
  })
})

type AgentStateType = typeof AgentState.State

// ============================================================================
// GRAPH NODES
// ============================================================================

/**
 * Intent Node: Extract user intent and update preferences
 */
async function intentNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const userMessage = state.userMessage
  const existingPrefs = state.preferences || {}
  const context = state.context || { lastRecommendedProducts: [], turnCount: 0 }

  console.log(`[IntentNode] Processing: "${userMessage}"`)
  console.log(`[IntentNode] Existing preferences:`, existingPrefs)

  // Extract intent using IntentExtractor
  const intent = await intentExtractor.extract(userMessage, {
    currentCategory: existingPrefs.category,
    activeFilters: existingPrefs,
    lastQuery: context.lastSearchQuery
  })

  console.log(`[IntentNode] Extracted intent:`, intent)

  // Build preference updates from intent
  const prefUpdates: PreferenceUpdate = {}

  if (intent.filters?.color) prefUpdates.color = intent.filters.color
  if (intent.filters?.price_max) prefUpdates.budget_max = intent.filters.price_max
  if (intent.filters?.price_min) prefUpdates.budget_min = intent.filters.price_min
  if (intent.filters?.category) prefUpdates.category = intent.filters.category

  // Extract additional preferences from message
  const sizeFromMessage = preferenceManager.detectSizeFromMessage(userMessage)
  if (sizeFromMessage) prefUpdates.size = sizeFromMessage

  const roomFromMessage = preferenceManager.detectRoomFromMessage(userMessage)
  if (roomFromMessage) prefUpdates.roomType = roomFromMessage

  // Handle style from intent preferences
  if (intent.preferences?.style) {
    prefUpdates.style = intent.preferences.style as UserPreferences["style"]
  }

  // Reasoning trace
  const reasoning: string[] = []
  reasoning.push(`Intent detected: ${intent.intent}`)
  if (Object.keys(prefUpdates).length > 0) {
    reasoning.push(`Preferences extracted: ${JSON.stringify(prefUpdates)}`)
  }

  return {
    intent,
    preferences: prefUpdates,
    language: intent.language,
    reasoning
  }
}

/**
 * Context Node: Handle context references ("this one", etc.)
 */
async function contextNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const intent = state.intent
  const context = state.context || { lastRecommendedProducts: [], turnCount: 0 }
  const userMessage = state.userMessage

  const reasoning: string[] = []

  // Check for context reference
  if (intent?.context_reference || intent?.color_variant_query) {
    const variantInfo = preferenceManager.detectColorVariantQuery(userMessage, context)

    if (variantInfo.isVariantQuery && variantInfo.requestedColor) {
      reasoning.push(`Detected color variant query for "${variantInfo.baseProduct}"`)
      reasoning.push(`Requested color: ${variantInfo.requestedColor}`)

      // Update preferences with the new color request
      return {
        preferences: { color: variantInfo.requestedColor },
        reasoning
      }
    }

    // Try to resolve basic context reference
    const resolved = preferenceManager.resolveContextReference(context, userMessage)
    if (resolved) {
      reasoning.push(`Resolved context reference to product: ${resolved.item_name}`)
    }
  }

  return { reasoning }
}

/**
 * Search Node: Perform RAG search using ProductSearchService
 */
async function searchNode(
  state: AgentStateType,
  client: MongoClient
): Promise<Partial<AgentStateType>> {
  const intent = state.intent
  const preferences = state.preferences || {}

  const reasoning: string[] = []

  // Skip search for non-search intents
  if (!intent || !["search", "context_query", "filter_add"].includes(intent.intent)) {
    reasoning.push(`Skipping search - intent is "${intent?.intent}"`)
    return { searchResults: [], reasoning }
  }

  // Build search query
  let searchQuery = intent.search_query || ""

  // If no search query but we have category, use that
  if (!searchQuery && preferences.category) {
    searchQuery = preferences.category
  }

  // Add style/room context to query for better semantic search
  if (preferences.style) {
    searchQuery = `${searchQuery} ${preferences.style}`.trim()
  }
  if (preferences.roomType) {
    searchQuery = `${searchQuery} for ${preferences.roomType}`.trim()
  }

  reasoning.push(`Search query: "${searchQuery}"`)

  // Build filters from preferences
  const filters = preferenceManager.buildSearchFilters(preferences)
  reasoning.push(`Filters applied: ${JSON.stringify(filters)}`)

  // Perform search
  const productSearch = createProductSearchService(client)
  const searchResult = await productSearch.search({
    query: searchQuery,
    n: 10,
    filters,
    searchType: "auto"
  })

  console.log(`[SearchNode] Found ${searchResult.count} products`)
  reasoning.push(`Found ${searchResult.count} products via ${searchResult.searchType} search`)

  // Update context with recommended products
  const updatedContext: ConversationContext = {
    lastRecommendedProducts: searchResult.products.slice(0, 5).map(p => ({
      item_id: p.item_id,
      item_name: p.item_name,
      category: p.categories?.[0]
    })),
    lastCategory: preferences.category,
    lastSearchQuery: searchQuery,
    turnCount: (state.context?.turnCount || 0) + 1
  }

  return {
    searchResults: searchResult.products,
    context: updatedContext,
    reasoning
  }
}

/**
 * Response Node: Generate natural language response
 */
async function responseNode(state: AgentStateType): Promise<Partial<AgentStateType>> {
  const intent = state.intent
  const searchResults = state.searchResults || []
  const preferences = state.preferences || {}
  const language = state.language || "en"

  const reasoning: string[] = []

  // Generate response using ResponseGenerator
  const generatedResponse = await responseGenerator.generate({
    language,
    hasProducts: searchResults.length > 0,
    productCount: searchResults.length,
    products: searchResults,
    searchQuery: state.context?.lastSearchQuery,
    activeFilters: preferences,
    intent: intent?.intent,
    faqTopic: intent?.faq_topic
  })

  reasoning.push(`Generated response with ${searchResults.length} products`)

  // Build final response
  const response: AgentResponse = {
    intro: generatedResponse.intro,
    products: searchResults.slice(0, 5),
    followUp: generatedResponse.followUp,
    meta: {
      hasProducts: searchResults.length > 0,
      searchType: searchResults.length > 0 ? "vector" : "none",
      productCount: searchResults.length,
      intent: intent?.intent,
      detectedLanguage: language,
      reasoning: state.reasoning?.join(" → ")
    }
  }

  return { response, reasoning }
}

// ============================================================================
// ROUTER FUNCTION
// ============================================================================

function routeByIntent(state: AgentStateType): string {
  const intent = state.intent

  if (!intent) {
    return "response"
  }

  switch (intent.intent) {
    case "greeting":
    case "help":
    case "faq_info":
      // Skip search for these intents
      return "response"

    case "context_query":
      // Handle context reference first
      return "context"

    case "search":
    case "filter_add":
    default:
      // Go to search node
      return "search"
  }
}

// ============================================================================
// MAIN AGENT FUNCTION
// ============================================================================

export async function callAgent(
  client: MongoClient,
  query: string,
  thread_id: string
): Promise<AgentResponse> {
  console.log(`\n[Agent] Processing message for thread ${thread_id}`)
  console.log(`[Agent] User message: "${query}"`)

  try {
    const dbName = "admin_db"

    // Build the graph
    const workflow = new StateGraph(AgentState)
      // Add nodes
      .addNode("intent", intentNode)
      .addNode("context", contextNode)
      .addNode("search", async (state) => searchNode(state, client))
      .addNode("response", responseNode)

      // Define edges
      .addEdge("__start__", "intent")
      .addConditionalEdges("intent", routeByIntent, {
        response: "response",
        context: "context",
        search: "search"
      })
      .addEdge("context", "search")
      .addEdge("search", "response")
      .addEdge("response", END)

    // Initialize checkpointer for conversation memory
    const checkpointer = new MongoDBSaver({ client, dbName })

    // Compile the graph
    const app = workflow.compile({ checkpointer })

    // Execute the graph
    const finalState = await app.invoke(
      {
        userMessage: query,
        messages: [new HumanMessage(query)]
      },
      {
        configurable: { thread_id }
      }
    )

    console.log(`[Agent] Final reasoning:`, finalState.reasoning)
    console.log(`[Agent] Final preferences:`, finalState.preferences)

    // Return the response
    if (finalState.response) {
      return finalState.response
    }

    // Fallback response
    return {
      intro: "I apologize, but I couldn't process your request. Could you please rephrase?",
      products: [],
      followUp: "What are you looking for today?",
      meta: {
        hasProducts: false,
        searchType: "none",
        productCount: 0
      }
    }

  } catch (error: any) {
    console.error("[Agent] Error:", error.message)

    if (error.status === 429) {
      throw new Error("Service temporarily unavailable due to rate limits. Please try again in a minute.")
    } else if (error.status === 401) {
      throw new Error("Authentication failed. Please check your API configuration.")
    } else {
      throw new Error(`Agent failed: ${error.message}`)
    }
  }
}

// ============================================================================
// UTILITY EXPORTS
// ============================================================================

export { AgentState }
export type { AgentStateType }