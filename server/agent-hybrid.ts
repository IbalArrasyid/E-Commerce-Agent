/**
 * HYBRID AGENT - AI + LOGIC
 *
 * Architecture:
 * User Message → Intent Extraction (Small LLM) → Update State (Deterministic)
 * → Product Search (Vector/Text) → Response Generation (LLM for narrative only)
 *
 * PRINCIPLES:
 * - LLM BUKAN DATABASE
 * - LLM BUKAN STATE MANAGER
 * - State di-manage secara deterministik
 * - Vector search di application layer
 * - LLM hanya untuk narasi singkat
 */

import { MongoClient } from "mongodb"
import "dotenv/config"

// Import services
import { stateStore, buildContextString, type StateUpdate, type ConversationState } from "./services/ConversationState.js"
import { intentExtractor, type Intent } from "./services/IntentExtractor.js"
import { createProductSearchService } from "./services/ProductSearch.js"
import { responseGenerator, type GeneratedResponse } from "./services/ResponseGenerator.js"

// ============================================================================
// TYPES
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
// MAIN AGENT FUNCTION
// ============================================================================

export async function callAgent(
  client: MongoClient,
  query: string,
  thread_id: string
): Promise<AgentResponse> {
  console.log(`\n[HybridAgent] Processing message for thread ${thread_id}`)
  console.log(`[HybridAgent] User message: "${query}"`)

  // ========================================================================
  // STEP 1: Get or Create Conversation State (Deterministic)
  // ========================================================================
  let state = stateStore.getOrCreate(thread_id)
  console.log(`[HybridAgent] Current state:`, {
    filters: state.filters,
    searchQuery: state.search.query,
    resultCount: state.search.resultCount
  })

  // ========================================================================
  // STEP 2: Intent Extraction (Small LLM)
  // ========================================================================
  const intent = await intentExtractor.extract(query, {
    currentCategory: state.filters.category,
    activeFilters: state.filters,
    lastQuery: state.search.query
  })

  console.log(`[HybridAgent] Extracted intent:`, intent)

  // Update language state
  if (intent.language) {
    state = stateStore.update(thread_id, {
      type: "SET_LANGUAGE",
      value: intent.language
    })
  }

  // ========================================================================
  // STEP 3: Handle Special Intents (No LLM needed)
  // ========================================================================
  if (intent.intent === "greeting" || intent.intent === "help") {
    const greetingResponse = await responseGenerator.generate({
      language: intent.language,
      hasProducts: false,
      productCount: 0,
      products: [],
      intent: intent.intent
    })

    // Add user message to history
    stateStore.update(thread_id, {
      type: "ADD_MESSAGE",
      role: "user",
      content: query
    })

    return {
      intro: greetingResponse.intro,
      products: [],
      followUp: greetingResponse.followUp,
      meta: {
        hasProducts: false,
        searchType: "none",
        productCount: 0,
        intent: intent.intent,
        detectedLanguage: intent.language
      }
    }
  }

  // ========================================================================
  // STEP 4: Update Filters (Deterministic)
  // ========================================================================
  if (intent.intent === "filter_clear") {
    state = stateStore.update(thread_id, { type: "CLEAR_FILTERS" })
    console.log(`[HybridAgent] Filters cleared`)
  }

  // Apply filter updates from intent
  if (intent.filters) {
    if (intent.filters.category) {
      state = stateStore.update(thread_id, {
        type: "SET_FILTER",
        key: "category",
        value: intent.filters.category
      })
    }
    if (intent.filters.color) {
      state = stateStore.update(thread_id, {
        type: "SET_FILTER",
        key: "color",
        value: intent.filters.color
      })
    }
    if (intent.filters.material) {
      state = stateStore.update(thread_id, {
        type: "SET_FILTER",
        key: "material",
        value: intent.filters.material
      })
    }
    if (intent.filters.brand) {
      state = stateStore.update(thread_id, {
        type: "SET_FILTER",
        key: "brand",
        value: intent.filters.brand
      })
    }
    if (intent.filters.price_min) {
      state = stateStore.update(thread_id, {
        type: "SET_FILTER",
        key: "priceMin",
        value: intent.filters.price_min
      })
    }
    if (intent.filters.price_max) {
      state = stateStore.update(thread_id, {
        type: "SET_FILTER",
        key: "priceMax",
        value: intent.filters.price_max
      })
    }
    console.log(`[HybridAgent] Updated filters:`, state.filters)
  }

  // ========================================================================
  // STEP 5: Product Search (Deterministic Vector/Text Search)
  // ========================================================================
  const searchQuery = intent.search_query || query
  const productSearch = createProductSearchService(client)

  const searchResult = await productSearch.search({
    query: searchQuery,
    n: 10,
    filters: state.filters,
    searchType: "auto"
  })

  console.log(`[HybridAgent] Search result: ${searchResult.count} products (${searchResult.searchType})`)

  // Update state with search results
  state = stateStore.update(thread_id, {
    type: "SET_SEARCH",
    query: searchQuery,
    results: searchResult.products,
    searchType: searchResult.searchType
  })

  // Add user message to history
  stateStore.update(thread_id, {
    type: "ADD_MESSAGE",
    role: "user",
    content: query
  })

  // ========================================================================
  // STEP 6: Generate Response (LLM for narrative only)
  // ========================================================================
  const generatedResponse = await responseGenerator.generate({
    language: intent.language,
    hasProducts: searchResult.count > 0,
    productCount: searchResult.count,
    products: searchResult.products,
    searchQuery: searchQuery,
    activeFilters: state.filters,
    intent: intent.intent
  })

  console.log(`[HybridAgent] Generated response:`, generatedResponse)

  // ========================================================================
  // STEP 7: Return Structured Response
  // ========================================================================
  return {
    intro: generatedResponse.intro,
    products: searchResult.products,
    followUp: generatedResponse.followUp,
    meta: {
      hasProducts: searchResult.count > 0,
      searchType: searchResult.searchType,
      productCount: searchResult.count,
      intent: intent.intent,
      detectedLanguage: intent.language
    }
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get conversation state untuk debugging/monitoring.
 */
export function getConversationState(threadId: string): ConversationState | undefined {
  return stateStore.get(threadId)
}

/**
 * Reset conversation state.
 */
export function resetConversation(threadId: string): void {
  stateStore.delete(threadId)
  console.log(`[HybridAgent] Reset conversation for thread ${threadId}`)
}

/**
 * Get state summary untuk debugging.
 */
export function getStateSummary(threadId: string): string {
  const state = stateStore.get(threadId)
  if (!state) return "No state found"

  return buildContextString(state)
}
