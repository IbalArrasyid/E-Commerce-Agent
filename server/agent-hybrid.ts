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
import { stateStore, buildContextString, type StateUpdate, type ConversationState } from "./services/ConversationState"
import { intentExtractor, type Intent } from "./services/IntentExtractor"
import { createProductSearchService } from "./services/ProductSearch"
import { responseGenerator, type GeneratedResponse } from "./services/ResponseGenerator"
import { queryReformulator, type ReformulatedQuery } from "./services/QueryReformulator"

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
// HELPER FUNCTIONS
// ============================================================================

/**
 * Detect apakah user memulai pencarian baru (kategori berbeda).
 * Rule-based detection yang cepat.
 */
async function detectNewSearch(
  userMessage: string,
  currentBaseQuery: string,
  intent: Intent
): Promise<boolean> {

  const msg = userMessage.toLowerCase().trim()
  const base = currentBaseQuery.toLowerCase()

  // Daftar kategori produk
  const categories = [
    "sofa", "settee", "couch",
    "meja", "table",
    "kursi", "chair",
    "lemari", "cabinet",
    "rak", "shelf",
    "buffet", "sideboard",
    "bed", "kasur", "mattress",
    "coffee table", "meja tamu", "meja kopi"
  ]

  // CEK 1: User menyebut kategori produk berbeda → NEW SEARCH
  for (const cat of categories) {
    if (msg.includes(cat) && !base.includes(cat)) {
      console.log(`[detectNewSearch] Different category mentioned: ${cat}`)
      return true
    }
  }

  // CEK 2: Intent menyebut category yang berbeda → NEW SEARCH
  if (intent.filters?.category && intent.filters.category !== base) {
    console.log(`[detectNewSearch] Intent has different category: ${intent.filters.category}`)
    return true
  }

  // CEK 3: Keyword pencarian baru TANPA attribute (warna/material/price)
  // Hanya anggap new search jika user benar-benar mencari produk berbeda
  const newSearchKeywords = [
    "cari meja", "cari kursi", "cari lemari", "cari rak", "cari bed",
    "find table", "find chair", "find cabinet",
    "show me", "tampilkan"
  ]

  for (const kw of newSearchKeywords) {
    if (msg.includes(kw)) {
      console.log(`[detectNewSearch] New search keyword: ${kw}`)
      return true
    }
  }

  // CEK 4: Jika message mengandung attribute (warna/material/price), ini CONTINUATION
  // BUKAN new search, meskipun ada "apa ada" atau "ada"
  const attributes = ["putih", "white", "hitam", "black", "coklat", "brown",
                      "merah", "red", "biru", "blue", "hijau", "green",
                      "kayu", "wood", "kulit", "leather", "kain", "fabric",
                      "murah", "cheap", "mahal", "expensive"]

  const hasAttribute = attributes.some(attr => msg.includes(attr))

  if (hasAttribute) {
    console.log(`[detectNewSearch] Has attribute, treating as continuation`)
    return false  // Ini continuation, BUKAN new search
  }

  console.log(`[detectNewSearch] No new search detected, treating as continuation`)
  return false
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
  // STEP 4: Query Reformulation (Smart continuation handling)
  // ========================================================================
  let reformulatedQuery: ReformulatedQuery

  // Debug: check current state
  console.log(`[HybridAgent] State check - baseQuery: "${state.search.baseQuery}", search.query: "${state.search.query}"`)

  // Tentukan base query (gunakan yang sudah ada atau extract dari pesan pertama)
  const currentBaseQuery = state.search.baseQuery || intent.search_query || query
  console.log(`[HybridAgent] Determined currentBaseQuery: "${currentBaseQuery}"`)

  // Cek apakah ini adalah pencarian baru (user mention kategori berbeda)
  const isNewSearch = await detectNewSearch(query, currentBaseQuery, intent)

  if (isNewSearch) {
    // Reset base query untuk pencarian baru
    console.log(`[HybridAgent] New search detected, resetting base query`)
    reformulatedQuery = {
      query: intent.search_query || query,
      isContinuation: false,
      isNewSearch: true,
      detectedAttributes: {}
    }
    // Update base query
    state = stateStore.update(thread_id, {
      type: "SET_BASE_QUERY",
      value: intent.search_query || query
    })
    console.log(`[HybridAgent] Set new baseQuery to: "${intent.search_query || query}"`)
  } else if (state.search.baseQuery) {
    // Gunakan QueryReformulator untuk continuation query
    console.log(`[HybridAgent] Using QueryReformulator with baseQuery: "${state.search.baseQuery}"`)
    reformulatedQuery = await queryReformulator.reformulate(query, {
      baseQuery: state.search.baseQuery,
      lastSearchQuery: state.search.query,
      activeFilters: state.filters,
      language: intent.language || state.language
    })
    console.log(`[HybridAgent] Reformulated query: "${reformulatedQuery.query}"`)
  } else {
    // Pertama kali, belum ada base query
    console.log(`[HybridAgent] First query, setting baseQuery`)
    reformulatedQuery = {
      query: intent.search_query || query,
      isContinuation: false,
      isNewSearch: false,
      detectedAttributes: {}
    }
    // Set base query dari query pertama
    const baseValue = intent.search_query || query
    state = stateStore.update(thread_id, {
      type: "SET_BASE_QUERY",
      value: baseValue
    })
    console.log(`[HybridAgent] Set initial baseQuery to: "${baseValue}"`)
  }

  // ========================================================================
  // STEP 5: Update Filters (Deterministic)
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
  // STEP 6: Product Search (Deterministic Vector/Text Search)
  // ========================================================================
  const searchQuery = reformulatedQuery.query
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
    baseQuery: state.search.baseQuery,
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