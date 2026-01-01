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
  slug?: string  // Clickable link slug derived from product name
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
 * Filter products to only include those that match the search query.
 * This ensures users only see relevant products.
 */
function filterRelevantProducts(products: ProductItem[], searchQuery: string): ProductItem[] {
  if (!searchQuery || products.length === 0) {
    return products
  }

  const queryTerms = searchQuery.toLowerCase().split(/\s+/).filter(t => t.length > 2)

  return products.filter(product => {
    const productText = `${product.item_name} ${product.item_description || ''} ${product.categories?.join(' ') || ''}`.toLowerCase()

    // Check if at least one query term matches the product
    const hasMatch = queryTerms.some(term => productText.includes(term))

    if (!hasMatch) {
      console.log(`[HybridAgent] Filtering out irrelevant: "${product.item_name}"`)
    }

    return hasMatch
  })
}

/**
 * Detect if user is starting a new search (different category).
 * Fast rule-based detection.
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
    "sofas", "sectional & corner sofas", "cleopatra day beds",
    "arm chairs", "dining chairs",
    "dining table", "side table",
    "coffee table", "bench",
    "ottoman & pouf", "decorative stools",
    "console table", "chest of drawers",
    "buffet sideboards", "tv stand", "room dividers",
    "beds", "headboards", "bedside nightstand",
    "make up dressing tables", "bed benchs",
    "bar chairs stools", "trolleys bar carts",
    "study tables", "study chairs",
    "bookcases", "ceiling light",
    "floor lamp", "table lamp",
    "vase ceramic jars", "photo frames",
    "bowls trays", "books bookends",
    "decor arts sculptures", "candle holders",
    "flower arrangements", "wall arts",
    "mirrors", "clocks",
    "square cushions", "round cushions",
    "rectangle cushions", "drapery fabrics",
    "promotional fabrics", "curtains",
    "sheer", "blackout curtains",
    "neutral rugs", "geometric rugs",
    "classical rugs", "art decor rugs",
    "abstract rugs", "plain rugs",
    "roller blinds"
  ]

  // CHECK 1: User mentions a different product category → NEW SEARCH
  for (const cat of categories) {
    if (msg.includes(cat) && !base.includes(cat)) {
      console.log(`[detectNewSearch] Different category mentioned: ${cat}`)
      return true
    }
  }

  // CHECK 2: Intent has a different category → NEW SEARCH
  if (intent.filters?.category && intent.filters.category !== base) {
    console.log(`[detectNewSearch] Intent has different category: ${intent.filters.category}`)
    return true
  }

  // CHECK 3: New search keywords WITHOUT attributes (color/material/price)
  // Only consider new search if user is truly searching for different product
  const newSearchKeywords = [
    "find table", "find chair", "find cabinet", "find sofa", "find bed",
    "show me", "looking for", "search for"
  ]

  for (const kw of newSearchKeywords) {
    if (msg.includes(kw)) {
      console.log(`[detectNewSearch] New search keyword: ${kw}`)
      return true
    }
  }

  // CHECK 4: If message contains attribute (color/material/price), treat as CONTINUATION
  // NOT a new search
  const attributes = ["white", "black", "brown",
    "red", "blue", "green", "grey", "beige", "cream",
    "wood", "wooden", "leather", "fabric", "metal", "glass",
    "cheap", "expensive", "budget", "premium", "modern", "minimalist", "japandi", "scandinavian"]

  const hasAttribute = attributes.some(attr => msg.includes(attr))

  if (hasAttribute) {
    console.log(`[detectNewSearch] Has attribute, treating as continuation`)
    return false  // This is continuation, NOT a new search
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
  let intent = await intentExtractor.extract(query, {
    currentCategory: state.filters.category,
    activeFilters: state.filters,
    lastQuery: state.search.query,
    lastIntent: state.lastIntent,
    lastFaqTopic: state.lastFaqTopic
  })

  console.log(`[HybridAgent] Extracted intent:`, intent)

  // ========================================================================
  // STEP 2.5: Smart Follow-up Detection
  // ========================================================================
  // Detect when user gives short affirmative responses like "iya", "ya", "ok"
  // and determine what they're responding to based on conversation context
  const isAffirmativeResponse = /^(iya+|ya+|yes|ok|oke|okay|mau|boleh|tentu|sure|please|tolong|ingin( tahu)?|bisa|dong)$/i.test(query.trim())
  const isShortMessage = query.trim().length < 20

  if (isAffirmativeResponse && isShortMessage && intent.intent === 'unknown') {
    console.log(`[HybridAgent] Detected affirmative short response, checking context...`)
    console.log(`[HybridAgent] Last intent: ${state.lastIntent}, has products: ${(state.search.resultCount || 0) > 0}`)

    // Case 1: User responds "iya" after FAQ about location -> give hours
    if (state.lastIntent === 'faq_info' && state.lastFaqTopic === 'location') {
      intent = { intent: 'faq_info', faq_topic: 'hours', language: intent.language }
      console.log(`[HybridAgent] Follow-up: FAQ location -> hours`)
    }
    // Case 2: User responds "iya" after product search -> they want more product info
    else if (state.lastIntent === 'search' && (state.search.resultCount || 0) > 0) {
      // User wants more info about the product shown - generate helpful response
      intent = { intent: 'product_info', language: intent.language }
      console.log(`[HybridAgent] Follow-up: Product search -> want more info`)
    }
    // Case 3: User responds after other FAQ
    else if (state.lastIntent === 'faq_info') {
      // Default to help if we don't know what they want
      intent = { intent: 'help', language: intent.language }
      console.log(`[HybridAgent] Follow-up: FAQ -> help`)
    }
    // Default: treat as wanting to explore products
    else {
      intent = { intent: 'help', language: intent.language }
      console.log(`[HybridAgent] Follow-up: Default -> help`)
    }
  }

  // Update language state
  if (intent.language) {
    state = stateStore.update(thread_id, {
      type: "SET_LANGUAGE",
      value: intent.language
    })
  }

  // ========================================================================
  // STEP 2.6: Language Check - Only English supported
  // ========================================================================
  if (intent.language === "id") {
    // Add user message to history
    stateStore.update(thread_id, {
      type: "ADD_MESSAGE",
      role: "user",
      content: query
    })

    return {
      intro: "Sorry, I can only communicate in English. Please type your message in English and I'll be happy to help you find the perfect furniture for your home! 🏠",
      products: [],
      followUp: "What are you looking for today?",
      meta: {
        hasProducts: false,
        searchType: "none",
        productCount: 0,
        intent: "language_unsupported",
        detectedLanguage: intent.language
      }
    }
  }

  // ========================================================================
  // STEP 3: Handle Special Intents (No LLM needed)
  // ========================================================================
  if (intent.intent === "greeting" || intent.intent === "help" || intent.intent === "faq_info" || intent.intent === "product_info") {

    // Handle product_info specially - show info about previously searched products
    if (intent.intent === "product_info" && state.search.results && state.search.results.length > 0) {
      const product = state.search.results[0]
      const productInfo = `
Berikut info detail produk ${product.item_name}:

📦 **Dimensi**: ${product.item_description.match(/\d+\s*x\s*\d+\s*x\s*\d+\s*(cm|mm)?/i)?.[0] || 'Silakan hubungi CS untuk dimensi'}
🪑 **Bahan**: ${product.item_description.match(/(kayu|leather|kulit|fabric|kain|glass|kaca|metal|besi|rotan)/i)?.[0] || 'Lihat deskripsi produk'}
💰 **Harga**: Rp ${product.prices?.[0]?.price?.toLocaleString('id-ID') || '-'}
🏷️ **Brand**: ${product.brand || 'Home Decor Indonesia'}

Untuk pemesanan atau tanya lebih lanjut, silakan klik produk atau hubungi CS kami! 😊
      `.trim()

      // Track intent
      stateStore.update(thread_id, {
        type: "SET_LAST_INTENT",
        intent: intent.intent
      })

      stateStore.update(thread_id, {
        type: "ADD_MESSAGE",
        role: "user",
        content: query
      })

      return {
        intro: productInfo,
        products: state.search.results.slice(0, 3), // Show products again
        followUp: "Ada yang lain yang bisa saya bantu? 🛋️",
        meta: {
          hasProducts: true,
          searchType: "none",
          productCount: state.search.results.length,
          intent: intent.intent,
          detectedLanguage: intent.language
        }
      }
    }

    const specialResponse = await responseGenerator.generate({
      language: intent.language,
      hasProducts: false,
      productCount: 0,
      products: [],
      intent: intent.intent,
      faqTopic: intent.faq_topic
    })

    // Track the last intent for follow-up detection
    stateStore.update(thread_id, {
      type: "SET_LAST_INTENT",
      intent: intent.intent,
      faqTopic: intent.faq_topic
    })

    // Add user message to history
    stateStore.update(thread_id, {
      type: "ADD_MESSAGE",
      role: "user",
      content: query
    })

    return {
      intro: specialResponse.intro,
      products: [],
      followUp: specialResponse.followUp,
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

  // Track last intent as 'search' for follow-up detection
  stateStore.update(thread_id, {
    type: "SET_LAST_INTENT",
    intent: "search"
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
  // STEP 7: Return Structured Response (with filtered relevant products)
  // ========================================================================

  // Filter to only include products that match the search query
  const relevantProducts = filterRelevantProducts(searchResult.products, searchQuery)

  console.log(`[HybridAgent] Final products: ${relevantProducts.length} (from ${searchResult.count})`)

  return {
    intro: generatedResponse.intro,
    products: relevantProducts,
    followUp: generatedResponse.followUp,
    meta: {
      hasProducts: relevantProducts.length > 0,
      searchType: searchResult.searchType,
      productCount: relevantProducts.length,
      intent: intent.intent,
      detectedLanguage: intent.language
    }
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get conversation state for debugging/monitoring.
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
 * Get state summary for debugging.
 */
export function getStateSummary(threadId: string): string {
  const state = stateStore.get(threadId)
  if (!state) return "No state found"

  return buildContextString(state)
}