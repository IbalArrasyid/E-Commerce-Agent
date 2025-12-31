/**
 * ConversationState - DETERMINISTIC STATE MANAGER
 *
 * LLM BUKAN DATABASE. LLM BUKAN STATE MANAGER.
 * State ini di-manage secara deterministik oleh application logic.
 */

// ============================================================================
// TYPES
// ============================================================================

export interface FilterState {
  category?: string      // e.g., "sofa", "meja", "kursi"
  color?: string         // e.g., "putih", "hitam", "coklat"
  material?: string      // e.g., "kayu", "kulit", "kain"
  priceMin?: number      // min price in IDR
  priceMax?: number      // max price in IDR
  brand?: string         // e.g., "IKEA", "Ace Hardware"
}

export interface SearchState {
  query?: string         // last search query (full, reformulated)
  baseQuery?: string     // base product query (without refinements)
  results?: ProductItem[] // search results
  resultCount?: number   // number of results
  searchType?: "vector" | "text" | "none"
}

export interface ConversationHistory {
  role: "user" | "assistant"
  content: string
  timestamp?: Date
}

export interface ConversationState {
  threadId: string
  filters: FilterState
  search: SearchState
  history: ConversationHistory[]
  language: "id" | "en"
  createdAt: Date
  updatedAt: Date
}

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

// ============================================================================
// STATE TRANSITIONS (DETERMINISTIC)
// ============================================================================

export type StateUpdate =
  | { type: "SET_CATEGORY"; value: string }
  | { type: "SET_FILTER"; key: keyof FilterState; value: string | number | undefined }
  | { type: "SET_LANGUAGE"; value: "id" | "en" }
  | { type: "CLEAR_FILTERS" }
  | { type: "SET_SEARCH"; query: string; baseQuery?: string; results: ProductItem[]; searchType: "vector" | "text" | "none" }
  | { type: "ADD_MESSAGE"; role: "user" | "assistant"; content: string }
  | { type: "SET_BASE_QUERY"; value: string }

/**
 * Apply state update secara deterministik.
 * Tidak ada LLM di sini - pure logic.
 */
export function applyStateUpdate(
  state: ConversationState,
  update: StateUpdate
): ConversationState {
  const newState = { ...state }
  const now = new Date()

  switch (update.type) {
    case "SET_CATEGORY":
      newState.filters = { ...state.filters, category: update.value }
      break

    case "SET_FILTER":
      if (update.value === undefined) {
        // Remove filter if value is undefined
        const { [update.key]: _, ...rest } = newState.filters
        newState.filters = rest
      } else {
        newState.filters = { ...state.filters, [update.key]: update.value }
      }
      break

    case "SET_LANGUAGE":
      newState.language = update.value
      break

    case "CLEAR_FILTERS":
      newState.filters = {}
      break

    case "SET_SEARCH":
      newState.search = {
        query: update.query,
        baseQuery: update.baseQuery,
        results: update.results,
        resultCount: update.results.length,
        searchType: update.searchType
      }
      break

    case "SET_BASE_QUERY":
      if (newState.search) {
        newState.search = { ...newState.search, baseQuery: update.value }
      }
      break

    case "ADD_MESSAGE":
      newState.history = [
        ...state.history,
        {
          role: update.role,
          content: update.content,
          timestamp: now
        }
      ]
      break
  }

  newState.updatedAt = now
  return newState
}

// ============================================================================
// STATE INITIALIZATION
// ============================================================================

export function createInitialState(threadId: string): ConversationState {
  return {
    threadId,
    filters: {},
    search: {},
    history: [],
    language: "id", // default
    createdAt: new Date(),
    updatedAt: new Date()
  }
}

// ============================================================================
// STATE QUERIES (DETERMINISTIC)
// ============================================================================

export interface StateSummary {
  hasActiveFilters: boolean
  activeFilters: Partial<FilterState>
  hasSearchResults: boolean
  resultCount: number
  currentCategory?: string
  lastQuery?: string
}

/**
 * Get summary of current state untuk prompt injection.
 * Deterministic - tidak ada LLM.
 */
export function getStateSummary(state: ConversationState): StateSummary {
  const hasActiveFilters = Object.keys(state.filters).length > 0
  const hasSearchResults = (state.search.resultCount || 0) > 0

  return {
    hasActiveFilters,
    activeFilters: state.filters,
    hasSearchResults,
    resultCount: state.search.resultCount || 0,
    currentCategory: state.filters.category,
    lastQuery: state.search.query
  }
}

/**
 * Build context string untuk LLM prompt.
 * Ini adalah satu-satunya bridge antara state dan LLM.
 */
export function buildContextString(state: ConversationState): string {
  const summary = getStateSummary(state)
  const parts: string[] = []

  parts.push(`Language: ${state.language === "id" ? "Indonesian" : "English"}`)

  if (summary.currentCategory) {
    parts.push(`Current category: ${summary.currentCategory}`)
  }

  if (summary.hasActiveFilters) {
    const filterParts: string[] = []
    if (summary.activeFilters.color) filterParts.push(`color: ${summary.activeFilters.color}`)
    if (summary.activeFilters.material) filterParts.push(`material: ${summary.activeFilters.material}`)
    if (summary.activeFilters.priceMin) filterParts.push(`price min: ${summary.activeFilters.priceMin}`)
    if (summary.activeFilters.priceMax) filterParts.push(`price max: ${summary.activeFilters.priceMax}`)
    if (summary.activeFilters.brand) filterParts.push(`brand: ${summary.activeFilters.brand}`)
    if (filterParts.length > 0) {
      parts.push(`Active filters: ${filterParts.join(", ")}`)
    }
  }

  if (summary.lastQuery) {
    parts.push(`Last search: "${summary.lastQuery}"`)
  }

  if (summary.hasSearchResults) {
    parts.push(`Found ${summary.resultCount} products`)
  }

  return parts.join("\n") || "No active context"
}

// ============================================================================
// IN-MEMORY STATE STORE (simple untuk production, ganti dengan Redis/DB)
// ============================================================================

class StateStore {
  private states: Map<string, ConversationState> = new Map()

  get(threadId: string): ConversationState | undefined {
    return this.states.get(threadId)
  }

  set(threadId: string, state: ConversationState): void {
    this.states.set(threadId, state)
  }

  getOrCreate(threadId: string): ConversationState {
    let state = this.states.get(threadId)
    if (!state) {
      state = createInitialState(threadId)
      this.states.set(threadId, state)
    }
    return state
  }

  update(threadId: string, update: StateUpdate): ConversationState {
    const currentState = this.getOrCreate(threadId)
    const newState = applyStateUpdate(currentState, update)
    this.states.set(threadId, newState)
    return newState
  }

  delete(threadId: string): void {
    this.states.delete(threadId)
  }

  // Cleanup old states (optional)
  cleanup(olderThanMs: number = 3600000): void { // default 1 hour
    const cutoff = new Date(Date.now() - olderThanMs)
    for (const [threadId, state] of this.states.entries()) {
      if (state.updatedAt < cutoff) {
        this.states.delete(threadId)
      }
    }
  }
}

// Export singleton instance
export const stateStore = new StateStore()

// Auto-cleanup every 30 minutes
if (typeof setInterval !== "undefined") {
  setInterval(() => stateStore.cleanup(), 1800000)
}