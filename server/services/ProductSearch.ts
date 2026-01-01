/**
 * ProductSearch - DETERMINISTIC VECTOR SEARCH
 *
 * Tugas:
 * 1. Vector search menggunakan MongoDB Atlas Vector Search
 * 2. Fallback ke text search jika vector search gagal
 * 3. Apply filters secara deterministik
 * 4. NO LLM - pure database operations
 *
 * LLM BUKAN DATABASE.
 */

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import { MongoClient, Collection } from "mongodb"
import type { FilterState } from "./ConversationState.js"

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
  slug: string  // Clickable link slug derived from product name
}

export interface SearchResult {
  products: ProductItem[]
  searchType: "vector" | "text" | "none"
  query: string
  count: number
  appliedFilters: Partial<FilterState>
}

export interface SearchOptions {
  query: string
  n?: number          // number of results
  filters?: FilterState
  searchType?: "vector" | "auto" // force vector or auto-select
}

// ============================================================================
// PRODUCT SEARCH SERVICE
// ============================================================================

export class ProductSearchService {
  private collection: Collection
  private dbName = "admin_db"
  private collectionName = "items"
  private vectorIndexName = "vector_index"
  private embeddings: GoogleGenerativeAIEmbeddings

  constructor(mongoClient: MongoClient) {
    this.collection = mongoClient.db(this.dbName).collection(this.collectionName)
    this.embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "text-embedding-004",
    })
  }

  /**
   * Main search method - DETERMINISTIC.
   * Tidak ada LLM reasoning, murni vector search + filter application.
   */
  async search(options: SearchOptions): Promise<SearchResult> {
    const { query, n = 10, filters, searchType = "auto" } = options

    console.log(`[ProductSearch] Query: "${query}", Filters:`, filters)

    try {
      // Build search query dengan filters
      const augmentedQuery = this.buildAugmentedQuery(query, filters)

      // Coba vector search dulu
      let vectorResults: Array<[any, number]> | null = null

      if (searchType === "vector" || searchType === "auto") {
        vectorResults = await this.vectorSearch(augmentedQuery, n)
      }

      // Apply filters secara deterministically
      let filteredProducts = vectorResults
        ? this.applyFiltersToVectorResults(vectorResults, filters)
        : []

      // Fallback ke text search jika vector tidak return results
      if (filteredProducts.length === 0) {
        console.log("[ProductSearch] Vector search empty, falling back to text search")
        filteredProducts = await this.textSearch(augmentedQuery, n, filters)
      }

      // Format hasil
      let products = this.formatProducts(filteredProducts.slice(0, n * 2)) // Get more to filter

      // Apply relevance filtering to remove non-matching product types
      // (e.g., filter out coffee table when searching for dining table)
      products = this.filterByRelevance(products, query)

      // Limit to requested count
      products = products.slice(0, n)

      console.log(`[ProductSearch] After relevance filter: ${products.length} products`)

      return {
        products,
        searchType: vectorResults && vectorResults.length > 0 ? "vector" : "text",
        query,
        count: products.length,
        appliedFilters: filters || {}
      }

    } catch (error) {
      console.error("[ProductSearch] Error:", error)

      // Ultimate fallback: text search only
      const fallbackResults = await this.textSearch(query, n, filters)
      return {
        products: this.formatProducts(fallbackResults.slice(0, n)),
        searchType: "text",
        query,
        count: fallbackResults.length,
        appliedFilters: filters || {}
      }
    }
  }

  /**
   * Build augmented query dengan filter terms.
   * Ini meningkatkan vector search accuracy.
   */
  private buildAugmentedQuery(query: string, filters?: FilterState): string {
    const parts: string[] = [query]

    if (filters) {
      if (filters.category) parts.push(filters.category)
      if (filters.color) parts.push(`warna ${filters.color}`)
      if (filters.material) parts.push(filters.material)
      if (filters.brand) parts.push(filters.brand)
    }

    return parts.join(" ")
  }

  /**
   * Vector search menggunakan MongoDB Atlas Vector Search.
   * Deterministic - input yang sama = output yang sama.
   */
  private async vectorSearch(
    query: string,
    n: number
  ): Promise<Array<[any, number]> | null> {
    try {
      const vectorStore = new MongoDBAtlasVectorSearch(
        this.embeddings,
        {
          collection: this.collection,
          indexName: this.vectorIndexName,
          textKey: "embedding_text",
          embeddingKey: "embedding",
        }
      )

      const results = await vectorStore.similaritySearchWithScore(query, n)
      console.log(`[ProductSearch] Vector search returned ${results.length} results`)

      return results
    } catch (error) {
      console.error("[ProductSearch] Vector search error:", error)
      return null
    }
  }

  /**
   * Apply filters ke vector search results.
   * Post-processing filtering - deterministic.
   */
  private applyFiltersToVectorResults(
    results: Array<[any, number]>,
    filters?: FilterState
  ): any[] {
    if (!filters || Object.keys(filters).length === 0) {
      return results
    }

    return results.filter(([doc, score]) => {
      const metadata = doc.metadata || doc
      return this.matchesFilters(metadata, filters)
    })
  }

  /**
   * Check jika product matches filters.
   * Pure logic - deterministic.
   */
  private matchesFilters(product: any, filters: FilterState): boolean {
    if (filters.category) {
      const categories = product.categories || []
      if (!categories.some((c: string) =>
        c.toLowerCase().includes(filters.category!.toLowerCase())
      )) {
        return false
      }
    }

    if (filters.color) {
      const desc = (product.item_description || "").toLowerCase()
      const name = (product.item_name || "").toLowerCase()
      const colorTerm = filters.color.toLowerCase()
      if (!desc.includes(colorTerm) && !name.includes(colorTerm)) {
        return false
      }
    }

    if (filters.material) {
      const desc = (product.item_description || "").toLowerCase()
      const materialTerm = filters.material.toLowerCase()
      if (!desc.includes(materialTerm)) {
        return false
      }
    }

    if (filters.brand) {
      const brand = (product.brand || "").toLowerCase()
      if (!brand.includes(filters.brand!.toLowerCase())) {
        return false
      }
    }

    if (filters.priceMin || filters.priceMax) {
      const prices = product.prices || []
      const hasValidPrice = prices.some((p: any) => {
        const price = p.price || 0
        if (filters.priceMin && price < filters.priceMin) return false
        if (filters.priceMax && price > filters.priceMax) return false
        return true
      })
      if (!hasValidPrice) return false
    }

    return true
  }

  /**
   * Text search fallback - MongoDB regex search.
   * Deterministic.
   */
  private async textSearch(
    query: string,
    n: number,
    filters?: FilterState
  ): Promise<any[]> {
    const regex = new RegExp(query, "i")

    const mongoFilter: any = {
      $or: [
        { item_name: { $regex: regex } },
        { item_description: { $regex: regex } },
        { categories: { $regex: regex } },
        { embedding_text: { $regex: regex } }
      ]
    }

    // Apply filters di MongoDB query level
    if (filters?.category) {
      mongoFilter.categories = { $regex: new RegExp(filters.category, "i") }
    }
    if (filters?.brand) {
      mongoFilter.brand = { $regex: new RegExp(filters.brand, "i") }
    }
    if (filters?.priceMin || filters?.priceMax) {
      mongoFilter["prices.price"] = {}
      if (filters.priceMin) mongoFilter["prices.price"].$gte = filters.priceMin
      if (filters.priceMax) mongoFilter["prices.price"].$lte = filters.priceMax
    }

    const results = await this.collection
      .find(mongoFilter)
      .limit(n)
      .toArray()

    console.log(`[ProductSearch] Text search returned ${results.length} results`)

    return results
  }

  /**
   * Generate slug from product name for clickable links.
   */
  private generateSlug(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')  // Remove special characters
      .replace(/\s+/g, '-')           // Replace spaces with hyphens
      .replace(/-+/g, '-')            // Remove multiple hyphens
      .trim()
  }

  /**
   * Format produk ke standard output.
   */
  private formatProducts(rawProducts: any[]): ProductItem[] {
    return rawProducts
      .map((item: any) => {
        const data = item[0]?.metadata || item.metadata || item
        const itemName = data.item_name || ''
        return {
          item_id: data.item_id,
          item_name: itemName,
          item_description: data.item_description,
          brand: data.brand,
          prices: data.prices || [],
          user_reviews: data.user_reviews,
          categories: data.categories || [],
          images: data.images || [],
          slug: this.generateSlug(itemName)  // Generate clickable slug
        }
      })
      .filter(p => p.item_id) // Filter invalid products
  }

  /**
   * Filter products by relevance to the search query.
   * This ensures only products matching the specific type are returned.
   * E.g., when searching for "dining table", filter out "coffee table", "console table", etc.
   */
  private filterByRelevance(products: ProductItem[], query: string): ProductItem[] {
    const queryLower = query.toLowerCase()

    // Define product type patterns that should be mutually exclusive
    const productTypePatterns = [
      { type: 'dining table', matchTerms: ['dining table', 'dining tables', 'meja makan'], excludeTerms: ['coffee', 'console', 'side', 'end', 'bedside', 'vanity'] },
      { type: 'coffee table', matchTerms: ['coffee table', 'coffee tables', 'meja tamu', 'meja kopi'], excludeTerms: ['dining', 'console', 'side', 'end'] },
      { type: 'console table', matchTerms: ['console table', 'console tables', 'meja konsol'], excludeTerms: ['dining', 'coffee', 'side', 'end'] },
      { type: 'side table', matchTerms: ['side table', 'side tables', 'meja samping'], excludeTerms: ['dining', 'coffee', 'console'] },
      { type: 'end table', matchTerms: ['end table', 'end tables'], excludeTerms: ['dining', 'coffee', 'console'] },
      { type: 'bedside table', matchTerms: ['bedside table', 'bedside tables', 'nightstand', 'meja nakas'], excludeTerms: ['dining', 'coffee', 'console'] },
      { type: 'dining chair', matchTerms: ['dining chair', 'dining chairs', 'kursi makan'], excludeTerms: ['office', 'lounge', 'arm', 'accent'] },
      { type: 'office chair', matchTerms: ['office chair', 'office chairs', 'kursi kantor'], excludeTerms: ['dining', 'lounge', 'arm'] },
      { type: 'lounge chair', matchTerms: ['lounge chair', 'lounge chairs', 'accent chair'], excludeTerms: ['dining', 'office'] },
      { type: 'sofa', matchTerms: ['sofa', 'sofas', 'couch'], excludeTerms: [] },
      { type: 'sectional', matchTerms: ['sectional', 'sectionals'], excludeTerms: [] },
    ]

    // Find which product type the user is searching for
    let matchedPattern: typeof productTypePatterns[0] | null = null
    for (const pattern of productTypePatterns) {
      if (pattern.matchTerms.some(term => queryLower.includes(term))) {
        matchedPattern = pattern
        console.log(`[ProductSearch] Detected product type: "${pattern.type}"`)
        break
      }
    }

    // If no specific pattern matched, return all products
    if (!matchedPattern || matchedPattern.excludeTerms.length === 0) {
      return products
    }

    // Filter products to only include those that match the type and exclude others
    const filtered = products.filter(product => {
      const productName = product.item_name.toLowerCase()

      // Check if product matches the searched type (e.g., "dining" for dining table)
      const searchedTypeKeyword = matchedPattern!.matchTerms[0].split(' ')[0] // "dining", "coffee", etc.
      const matchesSearchedType = new RegExp(`\\b${searchedTypeKeyword}\\b`, 'i').test(productName)

      // Check if product contains excluded terms as WHOLE WORDS (not substrings)
      // Using word boundary regex to avoid "Sendai" matching "end"
      const hasExcludedTerm = matchedPattern!.excludeTerms.some(term => {
        const wordBoundaryRegex = new RegExp(`\\b${term}\\b`, 'i')
        return wordBoundaryRegex.test(productName)
      })

      // If product matches searched type, keep it
      if (matchesSearchedType) {
        return true
      }

      // If product has an excluded term as a whole word, filter it out
      if (hasExcludedTerm) {
        console.log(`[ProductSearch] Filtering out: "${product.item_name}" (has excluded term)`)
        return false
      }

      return true
    })

    console.log(`[ProductSearch] Relevance filter: ${products.length} -> ${filtered.length} products`)
    return filtered
  }

  /**
   * Get product by ID - deterministic lookup.
   */
  async getById(itemId: string): Promise<ProductItem | null> {
    try {
      const doc = await this.collection.findOne({ item_id: itemId })
      if (!doc) return null

      return this.formatProducts([doc])[0]
    } catch (error) {
      console.error("[ProductSearch] Error getting product by ID:", error)
      return null
    }
  }

  /**
   * Get popular products by category.
   */
  async getPopularByCategory(category: string, n = 5): Promise<ProductItem[]> {
    try {
      const results = await this.collection
        .find({
          categories: { $regex: new RegExp(category, "i") }
        })
        .limit(n)
        .toArray()

      return this.formatProducts(results)
    } catch (error) {
      console.error("[ProductSearch] Error getting popular products:", error)
      return []
    }
  }
}

// ============================================================================
// EXPORT FACTORY FUNCTION
// ============================================================================

export function createProductSearchService(mongoClient: MongoClient): ProductSearchService {
  return new ProductSearchService(mongoClient)
}

