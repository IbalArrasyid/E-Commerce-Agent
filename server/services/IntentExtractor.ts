/**
 * IntentExtractor - SMALL LLM FOR INTENT EXTRACTION ONLY
 *
 * Tugas:
 * 1. Extract search query dari user message
 * 2. Extract/update filters (color, material, price, category)
 * 3. Detect language (ID/EN)
 * 4. Detect intent type (search, filter_update, greeting, etc.)
 *
 * PROMPT KECIL - murah token, fast response.
 */

import { ChatOpenAI } from "@langchain/openai"
import { z } from "zod"

// ============================================================================
// OUTPUT SCHEMA (Structured Output)
// ============================================================================

<<<<<<< HEAD
// Catatan: OpenAI/Groq structured output API TIDAK mendukung .optional()
// Semua field harus REQUIRED atau nullable. Kita gunakan nullable untuk optional.
// Setelah parsing, kita convert null → undefined.

export const IntentSchema = z.object({
  // Intent type - apa yang user mau lakukan (REQUIRED)
=======
export const IntentSchema = z.object({
  // Intent type - apa yang user mau lakukan
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
  intent: z.enum([
    "search",        // Mau cari produk
    "filter_add",    // Nambah filter
    "filter_clear",  // Reset filter
    "greeting",      // Sapaan
    "help",          // Minta bantuan
    "unknown"        // Tidak jelas
  ]),

<<<<<<< HEAD
  // Search query - nullable untuk optional (REQUIRED tapi boleh null)
  search_query: z.string().nullable(),

  // Filter updates - semua field nullable untuk optional (REQUIRED tapi boleh null)
  category: z.string().nullable(),
  color: z.string().nullable(),
  material: z.string().nullable(),
  price_min: z.number().nullable(),
  price_max: z.number().nullable(),
  brand: z.string().nullable(),

  // Language detection (REQUIRED)
  language: z.enum(["id", "en"])
}).transform(data => {
  // Convert null → undefined setelah parsing
  const result: any = {
    intent: data.intent,
    language: data.language
  }

  // search_query
  if (data.search_query) {
    result.search_query = data.search_query
  }

  // filters - kumpulkan field yang tidak null
  const filters: any = {}
  if (data.category) filters.category = data.category
  if (data.color) filters.color = data.color
  if (data.material) filters.material = data.material
  if (data.price_min != null) filters.price_min = data.price_min
  if (data.price_max != null) filters.price_max = data.price_max
  if (data.brand) filters.brand = data.brand

  if (Object.keys(filters).length > 0) {
    result.filters = filters
  }

  return result
=======
  // Search query - jika user mau cari sesuatu
  search_query: z.string().optional().describe("Search query if user wants to search"),

  // Filter updates
  filters: z.object({
    category: z.string().optional().describe("Product category like sofa, meja, kursi"),
    color: z.string().optional().describe("Color like putih, hitam, coklat, merah"),
    material: z.string().optional().describe("Material like kayu, kulit, kain, besi"),
    price_min: z.number().optional().describe("Minimum price"),
    price_max: z.number().optional().describe("Maximum price"),
    brand: z.string().optional().describe("Brand name")
  }).optional().describe("Filters to add/update"),

  // Language detection
  language: z.enum(["id", "en"]).describe("Detected language: id for Indonesian, en for English"),

  // Confidence score (optional)
  confidence: z.number().min(0).max(1).optional().describe("Confidence score 0-1")
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
})

export type Intent = z.infer<typeof IntentSchema>

// ============================================================================
// INTENT EXTRACTOR CLASS
// ============================================================================

export class IntentExtractor {
  private model: ChatOpenAI

  constructor() {
    // Gunakan model kecil/murah untuk intent extraction
    this.model = new ChatOpenAI({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama-3.3-70b-versatile", // atau "gpt-4o-mini" jika pakai OpenAI
      temperature: 0,
      configuration: {
        baseURL: "https://api.groq.com/openai/v1",
      },
    })
  }

  /**
   * Extract intent dari user message.
   *
   * @param userMessage - Pesan dari user
   * @param currentContext - Context state saat ini (optional, untuk better extraction)
   * @returns Structured intent object
   */
  async extract(
    userMessage: string,
    currentContext?: {
      currentCategory?: string
      activeFilters?: Record<string, any>
      lastQuery?: string
    }
  ): Promise<Intent> {
    const systemPrompt = this.buildSystemPrompt(currentContext)

    try {
      // Structured output with Zod schema
      const structuredLlm = this.model.withStructuredOutput(IntentSchema)

      const result = await structuredLlm.invoke([
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage }
      ])

      return result as Intent
    } catch (error) {
      console.error("Intent extraction error:", error)

<<<<<<< HEAD
      // Fallback: return default intent with context
      return this.fallbackIntent(userMessage, currentContext)
=======
      // Fallback: return default intent
      return this.fallbackIntent(userMessage)
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
    }
  }

  /**
   * Build system prompt dengan context injection.
   * Prompt kecil - hanya instruksi yang necessary.
   */
  private buildSystemPrompt(context?: {
    currentCategory?: string
    activeFilters?: Record<string, any>
    lastQuery?: string
  }): string {
    let prompt = `TASK: Extract intent and filters from user message.

RULES:
1. Return VALID JSON only
2. Extract search query if user wants to find products
<<<<<<< HEAD
3. IMPORTANT: Handle continuation/elliptical queries - if user only mentions a color, material,
   or attribute without a product name, combine it with the last search query.
   Example: last search was "sofa", user says "putih" → search_query: "sofa putih", color: "putih"
4. Extract filters: category, color, material, price range, brand
5. Detect language: "id" for Indonesian, "en" for English
6. Map common terms:
=======
3. Extract filters: category, color, material, price range, brand
4. Detect language: "id" for Indonesian, "en" for English
5. Map common terms:
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
   - sofa, settee, couch → category: "sofa"
   - meja, table → category: "meja"
   - kursi, chair → category: "kursi"
   - lemari, cabinet → category: "lemari"
   - putih, white → color: "putih"
   - hitam, black → color: "hitam"
   - kayu, wood → material: "kayu"
   - kulit, leather → material: "kulit"

<<<<<<< HEAD
CONTINUATION QUERY HANDLING:
- If message is ONLY a color (e.g., "putih", "merah"), combine with last_query
- If message is ONLY a material (e.g., "kayu", "kulit"), combine with last_query
- If message is a refinement (e.g., "yang warna putih"), combine with last_query
- The combined search_query should include both the product name and the new attribute

INTENT TYPES:
- search: User wants to find products (includes continuation queries)
=======
INTENT TYPES:
- search: User wants to find products
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
- filter_add: User wants to add/update filter
- filter_clear: User wants to reset filters
- greeting: Hi, hello, halo
- help: User needs help
- unknown: Cannot determine

<<<<<<< HEAD
OUTPUT FORMAT (ALL FIELDS REQUIRED, use null for empty):
{
  "intent": "search" | "filter_add" | "filter_clear" | "greeting" | "help" | "unknown",
  "search_query": "clean search query" | null,
  "category": "sofa" | null,
  "color": "putih" | null,
  "material": "kayu" | null,
  "price_min": 0 | null,
  "price_max": 1000000 | null,
  "brand": "IKEA" | null,
=======
OUTPUT FORMAT:
{
  "intent": "search" | "filter_add" | "filter_clear" | "greeting" | "help" | "unknown",
  "search_query": "clean search query" | null,
  "filters": {
    "category": "sofa" | null,
    "color": "putih" | null,
    "material": "kayu" | null,
    "price_min": 0 | null,
    "price_max": 1000000 | null,
    "brand": "IKEA" | null
  },
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
  "language": "id" | "en"
}`

    // Inject context jika ada
    if (context) {
      prompt += "\n\nCURRENT CONTEXT:\n"
      if (context.currentCategory) {
        prompt += `Current category: ${context.currentCategory}\n`
      }
      if (context.activeFilters && Object.keys(context.activeFilters).length > 0) {
        prompt += `Active filters: ${JSON.stringify(context.activeFilters)}\n`
      }
      if (context.lastQuery) {
<<<<<<< HEAD
        prompt += `Last search query: "${context.lastQuery}"\n`
        prompt += `⚠️ If current message is just an attribute (color/material), combine it with this last query!\n`
=======
        prompt += `Last search: "${context.lastQuery}"\n`
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
      }
    }

    return prompt
  }

  /**
   * Fallback ketika LLM gagal.
   * Simple rule-based extraction.
   */
<<<<<<< HEAD
  private fallbackIntent(message: string, context?: {
    currentCategory?: string
    activeFilters?: Record<string, any>
    lastQuery?: string
  }): Intent {
=======
  private fallbackIntent(message: string): Intent {
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
    const msg = message.toLowerCase()

    // Detect language
    const isIndonesian = /^(apa|ada|saya|mau|cari|tampil|tolong|halo|hai|boleh|bisa|produk|barang|warna|harga)/i.test(msg)
    const language: "id" | "en" = isIndonesian ? "id" : "en"

    // Detect intent type
    let intent: Intent["intent"] = "search"

    if (/^(hi|hello|halo|hai|selamat|pagi|siang|sore|malam)/i.test(msg)) {
      intent = "greeting"
    } else if (/^(help|bantu|bagaimana|cara)/i.test(msg)) {
      intent = "help"
    } else if (/reset|hapus|clear|kosongkan/i.test(msg)) {
      intent = "filter_clear"
    }

<<<<<<< HEAD
    // Extract filters and search query
    let searchQuery = message
      .replace(/^(saya mau|saya cari|cari|mau|tampil|show|find|looking for|i want|i need|adakah|apa ada)\s*/i, "")
      .replace(/^(hai|halo|hello|hi|selamat|pagi|siang|sore|malam)\s*,?\s*/i, "")
      .replace(/^(yang|yang warna|yang bahannya|warna|bahannya)\s*/i, "")
      .trim()

    const filters: Record<string, string | number> = {}

    // Color detection
    const colors = ["putih", "white", "hitam", "black", "coklat", "brown", "merah", "red",
                    "biru", "blue", "hijau", "green", "kuning", "yellow", "abu", "gray", "grey"]
    for (const color of colors) {
      if (msg.includes(color)) {
        filters.color = color.length <= 5 ? color : (color === "white" ? "putih" :
                        color === "black" ? "hitam" :
                        color === "brown" ? "coklat" :
                        color === "red" ? "merah" :
                        color === "blue" ? "biru" :
                        color === "green" ? "hijau" :
                        color === "yellow" ? "kuning" :
                        color === "gray" || color === "grey" ? "abu" : color)
        // Remove color from search query
        searchQuery = searchQuery.replace(new RegExp(color, "gi"), "").trim()
        break
      }
    }

    // Material detection
    const materials = ["kayu", "wood", "kulit", "leather", "kain", "fabric", "besi", "metal",
                       "rotan", "rattan", "plastik", "plastic", "kaca", "glass"]
    for (const material of materials) {
      if (msg.includes(material)) {
        filters.material = material.length <= 5 ? material : (material === "wood" ? "kayu" :
                          material === "leather" ? "kulit" :
                          material === "fabric" ? "kain" :
                          material === "metal" ? "besi" :
                          material === "rattan" ? "rotan" :
                          material === "plastic" ? "plastik" :
                          material === "glass" ? "kaca" : material)
        searchQuery = searchQuery.replace(new RegExp(material, "gi"), "").trim()
        break
      }
    }

    // CONTINUATION QUERY HANDLING: If search query is very short (just attribute) and we have lastQuery
    const isOnlyAttribute = searchQuery.length <= 10 || /^[a-z]{3,10}$/i.test(searchQuery)
    if (isOnlyAttribute && context?.lastQuery && Object.keys(filters).length > 0) {
      // Combine with last query
      searchQuery = `${context.lastQuery} ${message}`.trim()
    }

    if (searchQuery.length === 0) {
      searchQuery = context?.lastQuery || message // fallback
=======
    // Extract search query (simple - remove common words)
    let searchQuery = message
      .replace(/^(saya mau|saya cari|cari|mau|tampil|show|find|looking for|i want|i need|adakah|apa ada)\s*/i, "")
      .replace(/^(hai|halo|hello|hi|selamat|pagi|siang|sore|malam)\s*,?\s*/i, "")
      .trim()

    if (searchQuery.length === 0) {
      searchQuery = message // fallback
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
    }

    return {
      intent,
      search_query: searchQuery || undefined,
<<<<<<< HEAD
      filters: Object.keys(filters).length > 0 ? filters as any : undefined,
=======
      filters: {},
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
      language
    }
  }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

<<<<<<< HEAD
export const intentExtractor = new IntentExtractor()
=======
export const intentExtractor = new IntentExtractor()
>>>>>>> 65e9c20049a10c65392dc9f0a9849e4eb60c2622
