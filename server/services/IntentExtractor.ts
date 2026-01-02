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

// Catatan: OpenAI/Groq structured output API TIDAK mendukung .optional()
// Semua field harus REQUIRED atau nullable. Kita gunakan nullable untuk optional.
// Setelah parsing, kita convert null → undefined.

export const IntentSchema = z.object({
  // Intent type - apa yang user mau lakukan (REQUIRED)
  intent: z.enum([
    "search",        // Mau cari produk
    "filter_add",    // Nambah filter
    "filter_clear",  // Reset filter
    "greeting",      // Sapaan
    "help",          // Minta bantuan
    "faq_info",      // Pertanyaan umum (lokasi, jam buka, kontak, dll)
    "context_query", // Query about previous product ("this one", "that")
    "unknown"        // Tidak jelas
  ]),

  // Search query - nullable untuk optional (REQUIRED tapi boleh null)
  search_query: z.string().nullable(),

  // FAQ topic - untuk intent faq_info (REQUIRED tapi boleh null)
  faq_topic: z.string().nullable(),

  // Filter updates - semua field nullable untuk optional (REQUIRED tapi boleh null)
  category: z.string().nullable(),
  color: z.string().nullable(),
  material: z.string().nullable(),
  price_min: z.number().nullable(),
  price_max: z.number().nullable(),
  brand: z.string().nullable(),

  // NEW: Preference extraction fields
  size_preference: z.enum(["small", "medium", "large"]).nullable(),
  style_preference: z.enum(["warm", "minimalist", "modern", "classic", "scandinavian", "japandi"]).nullable(),
  room_type: z.string().nullable(),

  // NEW: Context reference detection
  context_reference: z.boolean().nullable(),  // True if user refers to "this one", "that", etc.
  color_variant_query: z.boolean().nullable(), // True if asking for different color of same product

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

  // faq_topic for faq_info intent
  if (data.faq_topic) {
    result.faq_topic = data.faq_topic
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

  // NEW: Preferences
  const preferences: any = {}
  if (data.size_preference) preferences.size = data.size_preference
  if (data.style_preference) preferences.style = data.style_preference
  if (data.room_type) preferences.room_type = data.room_type

  if (Object.keys(preferences).length > 0) {
    result.preferences = preferences
  }

  // NEW: Context flags
  if (data.context_reference) {
    result.context_reference = data.context_reference
  }
  if (data.color_variant_query) {
    result.color_variant_query = data.color_variant_query
  }

  return result
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
      lastIntent?: string
      lastFaqTopic?: string
    }
  ): Promise<Intent> {
    const systemPrompt = this.buildSystemPrompt(currentContext)

    try {
      // Use regular invoke with JSON parsing (Groq doesn't support json_schema format)
      const result = await this.model.invoke([
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage }
      ])

      const content = typeof result.content === "string" ? result.content : JSON.stringify(result.content)

      // Parse JSON from response
      const parsed = this.parseJsonResponse(content)

      // Validate and transform with Zod schema
      const validated = IntentSchema.parse(parsed)

      return validated as Intent
    } catch (error) {
      console.error("Intent extraction error:", error)

      // Fallback: return default intent with context
      return this.fallbackIntent(userMessage, currentContext)
    }
  }

  /**
   * Parse JSON from LLM response, handling markdown code blocks
   */
  private parseJsonResponse(content: string): any {
    // Remove markdown code blocks if present
    let cleaned = content
      .replace(/```json\n?/g, "")
      .replace(/```\n?/g, "")
      .trim()

    // Try to find JSON object in the response
    const jsonMatch = cleaned.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      cleaned = jsonMatch[0]
    }

    return JSON.parse(cleaned)
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
3. IMPORTANT: Handle continuation/elliptical queries - if user only mentions a color, material,
   or attribute without a product name, combine it with the last search query.
   Example: last search was "sofa", user says "white" → search_query: "white sofa", color: "white"
4. Extract filters: category, color, material, price range, brand
5. Detect language: "id" for Indonesian, "en" for English
6. Map common terms to ENGLISH categories:
   - sofa, settee, couch → category: "sofa"
   - table, dining table, coffee table → category: "table"
   - chair, armchair → category: "chair"
   - cabinet, wardrobe, cupboard → category: "cabinet"
   - shelf, bookshelf → category: "shelf"
   - bed, mattress → category: "bed"
7. Colors in English: white, black, brown, grey, gray, red, blue, green, yellow, cream, beige
8. Materials in English: wood, leather, fabric, metal, glass, rattan, plastic

CONTINUATION QUERY HANDLING:
- If message is ONLY a color (e.g., "putih", "merah"), combine with last_query
- If message is ONLY a material (e.g., "kayu", "kulit"), combine with last_query
- If message is a refinement (e.g., "yang warna putih"), combine with last_query
- The combined search_query should include both the product name and the new attribute

INTENT TYPES:
- search: User wants to find products (includes continuation queries)
- filter_add: User wants to add/update filter
- filter_clear: User wants to reset filters
- greeting: Hi, hello, halo
- help: User needs help
- faq_info: General questions NOT about products (lokasi toko, jam buka, kontak, pengiriman, pembayaran, garansi, dll)
- context_query: User refers to previous product ("this one", "that", "yang tadi", "is there a darker color for this")
- unknown: Cannot determine

CONTEXT REFERENCE DETECTION:
- Set context_reference: true if user uses words like "this", "that", "this one", "yang ini", "yang tadi", "for this"
- Set color_variant_query: true if user asks about different colors of same product ("darker color", "lighter", "warna lain")

PREFERENCES EXTRACTION:
- size_preference: "small" if user mentions small/kecil/compact, "large" for big/besar, null otherwise
- style_preference: "warm", "minimalist", "modern", "classic", "scandinavian", "japandi"
- room_type: living room, bedroom, dining room, office, etc.

FAQ TOPICS (for faq_info intent):
- location: Pertanyaan tentang lokasi/alamat toko
- hours: Jam buka/operasional
- contact: Nomor telepon, email, WhatsApp
- shipping: Pengiriman, ongkir, estimasi sampai
- payment: Metode pembayaran, transfer, cicilan
- warranty: Garansi, retur, komplain
- other: Pertanyaan umum lainnya

OUTPUT FORMAT (ALL FIELDS REQUIRED, use null for empty):
{
  "intent": "search" | "filter_add" | "filter_clear" | "greeting" | "help" | "faq_info" | "context_query" | "unknown",
  "search_query": "clean search query" | null,
  "faq_topic": "location" | "hours" | "contact" | "shipping" | "payment" | "warranty" | "other" | null,
  "category": "sofa" | null,
  "color": "putih" | null,
  "material": "kayu" | null,
  "price_min": 0 | null,
  "price_max": 1000000 | null,
  "brand": "IKEA" | null,
  "size_preference": "small" | "medium" | "large" | null,
  "style_preference": "warm" | "minimalist" | "modern" | "classic" | null,
  "room_type": "living room" | null,
  "context_reference": true | false | null,
  "color_variant_query": true | false | null,
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
        prompt += `Last search query: "${context.lastQuery}"\n`
        prompt += `⚠️ If current message is just an attribute (color/material), combine it with this last query!\n`
      }
    }

    return prompt
  }

  /**
   * Fallback ketika LLM gagal.
   * Simple rule-based extraction.
   */
  private fallbackIntent(message: string, context?: {
    currentCategory?: string
    activeFilters?: Record<string, any>
    lastQuery?: string
  }): Intent {
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
    }

    return {
      intent,
      search_query: searchQuery || undefined,
      filters: Object.keys(filters).length > 0 ? filters as any : undefined,
      language
    }
  }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

export const intentExtractor = new IntentExtractor()
