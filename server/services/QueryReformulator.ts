/**
 * QueryReformulator - SMART QUERY REFORMULATION
 *
 * Masalah: User sering ngomong singkat di percakapan berkelanjutan
 * - "sofa" → "putih" → "kulit" → "murah"
 *
 * Solusi: Reformulasi query yang cerdas, BUKAN sekadar menumpuk teks
 * - Base: "sofa"
 * - Setelah "putih": "sofa putih"
 * - Setelah "kulit": "sofa kulit putih" (clean reformulation, bukan "sofa putih kulit")
 * - Setelah "murah": "sofa kulit murah"
 *
 * PRINSIP:
 * - LLM hanya untuk reformulasi, bukan untuk state management
 * - Base query tetap, refinement di-reformulasi dengan cerdas
 */

import { ChatOpenAI } from "@langchain/openai"
import { z } from "zod"

// ============================================================================
// TYPES
// ============================================================================

export interface ReformulateContext {
  baseQuery: string        // Query dasar (e.g., "sofa", "meja makan")
  lastSearchQuery?: string // Full query terakhir (e.g., "sofa putih")
  activeFilters: {
    category?: string
    color?: string
    material?: string
    priceMin?: number
    priceMax?: number
    brand?: string
  }
  language: "id" | "en"
}

export interface ReformulatedQuery {
  query: string            // Clean, reformulated search query
  isContinuation: boolean  // Apakah ini query lanjutan
  isNewSearch: boolean     // Apakah ini pencarian baru (reset)
  detectedAttributes: {
    category?: string
    color?: string
    material?: string
    price?: string
  }
}

// ============================================================================
// SCHEMA
// ============================================================================

const ReformulateSchema = z.object({
  query: z.string().describe("Clean reformulated search query"),
  is_continuation: z.boolean().describe("True if this is a continuation of previous search"),
  is_new_search: z.boolean().describe("True if user is starting a completely new search"),
  detected_category: z.string().optional().nullable().describe("Product category if mentioned"),
  detected_color: z.string().optional().nullable().describe("Color if mentioned"),
  detected_material: z.string().optional().nullable().describe("Material if mentioned"),
  detected_price: z.string().optional().nullable().describe("Price reference if mentioned (murah/mahal)")
})

// ============================================================================
// QUERY REFORMULATOR CLASS
// ============================================================================

export class QueryReformulator {
  private model: ChatOpenAI

  constructor() {
    this.model = new ChatOpenAI({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama-3.3-70b-versatile",
      temperature: 0,
      configuration: {
        baseURL: "https://api.groq.com/openai/v1",
      },
    })
  }

  /**
   * Reformulasi query berdasarkan konteks percakapan.
   *
   * @param userMessage - Pesan user saat ini
   * @param context - Konteks percakapan sebelumnya
   * @returns Query yang sudah di-reformulasi dengan metadata
   */
  async reformulate(
    userMessage: string,
    context: ReformulateContext
  ): Promise<ReformulatedQuery> {

    console.log(`[QueryReformulator] Input - message: "${userMessage}", baseQuery: "${context.baseQuery}", lastSearchQuery: "${context.lastSearchQuery}"`)

    // Cek dulu secara rule-based untuk kasus sederhana (lebih cepat)
    const ruleBasedResult = this.ruleBasedReformulate(userMessage, context)
    if (ruleBasedResult) {
      console.log(`[QueryReformulator] Using rule-based result: "${ruleBasedResult.query}", isContinuation: ${ruleBasedResult.isContinuation}`)
      return ruleBasedResult
    }

    console.log(`[QueryReformulator] Rule-based didn't match, using LLM`)
    // Gunakan LLM untuk kasus yang lebih kompleks
    return this.llmReformulate(userMessage, context)
  }

  /**
   * Rule-based reformulation untuk kasus sederhana dan cepat.
   */
  private ruleBasedReformulate(
    userMessage: string,
    context: ReformulateContext
  ): ReformulatedQuery | null {

    const msg = userMessage.toLowerCase().trim()
    const baseQuery = context.baseQuery.toLowerCase()

    // Daftar filler words yang harus diabaikan saat analisis
    const fillerWords = ["saya", "mau", "yang", "ada", "adakah", "apa", "boleh", "bisa",
                        "yang", "warna", "bahannya", "buat", "untuk", "beli", "cari",
                        "i", "want", "would", "like", "the", "that", "this", "is", "are"]

    // Deteksi keyword yang menandakan pencarian baru
    const newSearchKeywords = [
      "cari ", "cariin", "tampil", "show", "find", "looking for",
      "ada ", "adakah", "apa ada", "is there"
    ]

    const isNewSearch = newSearchKeywords.some(kw => msg.startsWith(kw))

    // Jika user menyebut kategori produk baru, ini adalah pencarian baru
    const categories = ["sofa", "meja", "kursi", "lemari", "rak", "buffet", "bed", "kasur",
                        "couch", "table", "chair", "cabinet", "shelf", "coffee table"]

    const mentionedCategory = categories.find(cat => msg.includes(cat))

    if (mentionedCategory && mentionedCategory !== baseQuery) {
      // User mention kategori berbeda → new search
      return {
        query: userMessage, // Gunakan pesan asli sebagai query baru
        isContinuation: false,
        isNewSearch: true,
        detectedAttributes: {
          category: mentionedCategory
        }
      }
    }

    // Deteksi warna (continuation)
    const colors = ["putih", "white", "hitam", "black", "coklat", "brown", "merah", "red",
                    "biru", "blue", "hijau", "green", "kuning", "yellow", "abu", "gray",
                    "grey", "cream", "beige", "gold", "emas", "silver", "perak"]

    const mentionedColor = colors.find(c => msg.includes(c))

    // Deteksi material (continuation)
    const materials = ["kayu", "wood", "kulit", "leather", "kain", "fabric", "besi", "metal",
                       "rotan", "rattan", "plastik", "plastic", "kaca", "glass", "marmer",
                       "marble", "velvet", "beludru", "linen", "katun", "canvas"]

    const mentionedMaterial = materials.find(m => msg.includes(m))

    // Deteksi price reference (continuation)
    const priceRefs = ["murah", "cheap", "mahal", "expensive", "hemat", "economical",
                       "terjangkau", "affordable"]

    const mentionedPrice = priceRefs.find(p => msg.includes(p))

    // Jika ada attribute (color/material/price) dan ada baseQuery → cek apakah continuation
    const hasAttribute = !!(mentionedColor || mentionedMaterial || mentionedPrice)

    if (hasAttribute && baseQuery && !isNewSearch) {
      // Hitung "meaningful words" - abaikan filler words
      const words = msg.split(/\s+/).filter(w => w.length > 0)
      const meaningfulWords = words.filter(w => !fillerWords.includes(w))

      // Jika meaningful words ≤ 2, ini kemungkinan besar continuation
      // Contoh: "saya mau warna putih" → meaningful: "putih" (1 kata)
      // Contoh: "yang warna putih" → meaningful: "putih" (1 kata)
      if (meaningfulWords.length <= 2 || words.every(w => fillerWords.includes(w) || colors.includes(w) || materials.includes(w) || priceRefs.includes(w))) {
        // REFORMULASI CERDAS: baseQuery + new attribute
        // Bersihkan baseQuery dari attribute lama jika ada
        let cleanBaseQuery = baseQuery

        // Hapus warna lama dari baseQuery
        for (const c of colors) {
          cleanBaseQuery = cleanBaseQuery.replace(new RegExp(`\\b${c}\\b`, "gi"), "").trim()
        }

        // Hapus material lama dari baseQuery
        for (const m of materials) {
          cleanBaseQuery = cleanBaseQuery.replace(new RegExp(`\\b${m}\\b`, "gi"), "").trim()
        }

        // Hapus price reference lama dari baseQuery
        for (const p of priceRefs) {
          cleanBaseQuery = cleanBaseQuery.replace(new RegExp(`\\b${p}\\b`, "gi"), "").trim()
        }

        // Clean up: hilangkan "saya mau", "cari", dll dari baseQuery
        cleanBaseQuery = cleanBaseQuery
          .replace(/^(saya mau|saya cari|cari|mau|beli|tampil|show|find|looking for|i want|i need|adakah|apa ada)\s*/i, "")
          .replace(/^(yang|yang warna|yang bahannya|warna|bahannya)\s*/i, "")
          .trim()

        // Tambahkan attribute baru
        const attributes: string[] = []
        if (mentionedColor) attributes.push(mentionedColor)
        if (mentionedMaterial) attributes.push(mentionedMaterial)
        if (mentionedPrice) attributes.push(mentionedPrice)

        const newQuery = attributes.length > 0
          ? `${cleanBaseQuery} ${attributes.join(" ")}`.trim()
          : cleanBaseQuery

        return {
          query: newQuery,
          isContinuation: true,
          isNewSearch: false,
          detectedAttributes: {
            color: mentionedColor,
            material: mentionedMaterial,
            price: mentionedPrice
          }
        }
      }
    }

    // Jika baseQuery tidak ada, gunakan pesan asli
    if (!baseQuery) {
      return {
        query: userMessage,
        isContinuation: false,
        isNewSearch: false,
        detectedAttributes: {}
      }
    }

    // Default: tidak bisa handle rule-based, delegate ke LLM
    return null
  }

  /**
   * LLM-based reformulation untuk kasus kompleks.
   */
  private async llmReformulate(
    userMessage: string,
    context: ReformulateContext
  ): Promise<ReformulatedQuery> {

    const systemPrompt = this.buildSystemPrompt(context)

    try {
      const structuredLlm = this.model.withStructuredOutput(ReformulateSchema)

      const result = await structuredLlm.invoke([
        { role: "system", content: systemPrompt },
        { role: "user", content: userMessage }
      ])

      return {
        query: result.query,
        isContinuation: result.is_continuation,
        isNewSearch: result.is_new_search,
        detectedAttributes: {
          category: result.detected_category || undefined,
          color: result.detected_color || undefined,
          material: result.detected_material || undefined,
          price: result.detected_price || undefined
        }
      }
    } catch (error) {
      console.error("[QueryReformulator] LLM error, using fallback:", error)

      // Fallback: gunakan pesan asli
      return {
        query: userMessage,
        isContinuation: false,
        isNewSearch: false,
        detectedAttributes: {}
      }
    }
  }

  /**
   * Build system prompt untuk LLM.
   */
  private buildSystemPrompt(context: ReformulateContext): string {
    const { baseQuery, lastSearchQuery, activeFilters, language } = context

    const isIndonesian = language === "id"

    return `TASK: Reformulate search query based on conversation context.

CURRENT CONTEXT:
- Base product query: "${baseQuery}"
${lastSearchQuery ? `- Last search: "${lastSearchQuery}"` : ""}
${Object.keys(activeFilters).length > 0 ? `- Active filters: ${JSON.stringify(activeFilters)}` : ""}

CRITICAL RULES:
1. DO NOT just append text - REFORMULATE INTELLIGENTLY
2. If user message is JUST an attribute (color/material), replace old attribute with new one
   - Example: base="sofa putih", user="kulit" → "sofa kulit" (remove "putih", add "kulit")
   - Example: base="sofa", user="putih" → "sofa putih"
3. If user mentions a NEW product category, this is a NEW search
   - Example: base="sofa", user="ada meja" → "meja" (new search)
4. Keep query CONCISE - avoid word repetition
5. ${isIndonesian ? 'Use Indonesian for query terms' : 'Use English for query terms'}

ATTRIBUTE DETECTION:
- Colors: ${isIndonesian ? 'putih, hitam, coklat, merah, biru, hijau, abu' : 'white, black, brown, red, blue, green, gray'}
- Materials: ${isIndonesian ? 'kayu, kulit, kain, besi, rotan, kaca' : 'wood, leather, fabric, metal, rattan, glass'}
- Price: ${isIndonesian ? 'murah, mahal, hemat' : 'cheap, expensive, affordable'}

OUTPUT FORMAT JSON:
{
  "query": "clean reformulated search query",
  "is_continuation": true/false,
  "is_new_search": true/false,
  "detected_category": "category or null",
  "detected_color": "color or null",
  "detected_material": "material or null",
  "detected_price": "price reference or null"
}`
  }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

export const queryReformulator = new QueryReformulator()
