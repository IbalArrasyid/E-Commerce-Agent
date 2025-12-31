/**
 * ResponseGenerator - LLM FOR SHORT NARRATIVE ONLY
 *
 * Tugas:
 * 1. Generate intro message berdasarkan search results
 * 2. Generate follow-up question yang relevan
 * 3. Handle greetings dan edge cases
 *
 * LLM BUKAN DATABASE. LLM BUKAN STATE MANAGER.
 * Data produk sudah disediakan - LLM hanya bikin narasi.
 */

import { ChatOpenAI } from "@langchain/openai"
import type { ProductItem, FilterState } from "./ConversationState.js"

// ============================================================================
// TYPES
// ============================================================================

export interface GeneratedResponse {
  intro: string
  followUp: string
}

export interface ResponseContext {
  language: "id" | "en"
  hasProducts: boolean
  productCount: number
  products: ProductItem[]
  searchQuery?: string
  activeFilters?: Partial<FilterState>
  intent?: string
}

// ============================================================================
// RESPONSE GENERATOR CLASS
// ============================================================================

export class ResponseGenerator {
  private model: ChatOpenAI

  constructor() {
    // Gunakan model yang cepat untuk narasi singkat
    this.model = new ChatOpenAI({
      apiKey: process.env.GROQ_API_KEY,
      model: "llama-3.3-70b-versatile",
      temperature: 0.7, // Sedikit kreativitas untuk natural response
      configuration: {
        baseURL: "https://api.groq.com/openai/v1",
      },
    })
  }

  /**
   * Generate response berdasarkan context dan search results.
   *
   * Note: Semua data sudah disediakan. LLM TIDAK mencari data.
   * LLM hanya membuat narasi yang natural.
   */
  async generate(context: ResponseContext): Promise<GeneratedResponse> {
    const { intent, hasProducts, productCount, language } = context

    // Handle special intents tanpa LLM (lebih cepat)
    if (intent === "greeting") {
      return this.greetingResponse(language)
    }

    if (intent === "help") {
      return this.helpResponse(language)
    }

    if (!hasProducts) {
      return this.noResultsResponse(context)
    }

    // Generate response untuk search results
    return this.productResponse(context)
  }

  /**
   * Greeting response - static, no LLM needed.
   */
  private greetingResponse(language: "id" | "en"): GeneratedResponse {
    if (language === "id") {
      return {
        intro: "Halo! Saya asisten belanja Home Decor. Ada yang bisa saya bantu? Anda bisa mencari sofa, meja, kursi, atau furnitur lainnya.",
        followUp: "Apa yang sedang Anda cari hari ini?"
      }
    }
    return {
      intro: "Hi! I'm your Home Decor shopping assistant. How can I help you today? You can search for sofas, tables, chairs, and more.",
      followUp: "What are you looking for today?"
    }
  }

  /**
   * Help response - static, no LLM needed.
   */
  private helpResponse(language: "id" | "en"): GeneratedResponse {
    if (language === "id") {
      return {
        intro: "Saya bisa membantu Anda mencari produk furnitur. Cukup sebutkan apa yang Anda cari, seperti 'sofa warna putih' atau 'meja makan kayu'.",
        followUp: "Apa yang ingin Anda cari?"
      }
    }
    return {
      intro: "I can help you find furniture products. Just tell me what you're looking for, like 'white sofa' or 'wooden dining table'.",
      followUp: "What would you like to search for?"
    }
  }

  /**
   * No results response - bisa pakai LLM untuk lebih helpful.
   */
  private noResultsResponse(context: ResponseContext): GeneratedResponse {
    const { language, activeFilters } = context

    if (language === "id") {
      let followUp = "Coba cari dengan kata kunci lain."

      if (activeFilters && Object.keys(activeFilters).length > 0) {
        followUp = "Coba kurangi filter atau ganti kategori."
      }

      return {
        intro: "Maaf, saya tidak menemukan produk yang cocok dengan pencarian Anda.",
        followUp
      }
    }

    let followUp = "Try a different search term."

    if (activeFilters && Object.keys(activeFilters).length > 0) {
      followUp = "Try reducing your filters or changing the category."
    }

    return {
      intro: "Sorry, I couldn't find any products matching your criteria.",
      followUp
    }
  }

  /**
   * Product response - gunakan LLM untuk narasi yang natural.
   * PROMPT KECIL - context sudah disediakan, LLM hanya format.
   */
  private async productResponse(context: ResponseContext): Promise<GeneratedResponse> {
    const { language, productCount, products, activeFilters } = context

    // Build context info untuk LLM
    const productNames = products.slice(0, 3).map(p => p.item_name).join(", ")
    const categoryHint = activeFilters?.category || products[0]?.categories[0] || ""

    const prompt = this.buildPrompt(language, productCount, productNames, categoryHint)

    try {
      const result = await this.model.invoke([
        { role: "system", content: this.getSystemPrompt(language) },
        { role: "user", content: prompt }
      ])

      const content = typeof result.content === "string" ? result.content : JSON.stringify(result.content)

      // Parse response sederhana
      return this.parseResponse(content, language)

    } catch (error) {
      console.error("[ResponseGenerator] LLM error, using fallback:", error)
      return this.fallbackResponse(context)
    }
  }

  /**
   * System prompt - kecil dan focused.
   */
  private getSystemPrompt(language: "id" | "en"): string {
    if (language === "id") {
      return `Tugas: Buat respons singkat dan ramah untuk chat e-commerce.

ATURAN:
1. Respons pendek saja (1-2 kalimat)
2. Jangan sebut daftar produk (produk sudah ditampilkan di UI)
3. Buat 1 pertanyaan follow-up yang relevan

OUTPUT FORMAT JSON:
{
  "intro": "pesan pembuka singkat",
  "followUp": "satu pertanyaan singkat"
}`
    }

    return `Task: Create a short, friendly response for e-commerce chat.

RULES:
1. Keep it short (1-2 sentences)
2. Don't list products (products are shown in UI)
3. Ask 1 relevant follow-up question

OUTPUT FORMAT JSON:
{
  "intro": "short opening message",
  "followUp": "one short question"
}`
  }

  /**
   * Build user prompt dengan context.
   */
  private buildPrompt(
    language: "id" | "en",
    count: number,
    productNames: string,
    category: string
  ): string {
    if (language === "id") {
      return `Context:
- Ditemukan ${count} produk
- Contoh produk: ${productNames}
- Kategori: ${category || "berbagai"}

Buat respons JSON yang ramah dan singkat. Jangan sebutkan query pencarian di respons.`
    }

    return `Context:
- Found ${count} products
- Sample products: ${productNames}
- Category: ${category || "various"}

Create a friendly, short JSON response. Do not mention the search query in the response.`
  }

  /**
   * Parse LLM response.
   */
  private parseResponse(content: string, language: "id" | "en"): GeneratedResponse {
    try {
      // Coba parse JSON
      const cleaned = content.replace(/```json\n?|\n?```/g, "").trim()
      const parsed = JSON.parse(cleaned)

      if (parsed.intro && parsed.followUp) {
        return { intro: parsed.intro, followUp: parsed.followUp }
      }
    } catch (e) {
      // Fall through ke default
    }

    // Fallback parsing
    return {
      intro: content.split("\n")[0] || (language === "id" ? "Berikut beberapa produk untuk Anda." : "Here are some products for you."),
      followUp: language === "id" ? "Ada yang ingin Anda tanyakan?" : "Any questions?"
    }
  }

  /**
   * Fallback response ketika LLM gagal.
   */
  private fallbackResponse(context: ResponseContext): GeneratedResponse {
    const { language, productCount } = context

    if (language === "id") {
      return {
        intro: `Berikut ${productCount} produk yang mungkin Anda suka.`,
        followUp: "Ada yang ingin Anda tanyakan tentang produk ini?"
      }
    }

    return {
      intro: `Here are ${productCount} products you might like.`,
      followUp: "Any questions about these products?"
    }
  }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

export const responseGenerator = new ResponseGenerator()