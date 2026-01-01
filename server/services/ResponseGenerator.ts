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
  faqTopic?: string  // For faq_info intent: location, hours, contact, shipping, payment, warranty, other
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
    const { intent, hasProducts, productCount, language, faqTopic } = context

    // Handle special intents tanpa LLM (lebih cepat)
    if (intent === "greeting") {
      return this.greetingResponse(language)
    }

    if (intent === "help") {
      return this.helpResponse(language)
    }

    if (intent === "faq_info") {
      return this.faqResponse(language, faqTopic)
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
    return {
      intro: "Hi! I'm your Home Decor shopping assistant. How can I help you today? You can search for sofas, tables, chairs, and more.",
      followUp: "What are you looking for today?"
    }
  }

  /**
   * Help response - static, no LLM needed.
   */
  private helpResponse(language: "id" | "en"): GeneratedResponse {
    return {
      intro: "I can help you find furniture products. Just tell me what you're looking for, like 'white sofa' or 'wooden dining table'.",
      followUp: "What would you like to search for?"
    }
  }

  /**
   * FAQ response - static responses for general questions.
   */
  private faqResponse(language: "id" | "en", topic?: string): GeneratedResponse {
    // Define FAQ responses for each topic (English only)
    const faqResponses: Record<string, GeneratedResponse> = {
      location: {
        intro: "Our store is located at Jalan Rs Fatmawati No. 5A,C,D 1, RT.1/RW.6, Gandaria Utara, Kec. Kby. Baru, Kota Jakarta Selatan, Daerah Khusus Ibukota Jakarta 12140. We also have showrooms in several major cities.",
        followUp: "Would you like to know our operating hours or search for products?"
      },
      hours: {
        intro: "Our store is open Monday-Saturday 9:30 AM - 06:30 PM, and Sunday 11:59 AM - 05:00 PM.",
        followUp: "Is there anything else I can help you with?"
      },
      contact: {
        intro: "You can reach us via WhatsApp at 0818-0604-0506, email: support@homedecorindonesia.com, or phone (021) 722-4248.",
        followUp: "Would you like to search for products or have other questions?"
      },
      shipping: {
        intro: "We deliver throughout Indonesia. Shipping cost is calculated based on weight and location. Estimated delivery 3-7 days for Java, 7-14 days for outside Java.",
        followUp: "Would you like to see our products?"
      },
      payment: {
        intro: "We accept bank transfer (BCA, Mandiri, BNI), credit cards, and 0% installment up to 12 months. COD is also available for certain areas.",
        followUp: "Is there anything else I can help you with?"
      },
      warranty: {
        intro: "All our products come with a minimum 2-year warranty. Returns can be made within 7 days if the product is damaged or not as expected. Contact our CS for claims.",
        followUp: "Would you like to search for products or have other questions?"
      },
      other: {
        intro: "Thank you for your question. For more information, please contact our customer service at 0818-0604-0506 or email support@homedecorindonesia.com.",
        followUp: "In the meantime, would you like to browse our product collection?"
      }
    }

    // Get response for the topic, default to 'other' if not found
    return faqResponses[topic || "other"] || faqResponses.other
  }

  /**
   * No results response - English only.
   */
  private noResultsResponse(context: ResponseContext): GeneratedResponse {
    const { activeFilters } = context

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
    const { language, productCount, products, activeFilters, searchQuery } = context

    // Check if products are actually relevant to the search query
    const relevanceInfo = this.checkProductRelevance(products, searchQuery)
    console.log(`[ResponseGenerator] Relevance check: ${relevanceInfo.matchCount}/${products.length} products match, isRelevant: ${relevanceInfo.isRelevant}`)

    // Build context info untuk LLM
    const productNames = products.slice(0, 3).map(p => p.item_name).join(", ")
    const categoryHint = activeFilters?.category || products[0]?.categories[0] || ""

    const prompt = this.buildPrompt(language, productCount, productNames, categoryHint, relevanceInfo, searchQuery)

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
      return this.fallbackResponse(context, relevanceInfo)
    }
  }

  /**
   * Check if products are relevant to the search query
   */
  private checkProductRelevance(products: ProductItem[], searchQuery?: string): {
    isRelevant: boolean
    matchCount: number
    suggestedCategory?: string
  } {
    if (!searchQuery || products.length === 0) {
      return { isRelevant: true, matchCount: products.length }
    }

    const queryTerms = searchQuery.toLowerCase().split(/\s+/)
    let matchCount = 0

    for (const product of products) {
      const productText = `${product.item_name} ${product.item_description} ${product.categories.join(" ")}`.toLowerCase()

      // Check if at least one query term matches
      const hasMatch = queryTerms.some(term =>
        term.length > 2 && productText.includes(term)
      )

      if (hasMatch) matchCount++
    }

    // Consider relevant if at least 30% of products match
    const isRelevant = matchCount >= Math.ceil(products.length * 0.3)

    // Find most common category from products
    const categoryFreq: Record<string, number> = {}
    for (const product of products) {
      for (const cat of product.categories) {
        categoryFreq[cat] = (categoryFreq[cat] || 0) + 1
      }
    }
    const suggestedCategory = Object.entries(categoryFreq)
      .sort((a, b) => b[1] - a[1])[0]?.[0]

    return { isRelevant, matchCount, suggestedCategory }
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
3. Jangan sebutkan query pencarian di respons
4. Buat 1 pertanyaan follow-up yang relevan

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
3. Do not mention the search query in the response
4. Ask 1 relevant follow-up question

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
    category: string,
    relevanceInfo?: { isRelevant: boolean; matchCount: number; suggestedCategory?: string },
    searchQuery?: string
  ): string {
    // If products are NOT relevant to search, tell LLM to be honest
    const isNotRelevant = relevanceInfo && !relevanceInfo.isRelevant

    if (language === "id") {
      if (isNotRelevant) {
        return `Context:
- User mencari: "${searchQuery}"
- Produk tersebut TIDAK TERSEDIA di katalog kami
- Kami menampilkan ${count} produk alternatif: ${productNames}
- Kategori alternatif: ${category || relevanceInfo?.suggestedCategory || "berbagai"}

Buat respons JSON yang JUJUR:
- Sampaikan bahwa produk yang dicari tidak tersedia
- Tawarkan produk alternatif yang ditampilkan
- Tanyakan apakah user tertarik dengan alternatif ini`
      }

      return `Context:
- Ditemukan ${count} produk yang cocok
- Contoh produk: ${productNames}
- Kategori: ${category || "berbagai"}

Buat respons JSON yang ramah dan singkat. Jangan sebutkan query pencarian di respons.`
    }

    // English
    if (isNotRelevant) {
      return `Context:
- User searched for: "${searchQuery}"
- This product is NOT AVAILABLE in our catalog
- We're showing ${count} alternative products: ${productNames}
- Alternative category: ${category || relevanceInfo?.suggestedCategory || "various"}

Create an HONEST JSON response:
- Tell user the searched product is not available
- Offer the alternative products shown
- Ask if user is interested in these alternatives`
    }

    return `Context:
- Found ${count} matching products
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
  private fallbackResponse(
    context: ResponseContext,
    relevanceInfo?: { isRelevant: boolean; matchCount: number; suggestedCategory?: string }
  ): GeneratedResponse {
    const { language, productCount, searchQuery } = context
    const isNotRelevant = relevanceInfo && !relevanceInfo.isRelevant

    if (language === "id") {
      if (isNotRelevant) {
        return {
          intro: `Maaf, kami tidak memiliki "${searchQuery}" di katalog. Berikut ${productCount} produk alternatif yang mungkin Anda suka.`,
          followUp: "Apakah Anda tertarik dengan produk-produk ini, atau ingin mencari yang lain?"
        }
      }
      return {
        intro: `Berikut ${productCount} produk yang cocok untuk Anda.`,
        followUp: "Ada yang ingin Anda tanyakan tentang produk ini?"
      }
    }

    if (isNotRelevant) {
      return {
        intro: `Sorry, we don't have "${searchQuery}" in our catalog. Here are ${productCount} alternative products you might like.`,
        followUp: "Are you interested in these products, or would you like to search for something else?"
      }
    }

    return {
      intro: `Here are ${productCount} products that match your search.`,
      followUp: "Any questions about these products?"
    }
  }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

export const responseGenerator = new ResponseGenerator()
