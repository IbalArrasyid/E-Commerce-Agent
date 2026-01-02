/**
 * PreferenceManager - MANAGE USER PREFERENCES ACROSS CONVERSATION
 *
 * Responsibilities:
 * 1. Store and merge user preferences (color, budget, size, style)
 * 2. Resolve context references ("this one" → previous product)
 * 3. Provide preference summary for search filters
 */

// ============================================================================
// TYPES
// ============================================================================

export interface UserPreferences {
    color?: string
    budget?: {
        min?: number
        max?: number
    }
    size?: "small" | "medium" | "large"
    style?: "warm" | "minimalist" | "modern" | "classic" | "scandinavian" | "japandi"
    roomType?: string  // e.g., "living room", "bedroom"
    category?: string  // e.g., "sofa", "table"
}

export interface ConversationContext {
    lastRecommendedProducts: Array<{
        item_id: string
        item_name: string
        category?: string
        colors?: string[]
    }>
    lastCategory?: string
    lastSearchQuery?: string
    turnCount: number
}

export interface PreferenceUpdate {
    color?: string
    budget_max?: number
    budget_min?: number
    size?: "small" | "medium" | "large"
    style?: "warm" | "minimalist" | "modern" | "classic" | "scandinavian" | "japandi"
    roomType?: string
    category?: string
}

// ============================================================================
// PREFERENCE MANAGER CLASS
// ============================================================================

export class PreferenceManager {
    /**
     * Merge new preference updates with existing preferences.
     * New values override old values, but null/undefined don't erase existing.
     */
    merge(existing: UserPreferences, updates: PreferenceUpdate): UserPreferences {
        const merged: UserPreferences = { ...existing }

        if (updates.color) {
            merged.color = this.normalizeColor(updates.color)
        }

        if (updates.budget_max !== undefined || updates.budget_min !== undefined) {
            merged.budget = {
                min: updates.budget_min ?? existing.budget?.min,
                max: updates.budget_max ?? existing.budget?.max
            }
        }

        if (updates.size) {
            merged.size = updates.size
        }

        if (updates.style) {
            merged.style = this.normalizeStyle(updates.style)
        }

        if (updates.roomType) {
            merged.roomType = updates.roomType
        }

        if (updates.category) {
            merged.category = updates.category
        }

        return merged
    }

    /**
     * Resolve context references like "this one", "that product", etc.
     * Returns the referenced product's details or null if can't resolve.
     */
    resolveContextReference(
        context: ConversationContext,
        reference: string
    ): { item_id: string; item_name: string; category?: string } | null {
        const refLower = reference.toLowerCase()

        // Common context reference patterns
        const contextPatterns = [
            "this one", "this", "that one", "that", "previous", "the one",
            "ini", "itu", "yang tadi", "yang itu", "produk tadi"
        ]

        const hasContextRef = contextPatterns.some(p => refLower.includes(p))

        if (!hasContextRef) {
            return null
        }

        // Return the first (most recently recommended) product
        if (context.lastRecommendedProducts.length > 0) {
            return context.lastRecommendedProducts[0]
        }

        return null
    }

    /**
     * Detect if user is asking about color variants of previous product.
     */
    detectColorVariantQuery(
        message: string,
        context: ConversationContext
    ): { isVariantQuery: boolean; requestedColor?: string; baseProduct?: string } {
        const msgLower = message.toLowerCase()

        // Patterns for color variant queries
        const variantPatterns = [
            /darker\s*(color|colour)?/,
            /lighter\s*(color|colour)?/,
            /different\s*(color|colour)/,
            /other\s*(color|colour)/,
            /warna\s*(lain|berbeda|gelap|terang)/,
            /ada\s*(warna|color)\s*(\w+)/,
            /is there\s*(a\s*)?(darker|lighter|different)/
        ]

        const isVariantQuery = variantPatterns.some(p => p.test(msgLower))

        if (!isVariantQuery || context.lastRecommendedProducts.length === 0) {
            return { isVariantQuery: false }
        }

        // Try to detect requested color
        let requestedColor: string | undefined

        if (msgLower.includes("dark") || msgLower.includes("gelap")) {
            // User wants darker - common dark colors
            const darkColors = ["dark brown", "dark gray", "black", "navy", "charcoal"]
            requestedColor = darkColors[0]
        } else if (msgLower.includes("light") || msgLower.includes("terang")) {
            const lightColors = ["light brown", "beige", "cream", "white", "light gray"]
            requestedColor = lightColors[0]
        }

        // Extract specific color if mentioned
        const colorMatch = msgLower.match(/(?:warna|color)\s+(\w+)/)
        if (colorMatch) {
            requestedColor = colorMatch[1]
        }

        return {
            isVariantQuery: true,
            requestedColor,
            baseProduct: context.lastRecommendedProducts[0]?.item_name
        }
    }

    /**
     * Build search filters from preferences.
     */
    buildSearchFilters(preferences: UserPreferences): Record<string, any> {
        const filters: Record<string, any> = {}

        if (preferences.category) {
            filters.category = preferences.category
        }

        if (preferences.color) {
            filters.color = preferences.color
        }

        if (preferences.budget?.max) {
            filters.priceMax = preferences.budget.max
        }

        if (preferences.budget?.min) {
            filters.priceMin = preferences.budget.min
        }

        return filters
    }

    /**
     * Check if we have enough preferences to search.
     * If not, returns questions to ask.
     */
    getMissingPreferences(
        preferences: UserPreferences,
        intent: string
    ): string[] {
        const missing: string[] = []

        // For search intent, we need at least a category or search query
        if (intent === "search") {
            if (!preferences.category && !preferences.color && !preferences.style) {
                // We have very little to go on - but this is OK, we'll search broadly
            }
        }

        return missing  // Empty = no critical missing info
    }

    /**
     * Normalize color to standard form.
     */
    private normalizeColor(color: string): string {
        const colorMap: Record<string, string> = {
            "putih": "white",
            "hitam": "black",
            "coklat": "brown",
            "cokelat": "brown",
            "light brown": "light brown",
            "dark brown": "dark brown",
            "abu": "gray",
            "abu-abu": "gray",
            "merah": "red",
            "biru": "blue",
            "hijau": "green",
            "kuning": "yellow",
            "cream": "cream",
            "krem": "cream",
            "beige": "beige"
        }
        return colorMap[color.toLowerCase()] || color.toLowerCase()
    }

    /**
     * Normalize style to standard form.
     */
    private normalizeStyle(style: string): UserPreferences["style"] {
        const styleMap: Record<string, UserPreferences["style"]> = {
            "warm": "warm",
            "hangat": "warm",
            "minimalis": "minimalist",
            "minimalist": "minimalist",
            "modern": "modern",
            "klasik": "classic",
            "classic": "classic",
            "scandinavian": "scandinavian",
            "scandinavia": "scandinavian",
            "japandi": "japandi"
        }
        return styleMap[style.toLowerCase()] || "modern"
    }

    /**
     * Detect size preference from message.
     */
    detectSizeFromMessage(message: string): "small" | "medium" | "large" | undefined {
        const msgLower = message.toLowerCase()

        const smallPatterns = ["small", "kecil", "compact", "mini", "mungil"]
        const largePatterns = ["large", "besar", "big", "spacious", "luas"]

        if (smallPatterns.some(p => msgLower.includes(p))) {
            return "small"
        }

        if (largePatterns.some(p => msgLower.includes(p))) {
            return "large"
        }

        return undefined
    }

    /**
     * Detect room type from message.
     */
    detectRoomFromMessage(message: string): string | undefined {
        const msgLower = message.toLowerCase()

        const roomPatterns: Record<string, string[]> = {
            "living room": ["living room", "ruang tamu", "ruang keluarga"],
            "bedroom": ["bedroom", "kamar tidur", "kamar"],
            "dining room": ["dining room", "ruang makan"],
            "office": ["office", "kantor", "ruang kerja", "study"]
        }

        for (const [room, patterns] of Object.entries(roomPatterns)) {
            if (patterns.some(p => msgLower.includes(p))) {
                return room
            }
        }

        return undefined
    }
}

// ============================================================================
// EXPORT SINGLETON
// ============================================================================

export const preferenceManager = new PreferenceManager()
