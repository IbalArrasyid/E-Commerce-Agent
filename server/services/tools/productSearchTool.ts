/**
 * Product Search Tool for LangChain Tool Calling
 * 
 * This wraps the existing ProductSearchService as a tool that can be
 * called by the LLM agent when it needs to search inventory.
 */

import { tool } from "@langchain/core/tools"
import { z } from "zod"
import { MongoClient } from "mongodb"
import { createProductSearchService, type ProductItem } from "../ProductSearch"

// Schema for the tool parameters
const ProductSearchSchema = z.object({
    query: z.string().describe("Search keywords like 'grey sofa' or 'teak coffee table'"),
    filters: z.object({
        color: z.string().optional().describe("Color filter like 'white', 'grey', 'brown'"),
        material: z.string().optional().describe("Material filter like 'wood', 'leather', 'fabric'"),
        priceMax: z.number().optional().describe("Maximum price in IDR"),
        priceMin: z.number().optional().describe("Minimum price in IDR"),
        category: z.string().optional().describe("Category like 'sofa', 'table', 'chair'"),
        brand: z.string().optional().describe("Brand name"),
    }).optional().describe("Optional filters to narrow search results"),
})

/**
 * Create the product search tool with MongoDB client injection.
 * Returns both the tool and a way to extract products from the state.
 */
export function createProductSearchTool(client: MongoClient) {
    // Storage for the last search results (side-effect for UI)
    let lastSearchResults: ProductItem[] = []

    const searchTool = tool(
        async ({ query, filters }) => {
            console.log(`[ProductSearchTool] Called with query: "${query}", filters:`, filters)

            const productSearch = createProductSearchService(client)

            const result = await productSearch.search({
                query,
                n: 6,
                filters: filters || {},
                searchType: "auto"
            })

            // Store results as side-effect for UI carousel
            lastSearchResults = result.products

            if (result.products.length === 0) {
                return JSON.stringify({
                    found: false,
                    message: "No products found matching your criteria.",
                    count: 0
                })
            }

            // Return a summary for LLM to process (not all data)
            const summary = result.products.map(p => ({
                name: p.item_name,
                price: p.prices?.[0]?.price ? `Rp ${p.prices[0].price.toLocaleString('id-ID')}` : 'Contact for price',
                brand: p.brand || 'Home Decor Indonesia',
                categories: p.categories.slice(0, 2).join(', ')
            }))

            return JSON.stringify({
                found: true,
                count: result.products.length,
                products: summary,
                searchType: result.searchType
            })
        },
        {
            name: "search_products",
            description: `Search for furniture products in the Home Decor Indonesia inventory. 
Use this tool whenever you want to recommend or show specific products to the user.
Always use this tool before mentioning specific products - never hallucinate product names or prices.`,
            schema: ProductSearchSchema,
        }
    )

    // Return tool and getter for last results
    return {
        tool: searchTool,
        getLastResults: () => lastSearchResults,
        clearResults: () => { lastSearchResults = [] }
    }
}

export type { ProductItem }
