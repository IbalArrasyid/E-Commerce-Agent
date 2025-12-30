// Import required modules from LangChain ecosystem
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai" // For creating vector embeddings from text using Gemini
import { ChatOpenAI } from "@langchain/openai"                     // OpenAI chat model integration
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages" // Message types for conversations
import {
  ChatPromptTemplate,      // For creating structured prompts with placeholders
  MessagesPlaceholder,     // Placeholder for dynamic message history
} from "@langchain/core/prompts"
import { StateGraph } from "@langchain/langgraph"              // State-based workflow orchestration
import { Annotation } from "@langchain/langgraph"              // Type annotations for state management
import { tool } from "@langchain/core/tools"                   // For creating custom tools/functions
import { ToolNode } from "@langchain/langgraph/prebuilt"       // Pre-built node for executing tools
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb" // For saving conversation state
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"   // Vector search integration with MongoDB
import { MongoClient } from "mongodb"                          // MongoDB database client
import { z } from "zod"                                        // Schema validation library
import "dotenv/config"                                         // Load environment variables from .env file

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Represents price information for a product variant
 */
export interface ProductPrice {
  variant?: string
  price: number
  currency?: string
}

/**
 * Represents a single product item from the inventory
 */
export interface ProductItem {
  item_id: string
  item_name: string
  item_description: string
  brand: string
  prices: ProductPrice[]
  user_reviews?: {
    rating?: number
    count?: number
  }
  categories: string[]
  images: string[]
}

/**
 * Represents the structured agent response with separated text and products
 */
export interface AgentResponse {
  // Opening/conversational text from the AI
  intro: string
  // Product recommendations extracted from the search results (single source of truth)
  products: ProductItem[]
  // Follow-up question from the AI
  followUp: string
  // Metadata about the response
  meta?: {
    hasProducts: boolean
    searchType?: "vector" | "text" | "none"
    productCount: number
  }
}

/**
 * Represents the raw search results from the item lookup tool
 */
interface SearchResults {
  results?: Array<[any, number] | any>
  searchType?: string
  query?: string
  count?: number
  error?: string
  message?: string
}

// Function to parse product recommendations from AI response and search results
function extractProductsFromTool(searchResults: SearchResults | null): {
  hasProducts: boolean
  products: ProductItem[]
  searchType?: "vector" | "text" | "none"
} {
  if (!searchResults || !searchResults.results) {
    return { hasProducts: false, products: [], searchType: "none" }
  }

  const products: ProductItem[] = searchResults.results
    .map((item: any) => {
      const data = item[0]?.metadata || item.metadata || item
      return {
        item_id: data.item_id,
        item_name: data.item_name,
        item_description: data.ai_generated?.description || data.item_description,
        brand: data.brand,
        prices: data.prices,
        user_reviews: data.user_reviews,
        categories: data.categories,
        images: data.images || []
      }
    })
    .slice(0, 5)

  const searchType: "vector" | "text" = searchResults.searchType === "vector" ? "vector" : "text"

  return { hasProducts: products.length > 0, products, searchType }
}

// Utility function to handle API rate limits with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,    // The function to retry (generic type T for return value)
  maxRetries = 3           // Maximum number of retry attempts (default 3)
): Promise<T> {
  // Loop through retry attempts
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn()    // Try to execute the function
    } catch (error: any) {
      // Check if it's a rate limit error (HTTP 429) and we have retries left
      if (error.status === 429 && attempt < maxRetries) {
        // Calculate exponential backoff delay: 2^attempt seconds, max 30 seconds
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000)
        console.log(`Rate limit hit. Retrying in ${delay/1000} seconds...`)
        // Wait for the calculated delay before retrying
        await new Promise(resolve => setTimeout(resolve, delay))
        continue // Go to next iteration (retry)
      }
      throw error // If not rate limit or out of retries, throw the error
    }
  }
  throw new Error("Max retries exceeded") // This should never be reached
}

async function reformulateQuery(originalQuery: string) {
  const reformulator = new ChatOpenAI({
    apiKey: process.env.GROQ_API_KEY,
    model: "openai/gpt-oss-120b",
    temperature: 0,
    configuration: {
      baseURL: "https://api.groq.com/openai/v1",
    },
  })

  const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `TASK: Query rewriting ONLY.

    You are NOT a chatbot.
    You are NOT allowed to answer questions.
    You ONLY rewrite the user query.

    RULES (STRICT):
    - Rewrite the query into a short, explicit search query.
    - Keep the original intent EXACTLY.
    - DO NOT answer the question.
    - DO NOT add opinions.
    - DO NOT change meaning.
    - Output ONLY the rewritten query.
    - NO punctuation at the end.
    - NO explanations.
    - NO extra words.

    BAD EXAMPLES:
    User: "Do you have dining tables?"
    Wrong: "Yes, we have dining tables."
    Wrong: "I have dining tables available."
    Wrong: "Saya memiliki meja makan."

    GOOD EXAMPLES:
    User: "Do you have dining tables?"
    Output: "dining table furniture"

    User: "Meja makan kayu minimalis"
    Output: "meja makan kayu minimalis"

    If you violate the rules, your output is invalid.`
      ],
      ["human", "{query}"]
    ])

  const formatted = await prompt.formatMessages({
    query: originalQuery,
  })

  const result = await reformulator.invoke(formatted)

    // return typeof result.content === "string"
    //   ? result.content.trim()
    //   : originalQuery

  const text =
    typeof result.content === "string"
      ? result.content.trim()
      : originalQuery

  // Guardrail: kalau masih berbentuk kalimat jawaban
  if (
    text.toLowerCase().startsWith("saya") ||
    text.toLowerCase().startsWith("i ") ||
    text.endsWith("?") ||
    text.length > originalQuery.length * 2
  ) {
    return originalQuery
  }

  return text
}


// Main function that creates and runs the AI agent
export async function callAgent(
  client: MongoClient,
  query: string,
  thread_id: string
): Promise<AgentResponse> {
  try {
    // Database configuration
    const dbName = "admin_db"        // Name of the MongoDB database
    const db = client.db(dbName)              // Get database instance
    const collection = db.collection("items") // Get the 'items' collection

    // Define the state structure for the agent workflow
    const GraphState = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        // Reducer function: how to combine old and new messages
        reducer: (x, y) => x.concat(y), // Simply concatenate old messages (x) with new messages (y)
      }),
    })

    // Create a custom tool for searching furniture inventory
    const itemLookupTool = tool(
      // The actual function that will be executed when tool is called
      async ({ query, n = 10 }) => {
        try {
          console.log("Item lookup tool called with query:", query)

          // Check if database has any data at all
          const totalCount = await collection.countDocuments()
          console.log(`Total documents in collection: ${totalCount}`)

          // Early return if database is empty
          if (totalCount === 0) {
            console.log("Collection is empty")
            return JSON.stringify({ 
              error: "No items found in inventory", 
              message: "The inventory database appears to be empty",
              count: 0 
            })
          }

          // Get sample documents for debugging purposes
          const sampleDocs = await collection.find({}).limit(3).toArray()
          console.log("Sample documents:", sampleDocs)

          // Configuration for MongoDB Atlas Vector Search
          const dbConfig = {
            collection: collection,           // MongoDB collection to search
            indexName: "vector_index",       // Name of the vector search index
            textKey: "embedding_text",       // Field containing the text used for embeddings
            embeddingKey: "embedding",       // Field containing the vector embeddings
          }

          // Create vector store instance for semantic search using Google Gemini embeddings
          const vectorStore = new MongoDBAtlasVectorSearch(
            new GoogleGenerativeAIEmbeddings({
              apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment
              model: "text-embedding-004",         // Gemini embedding model
            }),
            dbConfig
          )

          console.log("Performing vector search...")
          // Perform semantic search using vector embeddings
          const result = await vectorStore.similaritySearchWithScore(query, n)
          console.log(`Vector search returned ${result.length} results`)
          
          // If vector search returns no results, fall back to text search
          if (result.length === 0) {
            console.log("Vector search returned no results, trying text search...")
            // MongoDB text search using regular expressions
            const textResults = await collection.find({
              $or: [ // OR condition - match any of these fields
                { item_name: { $regex: query, $options: 'i' } },        // Case-insensitive search in item name
                { item_description: { $regex: query, $options: 'i' } }, // Case-insensitive search in description
                { categories: { $regex: query, $options: 'i' } },       // Case-insensitive search in categories
                { embedding_text: { $regex: query, $options: 'i' } }    // Case-insensitive search in embedding text
              ]
            }).limit(n).toArray() // Limit results and convert to array
            
            console.log(`Text search returned ${textResults.length} results`)
            // Return text search results as JSON string
            return JSON.stringify({
              results: textResults,
              searchType: "text",    // Indicate this was a text search
              query: query,
              count: textResults.length
            })
          }

          // Return vector search results as JSON string
          return JSON.stringify({
            results: result,
            searchType: "vector",   // Indicate this was a vector search
            query: query,
            count: result.length
          })
          
        } catch (error: any) {
          // Log detailed error information for debugging
          console.error("Error in item lookup:", error)
          console.error("Error details:", {
            message: error.message,
            stack: error.stack,
            name: error.name
          })
          
          // Return error information as JSON string
          return JSON.stringify({ 
            error: "Failed to search inventory", 
            details: error.message,
            query: query
          })
        }
      },
      // Tool metadata and schema definition
      {
        name: "item_lookup",                                    // Tool name that the AI will reference
        description: "Gathers furniture item details from the Inventory database", // Description for the AI
        schema: z.object({                                      // Input validation schema
          query: z.string().describe("The search query"),      // Required string parameter
          n: z.number().optional().default(10)                 // Optional number parameter with default
            .describe("Number of results to return"),
        }),
      }
    )

    // Array of all available tools (just one in this case)
    const tools = [itemLookupTool]
    // Create a tool execution node for the workflow
    const toolNode = new ToolNode<typeof GraphState.State>(tools)

    // Initialize the AI model (Google's Gemini)
    const model = new ChatOpenAI({
      apiKey: process.env.GROQ_API_KEY,
      model: "openai/gpt-oss-120b",
      temperature: 0,
      configuration: {
        baseURL: "https://api.groq.com/openai/v1",
      },
    }).bindTools(tools)

    // Decision function: determines next step in the workflow
    function shouldContinue(state: typeof GraphState.State) {
      const messages = state.messages                               // Get all messages
      const lastMessage = messages[messages.length - 1] as AIMessage // Get the most recent message

      // If the AI wants to use tools, go to tools node; otherwise end
      if (lastMessage.tool_calls?.length) {
        return "tools"  // Route to tool execution
      }
      return "__end__"  // End the workflow
    }

    // Function that calls the AI model with retry logic
    async function callModel(state: typeof GraphState.State) {
      return retryWithBackoff(async () => { // Wrap in retry logic
        // Create a structured prompt template
        const prompt = ChatPromptTemplate.fromMessages([
          [
            "system", // System message defines the AI's role and behavior
            `You are a helpful E-commerce Sales Agent for a Home Furnishing & Furniture store.

IMPORTANT: You have access to an item_lookup tool that searches the furniture inventory database. ALWAYS use this tool when customers ask about furniture items, even if the tool returns errors or empty results.

LANGUAGE RULES:
- If the user's message is in Indonesian, respond fully in Indonesian.
- If the user's message is in English, respond in English.
- Use clear, natural, and friendly language.

TOOL USAGE (MANDATORY):
- You MUST always use the item_lookup tool when the user asks about furniture, products, prices, or recommendations.
- NEVER recommend products without calling the tool first.
- ONLY use products returned by the tool.
- DO NOT invent or assume products.

PRODUCT RULES:
- When a user asks for a category (e.g. sofa, kursi, meja):
  - Recommend 3–5 products from the SAME category.
  - The order of products in the array MUST match the tool results order exactly.
- Each recommended product must exist in the database.

RESPONSE FORMAT (STRICT – MUST FOLLOW):

You MUST return a VALID JSON object and NOTHING ELSE.

Return exactly this JSON structure:
- intro: your opening message to the user
- products: array of product objects with item_id, name, brand, description, price, original_price
- follow_up: one short follow-up question

Example JSON format (you must output valid JSON):
{{"intro": "Your message here", "products": [], "follow_up": "Your question?"}}

Rules:
- Do NOT include markdown
- Do NOT include explanations
- Do NOT include text outside JSON
- products MUST match the tool results order exactly
- follow_up must be ONE short question

IF NO PRODUCTS ARE FOUND:
- Politely inform the user that no products were found in the intro field.
- Set products to empty array [].
- Suggest refining the request in the follow_up field.

Current time: {time}`,
          ],
          new MessagesPlaceholder("messages"), // Placeholder for conversation history
        ])

        // Fill in the prompt template with actual values
        const formattedPrompt = await prompt.formatMessages({
          time: new Date().toISOString(), // Current timestamp
          messages: state.messages,       // All previous messages
        })

        // Call the AI model with the formatted prompt
        const result = await model.invoke(formattedPrompt)
        // Return new state with the AI's response added
        return { messages: [result] }
      })
    }

    // Build the workflow graph
    const workflow = new StateGraph(GraphState)
      .addNode("agent", callModel)                    // Add AI model node
      .addNode("tools", toolNode)                     // Add tool execution node
      .addEdge("__start__", "agent")                  // Start workflow at agent
      .addConditionalEdges("agent", shouldContinue)   // Agent decides: tools or end
      .addEdge("tools", "agent")                      // After tools, go back to agent

    // Initialize conversation state persistence
    const checkpointer = new MongoDBSaver({ client, dbName })
    // Compile the workflow with state saving
    const app = workflow.compile({ checkpointer })

    // Execute the workflow
    const rewrittenQuery = await reformulateQuery(query)

    console.log("Original query:", query)
    console.log("Reformulated query:", rewrittenQuery)

    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(rewrittenQuery)],
      },
      { 
        recursionLimit: 15,
        configurable: { thread_id: thread_id }
      }
    )

    // Extract the final response from the conversation
    const responseMessage = finalState.messages[finalState.messages.length - 1] as AIMessage

    // Parse AI structured JSON response
    let aiStructuredResponse: {
      intro: string
      products: any[]
      follow_up: string
    }

    try {
      aiStructuredResponse =
        typeof responseMessage.content === "string"
          ? JSON.parse(responseMessage.content)
          : responseMessage.content
    } catch (e) {
      throw new Error("AI response is not valid JSON")
    }

    console.log("AI structured response:", aiStructuredResponse)

        // Check if there were tool executions with product data
        let searchResults: SearchResults | null = null;
        for (let i = finalState.messages.length - 1; i >= 0; i--) {
          const msg = finalState.messages[i] as AIMessage;
          if (msg.tool_calls && msg.tool_calls.length > 0) {
            const toolResultMsg = finalState.messages[i + 1];
            if (toolResultMsg && toolResultMsg.content) {
              try {
                const contentStr = typeof toolResultMsg.content === 'string' 
                  ? toolResultMsg.content 
                  : JSON.stringify(toolResultMsg.content);
                searchResults = JSON.parse(contentStr);
                break;
              } catch (e) {
                console.error("Error parsing tool result:", e);
              }
            }
          }
        }
        
    // Parse and format product recommendations from tool results (single source of truth)
    const productData = extractProductsFromTool(searchResults)

    console.log("Has products:", productData.hasProducts)

    // Return structured AgentResponse
    return {
      intro: aiStructuredResponse.intro,
      products: productData.products, // from tool (single source of truth)
      followUp: aiStructuredResponse.follow_up,
      meta: {
        hasProducts: productData.hasProducts,
        searchType: productData.searchType,
        productCount: productData.products.length
      }
    }
        
    

  } catch (error: any) {
    // Handle different types of errors with user-friendly messages
    console.error("Error in callAgent:", error.message)
    
    if (error.status === 429) { // Rate limit error
      throw new Error("Service temporarily unavailable due to rate limits. Please try again in a minute.")
    } else if (error.status === 401) { // Authentication error
      throw new Error("Authentication failed. Please check your API configuration.")
    } else { // Generic error
      throw new Error(`Agent failed: ${error.message}`)
    }
  }
}
