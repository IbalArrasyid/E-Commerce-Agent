// Import Google's Gemini embeddings for vector creation
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
// Import MongoDB client for database connection
import { MongoClient } from "mongodb"
// Import MongoDB Atlas vector search for storing and searching embeddings
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
// Import file system and path modules for reading JSON file
import { readFileSync } from "fs"
import { join } from "path"
// Import Zod for data schema validation and type safety
import { z } from "zod"
// Load environment variables from .env file (API keys, connection strings)
import "dotenv/config"

// Create MongoDB client instance using connection string from environment variables
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string)


// Define schema for furniture item structure using Zod validation (matching products_enriched.json)
// Initialize Google Gemini chat model for generating synthetic furniture data
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",  // Use Gemini 2.5 Flash model
  temperature: 0.7,               // Set creativity level (0.7 = moderately creative)
  apiKey: process.env.GOOGLE_API_KEY, // Google API key from environment variables
})

// Define schema for furniture item structure using Zod validation
const itemSchema = z.object({
  item_id: z.string(),                    // Unique identifier for the item
  item_name: z.string(),                  // Name of the furniture item
  item_description: z.string(),           // Detailed description of the item
  brand: z.string(),                      // Brand/manufacturer name
  prices: z.object({                      // Nested object for pricing information
    full_price: z.number(),               // Regular price
    sale_price: z.number(),               // Discounted price
  }),
  categories: z.array(z.string()),        // Array of category tags
  images: z.array(z.string()),            // Array of image URLs
  notes: z.string(),                      // Additional notes about the item
  raw_data_for_ai: z.string(),            // Raw data for AI processing
  ai_generated: z.object({                // AI-generated content (optional)
    style: z.union([z.string(), z.array(z.string())]), // Style can be string or array of strings
    description: z.string(),              // AI-generated description
    specs_summary: z.string(),            // Specifications summary
  }).optional(),                          // Make optional since some products don't have it
})

// Create TypeScript type from Zod schema for type safety
type Item = z.infer<typeof itemSchema>

// Function to load product data from products_enriched.json
async function loadProductsFromJson(): Promise<Item[]> {
  try {
    console.log("Loading products from products_enriched.json...")

    // Read the JSON file
    const jsonPath = join(__dirname, "products_enriched.json")
    const jsonData = readFileSync(jsonPath, "utf-8")

    // Parse JSON and validate with Zod schema
    const rawData = JSON.parse(jsonData)
    const validatedData = z.array(itemSchema).parse(rawData)

    console.log(`Successfully loaded ${validatedData.length} products from JSON file`)
    return validatedData
  } catch (error) {
    console.error("Error loading products from JSON:", error)
    throw error
  }
}

// Function to create database and collection before seeding
async function setupDatabaseAndCollection(): Promise<void> {
  console.log("Setting up database and collection...")

  // Get reference to the admin_db database
  const db = client.db("admin_db")

  // Create the items collection if it doesn't exist
  const collections = await db.listCollections({ name: "items" }).toArray()

  if (collections.length === 0) {
    await db.createCollection("items")
    console.log("Created 'items' collection in 'admin_db' database")
  } else {
    console.log("'items' collection already exists in 'admin_db' database")
  }
}

// Function to create vector search index
async function createVectorSearchIndex(): Promise<void> {
  try {
    const db = client.db("admin_db")
    const collection = db.collection("items")
    await collection.dropIndexes()
    const vectorSearchIdx = {
      name: "vector_index",
      type: "vectorSearch",
      definition: {
        "fields": [
          {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 768,
            "similarity": "cosine"
          }
        ]
      }
    }
    console.log("Creating vector search index...")
    await collection.createSearchIndex(vectorSearchIdx);

    console.log("Successfully created vector search index");
  } catch (e) {
    console.error('Failed to create vector search index:', e);
  }
}

async function loadProductData(): Promise<Item[]> {
  // Load products from the JSON file instead of generating synthetic data
  return await loadProductsFromJson()
}

// Function to create a searchable text summary from furniture item data
async function createItemSummary(item: Item): Promise<string> {
  // Return Promise for async compatibility (though this function is synchronous)
  return new Promise((resolve) => {
    // Join all categories into comma-separated string
    const categories = item.categories.join(", ")
    // Create basic item information string
    const basicInfo = `${item.item_name} ${item.item_description} from the brand ${item.brand}`
    // Format pricing information
    const price = `At full price it costs: ${item.prices.full_price} IDR, On sale it costs: ${item.prices.sale_price} IDR`
    // Get additional notes
    const notes = item.notes
    // Get AI-generated content if available
    const aiGenerated = item.ai_generated
    const style = aiGenerated?.style ? (Array.isArray(aiGenerated.style) ? aiGenerated.style.join(", ") : aiGenerated.style) : "Unknown"
    const aiDescription = aiGenerated?.description || ""

    // Combine all information into comprehensive summary for vector search
    const summary = `${basicInfo}. Categories: ${categories}. Price: ${price}. Notes: ${notes}. Style: ${style}. ${aiDescription}`

    // Resolve promise with complete summary
    resolve(summary)
  })
}

// Main function to populate database with AI-generated furniture data
async function seedDatabase(): Promise<void> {
  try {
    // Establish connection to MongoDB Atlas
    await client.connect()
    // Ping database to verify connection works
    await client.db("admin").command({ ping: 1 })
    // Log successful connection
    console.log("You successfully connected to MongoDB!")

    // Setup database and collection
    await setupDatabaseAndCollection()

    // Create vector search index
    await createVectorSearchIndex()

    // Get reference to specific database
    const db = client.db("admin_db")
    // Get reference to items collection
    const collection = db.collection("items")

    // Clear existing data from collection (fresh start)
    await collection.deleteMany({})
    console.log("Cleared existing data from items collection")

    // Load product data from JSON file
    const productData = await loadProductData()

    // Process each item: create summary and prepare for vector storage
    const recordsWithSummaries = await Promise.all(
      productData.map(async (record) => ({
        pageContent: await createItemSummary(record),  // Create searchable summary
        metadata: { ...record },                         // Preserve original item data
      }))
    )

    // Store each record with vector embeddings in MongoDB
    for (const record of recordsWithSummaries) {
      // Create vector embeddings and store in MongoDB Atlas using Google embeddings
      await MongoDBAtlasVectorSearch.fromDocuments(
        [record],                    // Array containing single record
        new GoogleGenerativeAIEmbeddings({            // Google embedding model
          apiKey: process.env.GOOGLE_API_KEY,         // Google API key
          modelName: "text-embedding-004",            // Google's standard embedding model (768 dimensions)
        }),
        {
          collection,                // MongoDB collection reference
          indexName: "vector_index", // Name of vector search index
          textKey: "embedding_text", // Field name for searchable text
          embeddingKey: "embedding", // Field name for vector embeddings
        }
      )

      // Log progress for each successfully processed item
      console.log("Successfully processed & saved record:", record.metadata.item_id)
    }

    // Log completion of entire seeding process
    console.log("Database seeding completed")

  } catch (error) {
    // Log any errors that occur during database seeding
    console.error("Error seeding database:", error)
  } finally {
    // Always close database connection when finished (cleanup)
    await client.close()
  }
}

// Execute the database seeding function and handle any errors
seedDatabase().catch(console.error)
