// Load environment variables from .env file (must be first import)
import 'dotenv/config'
// Import Express framework for creating web server
import express, { Express, Request, Response } from "express"
// Import MongoDB client for database connection
import { MongoClient } from "mongodb"
// Import Zod for schema validation
import { z } from "zod"
// Import our custom AI agent function
// OLD: LangGraph-based agent
// import { callAgent } from './agent'
// NEW: Hybrid AI + Logic agent
import { callAgent, getStateSummary, resetConversation } from './agent-hybrid'

// ============================================================================
// REQUEST VALIDATION SCHEMAS
// ============================================================================

// Schema for POST /chat - new conversation
const NewChatRequestSchema = z.object({
  message: z.string().min(1, "Message is required").max(1000, "Message too long")
})

// Schema for POST /chat/:threadId - continue conversation
const ContinueChatRequestSchema = z.object({
  message: z.string().min(1, "Message is required").max(1000, "Message too long")
})

// Type for validated request
interface ChatRequestBody {
  message: string
}

// Create Express application instance
const app: Express = express()
// Import CORS middleware for handling cross-origin requests
import cors from 'cors'
// Enable CORS for all routes (allows frontend to call this API)
app.use(cors())
// Enable JSON parsing for incoming requests (req.body will contain parsed JSON)
app.use(express.json({
  limit: '1mb',
  strict: true  // Only accept arrays and objects (no primitive JSON values)
}))

// Handle malformed JSON errors
app.use((err: any, req: Request, res: Response, next: any) => {
  if (err instanceof SyntaxError && 'body' in err) {
    console.error('[API] JSON parse error:', err)
    return res.status(400).json({ error: 'Invalid JSON format' })
  }
  next(err)
})

// Create MongoDB client using connection string from environment variables
const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string)

// Async function to initialize and start the server
async function startServer() {
  try {
    // Establish connection to MongoDB Atlas
    await client.connect()
    // Ping MongoDB to verify connection is working
    await client.db("db_admin").command({ ping: 1 })
    // Log successful connection
    console.log("You successfully connected to MongoDB!")

    // Define root endpoint (GET /) - simple health check
    app.get('/', (req: Request, res: Response) => {
      // Send simple response to confirm server is running
      res.send('Hybrid Chat Agent Server (AI + Deterministic Logic)')
    })

    // Debug endpoint - get conversation state (GET /chat/:threadId/state)
    app.get('/chat/:threadId/state', (req: Request, res: Response) => {
      const { threadId } = req.params
      const stateSummary = getStateSummary(threadId)
      res.json({ threadId, state: stateSummary })
    })

    // Reset conversation endpoint (DELETE /chat/:threadId)
    app.delete('/chat/:threadId', (req: Request, res: Response) => {
      const { threadId } = req.params
      resetConversation(threadId)
      res.json({ message: 'Conversation reset', threadId })
    })

    // Define endpoint for starting new conversations (POST /chat)
    app.post('/chat', async (req: Request, res: Response) => {
      // Validate request body
      const validationResult = NewChatRequestSchema.safeParse(req.body)

      if (!validationResult.success) {
        console.error("[API] Validation error:", validationResult.error.format())
        return res.status(400).json({
          error: 'Invalid request',
          details: validationResult.error.format()
        })
      }

      const { message: initialMessage } = validationResult.data

      // Generate unique thread ID using current timestamp
      const threadId = Date.now().toString()

      // Log the incoming message for debugging
      console.log(`[API] New chat request - Thread: ${threadId}, Message: "${initialMessage}"`)

      try {
        // Call our AI agent with the message and new thread ID
        const response = await callAgent(client, initialMessage, threadId)
        res.json({
          threadId,
          intro: response.intro,
          products: response.products,
          followUp: response.followUp,
          meta: response.meta
        })
      } catch (error) {
        // Log any errors that occur during agent execution
        console.error('[API] Error starting conversation:', error)
        // Send error response with 500 status code
        res.status(500).json({ error: 'Internal server error' })
      }
    })

    // Define endpoint for continuing existing conversations (POST /chat/:threadId)
    app.post('/chat/:threadId', async (req: Request, res: Response) => {
      // Extract thread ID from URL parameters
      const { threadId } = req.params

      // Validate request body
      const validationResult = ContinueChatRequestSchema.safeParse(req.body)

      if (!validationResult.success) {
        console.error("[API] Validation error:", validationResult.error.format())
        return res.status(400).json({
          error: 'Invalid request',
          details: validationResult.error.format()
        })
      }

      const { message } = validationResult.data

      console.log(`[API] Continue chat - Thread: ${threadId}, Message: "${message}"`)

      try {
        // Call AI agent with message and existing thread ID (continues conversation)
        const response = await callAgent(client, message, threadId)
        res.json({
          intro: response.intro,
          products: response.products,
          followUp: response.followUp,
          meta: response.meta
        })
      } catch (error) {
        // Log any errors that occur during agent execution
        console.error('[API] Error in chat:', error)
        // Send error response with 500 status code
        res.status(500).json({ error: 'Internal server error' })
      }
    })

    // Get port from environment variable or default to 8000
    const PORT = process.env.PORT || 8000
    // Start the Express server on specified port
    app.listen(PORT, () => {
      // Log that server is running and on which port
      console.log(`Server running on port ${PORT}`)
    })
  } catch (error) {
    // Handle any errors during server startup (especially MongoDB connection)
    console.error('Error connecting to MongoDB:', error)
    // Exit the process with error code 1 (indicates failure)
    process.exit(1)
  }
}

// Start the server (entry point of the application)
startServer()
