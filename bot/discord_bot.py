import discord
import os
import logging
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
from asyncio import TimeoutError
import time

# Load environment variables from .env file
load_dotenv()

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging with a timestamp and log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Discord bot setup with message intents for responding to user messages
intents = discord.Intents.default()
intents.messages = True
client = discord.Client(intents=intents)

# Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Pinecone setup: Initialize Pinecone instance using API key from environment
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "neu-ogs"

# Check if Pinecone index exists, if not, create a new one
# if index_name not in pc.list_indexes().names():
#     logging.info(f"Creating Pinecone index: {index_name}")
#     pc.create_index(
#         name=index_name, 
#         dimension=384, 
#         metric='cosine',  # Using cosine similarity for queries
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )

# Get the index instance for querying
index = pc.Index(index_name)

# Function to get embeddings using SentenceTransformer
def get_embeddings(texts):
    """
    Retrieve embeddings for the provided texts using SentenceTransformer.

    :param texts: List of texts to embed
    :return: List of embeddings
    """
    logging.info("Generating embeddings for texts")
    embeddings = model.encode(texts)  # Generate embeddings
    logging.info(f"Generated {len(embeddings)} embeddings")
    return embeddings

# Function to query Pinecone vector database with a relevance threshold
def query_vector_db(query, timeout=5, relevance_threshold=0.5):
    """
    Query the vector database with the provided query and apply a relevance threshold.

    :param query: The user's query string
    :param timeout: The time in seconds to wait for a response
    :param relevance_threshold: Minimum similarity score to consider a match relevant
    :return: List of relevant matches or None if no relevant matches
    """
    logging.info(f"Querying vector database with query: {query}")
    
    # Generate embedding for the query using SentenceTransformer
    embedding = get_embeddings([query])[0].tolist()  # Convert embedding to list

    try:
        # Adding a timeout to the Pinecone query
        results = index.query(vector=embedding, top_k=5, include_metadata=True, metric="cosine", timeout=timeout)
        matches = results.get('matches', [])

        # Filter matches by relevance threshold
        relevant_matches = [
            match['metadata']['text'] for match in matches
            if match['score'] >= relevance_threshold and 'metadata' in match and 'text' in match['metadata']
        ]

        if relevant_matches:
            logging.info(f"Found {len(relevant_matches)} relevant matches")
            return relevant_matches
        else:
            logging.info(f"No matches above relevance threshold of {relevance_threshold}")
            return None
    except TimeoutError:
        logging.error(f"Query to Pinecone timed out after {timeout} seconds")
        return None


# Azure OpenAI setup using the new style
client_openai = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version='2023-03-15-preview',
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Define the deployment name for the Azure OpenAI model
deployment_name = 'gpt-4'

# Function to get a response from Azure OpenAI model with retries and backoff
def get_llm_response(query, context=None, max_retries=3, backoff_factor=2):
    """
    Get a response from the Azure OpenAI model based on the query and context with retries.

    :param query: The user's question
    :param context: Additional context from the vector database
    :param max_retries: The maximum number of retries in case of failure
    :param backoff_factor: The factor by which to increase the wait time between retries
    :return: The generated response text
    """
    logging.info(f"Generating Azure LLM response for query: {query}")

    # System message with predefined guidelines
    system_message = {
    "role": "system",
    "content": """
You are NU Support Bot, a dedicated assistant for Northeastern University international students, specializing in the Office of Global Services (OGS). Use emojis where relevant to make your responses welcoming and clear, while maintaining a supportive and professional tone. Follow these guidelines:

1. Answer all OGS-related questions using relevant emojis (e.g., 🎓 for student queries, ✈️ for travel, 📝 for forms) to make your responses easier to follow.
2. For general inquiries about OGS services, provide concise answers with emojis like 📖 for guides or 🛂 for visa-related matters, helping users navigate easily.
3. Include step-by-step instructions for complex issues, using emojis to represent actions or outcomes (e.g., 🛠️ for troubleshooting or ✅ for completion).
4. Provide links to relevant resources, such as OGS webpages (https://international.northeastern.edu/ogs/) with emoji cues like 📎 or 🌐 to guide users to helpful information.
5. If a student's question requires further assistance, suggest they contact OGS via email or phone, using 🤝 to express support and 🙏 to empathize with any challenges they are facing.
6. Use friendly and inclusive language, along with supportive emojis (e.g., 🎉, 🚀) to encourage students and help them feel confident about navigating their questions.
7. Always maintain a professional yet approachable tone. Use emojis like 🌍 or 🏫 to emphasize the global and educational nature of your support.
"""
}

    user_message = {"role": "user", "content": query}
    
    # Limit context to 500 characters to avoid overly large inputs (change 3)
    # context = context[:500] if context else ""
    
    if context:
        context_message = {"role": "user", "content": f"Relevant information from our knowledge base: {context}"}
        messages = [system_message, context_message, user_message]
    else:
        messages = [system_message, user_message]

    retries = 0
    while retries < max_retries:
        try:
            # Call the Azure OpenAI API to get a response
            response = client_openai.chat.completions.create(
                model=deployment_name,
                messages=messages,
                max_tokens=300,  # Increased for more comprehensive responses
                temperature=0.3  # Lowered for more focused and accurate responses
            )
            result = response.choices[0].message.content.strip()  # Extract the response text
            return result
        except Exception as e:
            retries += 1
            wait_time = backoff_factor ** retries
            logging.error(f"Error with Azure OpenAI (attempt {retries}/{max_retries}): {str(e)}. Retrying in {wait_time} seconds.")
            time.sleep(wait_time)

    # If max retries are exceeded, return an error message (change 2)
    logging.error(f"Max retries reached. Unable to get a response from Azure OpenAI.")
    return "I apologize, but an error occurred while processing your request. Please try again later or contact our support team for assistance."

# Function to split long messages into smaller chunks (Discord limit: 2000 chars)
def split_message(message, max_length=2000):
    """
    Splits a message into chunks of a specified size.

    :param message: The original message to split
    :param max_length: Maximum length of each chunk (default is 2000 characters for Discord)
    :return: A list of message chunks
    """
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

# Event handler for when the Discord bot is ready
@client.event
async def on_ready():
    logging.info(f'Bot is logged in as {client.user}')

# Event handler for processing incoming messages
@client.event
async def on_message(message):
    if message.author == client.user:
        return  # Ignore messages from the bot itself

    if message.attachments:
        await message.reply("👋 Hi! It looks like you sent me an attachment. Unfortunately, I can't process attachments right now. Could you send me a text message instead? 😊 Thanks!")
        return  # Exit early

    user_question = message.content.strip()
    if not user_question:
        await message.reply("💬 I didn't catch that. Could you please ask your question again in text format? 📝")
        return  # Exit early if no text content

    logging.info(f"Received question from user: {user_question}")

    try:
        # Step 1: Query vector database for relevant context (no await here)
        relevant_info = query_vector_db(user_question)

        # Step 2: Use relevant info if found, else go directly to LLM
        context = " ".join(relevant_info) if relevant_info else ""
        if context:
            logging.info(f"Using context: {context}")
        else:
            logging.info("No relevant information found, exiting")
            await message.reply("🤔 It seems like I don't have any information about that topic at the moment. You can always check our documentation at https://international.northeastern.edu/ogs/ for more details. 📚 Let me know if there's anything else I can help with!")
            return

        # Get the response from the LLM (remove await here)
        response = get_llm_response(user_question, context)

        # Step 3: Split the response into chunks if it exceeds 2000 characters
        response_chunks = split_message(response)

        # Send each chunk separately
        for chunk in response_chunks:
            await message.reply(f"{chunk}")

    except Exception as e:
        logging.error(f"Error processing message: {str(e)}")
        await message.reply("⚠️ Oops! Something went wrong while processing your request. 😕 Please try again later or feel free to reach out to our support team for help. 🙏")

# Run the Discord bot with the token from the environment
client.run(os.getenv("DISCORD_BOT_TOKEN"))
