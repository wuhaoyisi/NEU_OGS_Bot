import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
# Function to read the text file
def read_file(file_path: str) -> list[str]:
    """
    Read the text file and return its content as a list of lines.

    :param file_path: Path to the text file
    :return: List of lines from the file with leading/trailing spaces removed
    """
    with open(file_path, 'r') as file:
        content = file.readlines()  # Read all lines from the file
    return [line.strip() for line in content]  # Remove leading/trailing spaces

# Function to get embeddings using SentenceTransformer
def get_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[list[float]]:
    """
    Retrieve embeddings for the provided texts using SentenceTransformer.

    :param texts: List of texts to embed
    :param model_name: SentenceTransformer model name to use (default is "all-MiniLM-L6-v2")
    :return: List of embeddings corresponding to the input texts
    """
    model = SentenceTransformer(model_name)  # Load the SentenceTransformer model
    embeddings = model.encode(texts)  # Generate embeddings
    return embeddings

# Initialize Pinecone client and create an index if it doesn't exist
def init_pinecone_and_create_index(api_key: str, index_name: str, dimension: int) -> None:
    """
    Initialize Pinecone client and create an index if it doesn't already exist.

    :param api_key: Pinecone API key
    :param index_name: Name of the index
    :param dimension: Dimension of the vectors to store
    """
    pc = Pinecone(api_key=api_key)  # Initialize Pinecone with API key

    # Check if the index already exists; if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,  # Model embedding dimension
            metric="cosine",  # Use cosine similarity as the metric
            spec=ServerlessSpec(
                cloud="aws",  # Cloud provider
                region="us-east-1"  # Region
            )
        )
        print(f"Index '{index_name}' created.")
    else:
        print(f"Index '{index_name}' already exists.")

# Function to upsert embeddings into Pinecone in batches
def upsert_embeddings_to_pinecone_in_batches(embeddings: list[list[float]], texts: list[str], index_name: str, batch_size: int = 1000) -> None:
    """
    Upsert embeddings and their corresponding texts into Pinecone in batches.

    :param embeddings: List of embeddings
    :param texts: List of original texts
    :param index_name: Name of the Pinecone index
    :param batch_size: Number of vectors to include in each batch (default is 1000)
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  # Initialize Pinecone client
    index = pc.Index(index_name)  # Get the Pinecone index

    # Batch and upsert data in chunks
    for i in range(0, len(embeddings), batch_size):
        batch_embeddings = embeddings[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]

        # Prepare data for upserting: vector ID, embedding, and metadata (text)
        upsert_data = [
            (str(i + j), embedding, {'text': batch_texts[j]}) for j, embedding in enumerate(batch_embeddings)
        ]

        # Perform the upsert operation for the current batch
        index.upsert(vectors=upsert_data)
        print(f"Batch {i // batch_size + 1} upserted successfully.")

# Main function to process the file, generate embeddings, and upsert to Pinecone
def process_and_save_to_pinecone(file_path: str, api_key: str, index_name: str, dimension: int) -> None:
    """
    Process the input file, generate embeddings, and upsert them to Pinecone.

    :param file_path: Path to the text file to process
    :param api_key: Pinecone API key
    :param index_name: Pinecone index name
    :param dimension: Dimension of the vectors (based on the model being used)
    """
    # Initialize Pinecone and create the index if it doesn't exist
    init_pinecone_and_create_index(api_key, index_name, dimension)

    # Read and process the file to extract text content
    file_content = read_file(file_path)

    # Get embeddings for the file content
    embeddings = get_embeddings(file_content)

    # Upsert embeddings into Pinecone in batches
    upsert_embeddings_to_pinecone_in_batches(embeddings, file_content, index_name)

# Example usage: Process the file and save embeddings to Pinecone
if __name__ == "__main__":
    file_path = '/Users/keith/Downloads/Projects/NEU_OGS_Bot/knowledge_base/crawler_txt_00/combined_content.txt'  # Replace with your file path
    pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Pinecone API key from environment
    index_name = "neu-ogs"  # Name of the Pinecone index
    model_dimension = 384  # Embedding dimension for the model (e.g., 384 for MiniLM)

    # Process the file and upsert the embeddings to Pinecone
    process_and_save_to_pinecone(file_path, pinecone_api_key, index_name, model_dimension)
