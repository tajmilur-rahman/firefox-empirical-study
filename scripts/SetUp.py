from llama_index.core import Document, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

# Step 1: create an embedding model
embed_model = OpenAIEmbedding()

# Step 2: load documents
# Example: load text files from a folder
documents = SimpleDirectoryReader("../data/").load_data()

# Step 3: create the index with the embedding model
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Optional: use the index
query_engine = index.as_query_engine()
response = query_engine.query("What is this about?")
print(response)
