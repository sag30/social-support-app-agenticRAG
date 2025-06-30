from chromadb import PersistentClient

# 1) Connect to your Chroma store
client = PersistentClient(path="chromadb_data")
collection = client.get_collection("support_documents")

# 2) Grab all IDs
all_ids = collection.get()["ids"]

# 3) Fetch docs + metadatas + embeddings
resp = collection.get(
    ids=all_ids,
    include=["documents", "metadatas", "embeddings"]
)

print("IDs:", resp["ids"])
print("Docs:", resp["documents"])
print("Metadatas:", resp["metadatas"])
print("Embeddings:", resp["embeddings"])