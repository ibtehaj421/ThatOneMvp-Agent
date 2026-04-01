import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document

# 1. Initialize the local embedding model
# This will automatically utilize your GPU if CUDA is available, making ingestion blazingly fast.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2. Connect to the pgvector table
# Update with your actual local Postgres credentials
connection_string = "postgresql+psycopg2://postgres:postgres@localhost:5432/postgres"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="clinical_sops", 
    connection=connection_string,
    use_jsonb=True,
)

# 3. Create a sample localized SOP for our test
sample_sop_content = """
LEPTOSPIROSIS OUTBREAK TRIAGE PROTOCOL:
If a patient presents with sudden onset of high fever, severe myalgia (especially in the calves or lower back), and headache, immediately screen for leptospirosis.
Follow-up questions MUST include:
1. Have you been exposed to floodwaters, mud, or stagnant water in the past 14 days?
2. Are you experiencing any redness in the eyes (conjunctival suffusion)?
3. Have you noticed any decreased urine output or jaundice (yellowing of skin/eyes)?
If exposure is confirmed alongside myalgia and fever, flag for immediate physician review and consider initiating empiric doxycycline.
"""

# Create the document with organization metadata
doc = Document(
    page_content=sample_sop_content,
    metadata={
        "organization_id": 1, # Linking this SOP to a specific organization
        "title": "Leptospirosis Triage Guidelines"
    }
)

# 4. Ingest into Postgres
print("Generating embeddings and saving to pgvector...")
vector_store.add_documents([doc])
print("SOP ingested successfully.")