from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your environment variables.")

# Set tokenizer parallelism flag to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_response(relevant_chunks, query, use_rag=True):


    if use_rag:
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        system_message = "You are a helpful assistant with expertise in grammar and English language."
        user_message = f"""
        Context: {context}
        
        Question: {query}
        """
    else:
        system_message = "You are a helpful assistant with expertise in grammar and English language."
        user_message = f"Question: {query}"

    client = OpenAI(
        api_key=api_key
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        model="gpt-4o-mini-2024-07-18",  # Specify the model
        max_tokens=300,
        temperature=0.7,
    )

    return chat_completion.choices[0].message.content


def query_faiss(faiss_index_path, query):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever()
    relevant_chunks = retriever.get_relevant_documents(query)

    print(f"\n=== Retrieved {len(relevant_chunks)} Relevant Chunks ===")
    for i, chunk in enumerate(relevant_chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content[:500])  # Print first 500 characters for visualization

    return relevant_chunks


if __name__ == "__main__":
    faiss_index_path = "faiss_index"
    user_query = "What is the difference between present continuous and simple present?"


    use_rag = False
    if use_rag:
        relevant_chunks = query_faiss(faiss_index_path, user_query)
    else:
        relevant_chunks = [] 

    response = generate_response(relevant_chunks, user_query, use_rag=use_rag)
    print("\n=== Generated Response ===")
    print(response)
