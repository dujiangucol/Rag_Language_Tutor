

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


def generate_response(relevant_chunks, query, use_rag):
    system_prompt = """
    You are a knowledgeable and patient English language tutor specializing in grammar and language usage. 
    Your goal is to help learners understand complex concepts easily. When providing explanations, you should:

    - Begin with a concise summary of the answer.
    - Offer a detailed explanation using clear and simple language.
    - Provide examples to illustrate the concepts.
    - Encourage the learner and invite follow-up questions.

    Be polite and patient
    """

    # Construct the user prompt based on whether RAG is used
    if use_rag and relevant_chunks:
        context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        user_prompt = f"""
        Based on the following context, please answer the question.

        ### Context:
        {context}

        ### Question:
        {query}
        """
    else:
        user_prompt = f"""
        Please answer the following question in detail.

        ### Question:
        {query}
        """

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Generate the chat completion with adjusted parameters
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            model="gpt-4o-mini-2024-07-18",  # Use the GPT-4 model for better responses
            max_tokens=350,  # Increase to allow more detailed answers
            temperature=0.5,  # Lower temperature for more focused responses
            n=1,  # Generate one response
            stop=None,  # Let the model decide when to stop
        )
        response = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        response = f"An error occurred while generating the response: {e}"

    return response

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


    use_rag = True
    if use_rag:
        relevant_chunks = query_faiss(faiss_index_path, user_query)
    else:
        relevant_chunks = [] 

    response = generate_response(relevant_chunks, user_query, use_rag=use_rag)
    print("\n=== Generated Response ===")
    print(response)