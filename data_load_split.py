from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

def load_split_data (path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    #print("the page content is: ", document.page_content[:500])
    #print("the meta data is :", document.metadata)
    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50,
                                                   separators=["\n\n", "\n", "\. ", " ", ""])
    texts = text_splitter.split_documents(documents)

        # Step 3: Select the middle 300 chunks
    total_chunks = len(texts)
    if total_chunks < 300:
        print(f"Document has fewer than 300 chunks ({total_chunks} chunks available). Processing all chunks.")
        selected_texts = texts
    else:
        start = (total_chunks - 300) // 2
        selected_texts = texts[start: start + 300]
        print(f"Processing the middle 300 chunks from chunk {start} to {start + 300}.")


    print("=== Visualization of Selected Texts ===")
    print(f"Total Chunks Selected: {len(selected_texts)}")
    for i, chunk in enumerate(selected_texts[:3]):  
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content[:500])  

    return selected_texts

def embedding_faiss(texts):
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    text_list = [chunk.page_content for chunk in texts]  # Extract content
    vector_store = FAISS.from_texts(text_list, embedding_model)
    vector_store.save_local("faiss_index")
    print(f"FAISS index saved!")


if __name__ == "__main__":
    text = load_split_data("English_grammar.pdf")
    embedding_faiss(text)
