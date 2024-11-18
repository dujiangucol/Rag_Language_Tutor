from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_split_data (path):
    loader = PyPDFLoader(path)
    documents = loader.load()
    document = documents[30]
    print("the page content is: ", document.page_content[:500])
    print("the meta data is :", document.metadata)


    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Output the first chunk
    print(f"Number of chunks: {len(texts)}")
    print(f"Sample chunk: {texts[0].page_content}")

    #test
    first_chunk = texts[0].page_content
    words = first_chunk.split()[:10]  # Get the first 10 words
    print(f"First 10 words: {words}")

if __name__ == "__main__":
    load_split_data("Learn_to_Speak_Chinese.pdf")