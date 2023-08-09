from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

data_path = 'data/'
db_faiss_path = "vectorstores/db_faiss"
print("777777")

## Create vector database
def creat_vector_db():
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'})
    
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(db_faiss_path)
    print("Done")

if __name__== '__main__':
    creat_vector_db()
