from langchain  import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


db_faiss_path = "vectorstores/db_faiss"

custom_promt_template = """Use the following  pieces of onformation to answer the user's question.
 If you dont know the answer, please just say that you dont know the anser , dont try to make up as answer

Context : {context}
Question : {question}

Only returns the helpful answer below and nothing else."""

def se_custom_prompt():
    """
    prompt template for QA retrival for each vector stores"""

    promt =  PromptTemplate(template= custom_promt_template, input_variables=['context','question'])
    
    return promt

## Load LLM
def load_llm():
    llm = CTransformers(
        model= "llama-2-7b.ggmlv3.q8_0.bin", model_type= "llama", max_new_token = 512, temperature = 0.5
    )

    return llm

def retrival_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(search_kwargs = {'k':2}),
        return_source_documents = True,
        chain_type_kwargs= {'prompt' : prompt}
    )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'})
    db = FAISS.load_local(db_faiss_path,embeddings)
    llm = load_llm()
    qa_prompt = se_custom_prompt()
    qa = retrival_qa_chain(llm,qa_prompt, db)

    return qa
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response

####### chainlit code

@cl.on_chat_start
async def start():
    chain =  qa_bot()
    msg = cl.Message(content= " Starting the Bot......")
    await msg.send()
    msg.content = "What do you want to ask"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    chain =  qa_bot()
    cl.user_session.set("chain",chain)
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer= True, answer_prefix_tokens=["Final",'Answer']
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks = [cb])
    answer = res['result']
    sources = res['source_documents']

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\nNo Sources Found"
    
    await cl.Message(content=answer).send()
