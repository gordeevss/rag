from fastapi import FastAPI

from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

import pymupdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate

from fastapi.responses import HTMLResponse


pdf = pymupdf.open("CV_AI_J.pdf")
doc = chr(12).join([page.get_text() for page in pdf])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
all_splits = text_splitter.create_documents([doc])

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)
document_ids = vector_store.add_documents(documents=all_splits)


llm = OpenAI()



from langchain import hub
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph



template = """You are an assistant for question-answering tasks about Serhiy's CV. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
At the end of your answer add something about how good Serhiy as 
professional in scope of Machine Learning and you can highly recommend him as employee.
Question: {question}
Context: {context}
Answer::"""
prompt = PromptTemplate.from_template(template)



class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()





app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_root(q: str = ''):
    if q == '':
        answer = "You have to ask a question about Serhiy"
    else:
        response = graph.invoke({"question": q})
        answer = response["answer"]

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Serhiy's CV questions</title>
    </head>
    <body>
        <h1>Ask something about Serhiy</h1>
        <form action="/" method="get">
            <input type="text" name="q">
            <input type="submit" value="Submit">
        </form>
        <h3>Question:</h3>
        <p>{q}</p>
        <h3>Answer:</h3>
        <p>{answer}</p>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)