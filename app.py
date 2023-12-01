
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

"""### Download Text File"""

# import requests
# url = "Minutes.txt"
# res = requests.get(url)
# with open("Minutes.txt", "w") as f:
#   f.write(res.text)

# Document Loader
from langchain.document_loaders import TextLoader
loader = TextLoader('Minutes.txt')
documents = loader.load()

# documents

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

print(wrap_text_preserve_newlines(str(documents[0])))

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# len(docs)

# docs[0]

"""### Embeddings"""

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()



# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)

query = "What did the president say about the Supreme Court"
docs = db.similarity_search(query)

print(wrap_text_preserve_newlines(str(docs[0].page_content)))

"""### Create QA Chain"""

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llms=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1, "max_length":512})

# chain = load_qa_chain(llms, chain_type="stuff")

# query = "What did the president say about the Supreme Court"
# docs = db.similarity_search(query)
# chain.run(input_documents=docs, question=query)

# query = "What did the president say about economy?"
# docs = db.similarity_search(query)
# chain.run(input_documents=docs, question=query)

import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

# from IPython.display import display
# import ipywidgets as widgets

# qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

# chat_history = []

# def on_submit(_):
#     query = input_box.value
#     input_box.value = ""

#     if query.lower() == 'exit':
#         print("Thank you for using the Multi-PDF Chatbot!")
#         return

#     result = qa({"question": query, "chat_history": chat_history})
#     chat_history.append((query, result['answer']))

#     display(widgets.HTML(f'<b>User:</b> {query}'))
#     display(widgets.HTML(f'<b><font color="blue">Chatbot:</font></b> {result["answer"]}'))

# print("Welcome to the Multi-PDF Chatbot! Type 'exit' to stop.")

# input_box = widgets.Text(placeholder='Please enter your question:')
# input_box.on_submit(on_submit)

# display(input_box)

from flask import Flask, render_template, request
from IPython.display import display
import ipywidgets as widgets

app = Flask(__name__)

# Create conversation chain that uses our vectordb as retriever, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(llms, db.as_retriever())
chat_history = []

def chatbot_response(query):
    if query.lower() == 'exit':
        return "Thank you for using our conversational chatbot!"

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    
    return result["answer"]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        response = chatbot_response(query)
        return render_template('index1.html', query=query, response=response)
    return render_template('index1.html')

if __name__ == '__main__':
    print("Welcome to the Transformers chatbot! Type 'exit' to stop.")
    app.run(debug=True)
