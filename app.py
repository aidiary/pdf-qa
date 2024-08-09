import os
import time
from uuid import uuid4

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "pdf-qa"

# もしindexが存在しない場合は作成
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

namespaces = set()

welcome_message = """Chainlit PDF QA デモへようこそ！はじめるには
1. PDFファイルまたはテキストファイルをアップロードしてください
2. ファイルについて質問してください
"""


def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    loader = Loader(file.path)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"

    return docs


def get_vector_store(file: AskFileResponse):
    docs = process_file(file)

    cl.user_session.set("docs", docs)

    namespace = file.name

    if namespace in namespaces:
        vector_store = PineconeVectorStore(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
    else:
        vector_store = PineconeVectorStore(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
        namespaces.add(namespace)

    return vector_store


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    print(file)

    msg = cl.Message(content=f"`{file.name}` を処理中です...")
    await msg.send()

    vector_store = await cl.make_async(get_vector_store)(file)
    print(vector_store)


@cl.on_message
async def on_message(message: cl.Message):
    pass
