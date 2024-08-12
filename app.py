import os

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain import hub
from langchain.schema.runnable import RunnableConfig, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 事前にPineconeにログインしてインデックスを作成しておく
index_name = "pdf-qa"
index = pc.Index(index_name)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


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

    existing_namespaces = set(index.describe_index_stats()["namespaces"].keys())
    print("***", existing_namespaces)

    if namespace in existing_namespaces:
        vectorstore = PineconeVectorStore(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
    else:
        vectorstore = PineconeVectorStore(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
        vectorstore.add_documents(docs)

    return vectorstore


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

    msg = cl.Message(content=f"`{file.name}` を処理中です...")
    await msg.send()

    vectorstore = await cl.make_async(get_vector_store)(file)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "namespace": file.name})
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-4o-mini")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    msg.content = f"`{file.name}` の処理が完了しました！質問できます！"
    await msg.update()

    cl.user_session.set("chain", rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")

    msg = cl.Message(content="")

    async for chunk in chain.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
