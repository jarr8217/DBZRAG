import os 

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


FILENAME = 'dragon_ball_characters.txt'
PERSIST_DIR = './chroma_dragonball'

MODEL_ID = 'llama3.2:3b'
OLLAMA_URL = 'http://localhost:11434'
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
HISTORY_TURNS = 10

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ('system',
     'Given the chat history and the lastest user question, reqrite the question '
     'so that it can be understood without the chat history. do NOT answer the '
     'question, If it already stands on its own, return it unchanged. '
     'if the question is not related to the context, return "I do not know".'),
     MessagesPlaceholder('chat_history'),
     ('human', '{input}'),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ('system,'
    'You answer questions about Dragon Ball characters using only the context '
    'below. If the answer is not in the context, say you do not know. '
    'Do no make anything up. \n\n'
    'Context:\n{context}'),
    MessagesPlaceholder('chat_history'),
    ('human', '{input}'),
])

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# load the persistent vector store, if it does not exist, it creates it from the text file
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.isdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    if not os.path.exists(FILENAME):
        raise FileNotFoundError(
            f"{FILENAME} not found in the current directory. "
            "It must be in the same folder as the main script or FILENAME "
            "must be set to the file's full path."
        )

    print("Building index (first run only)...")
    docs = TextLoader(FILENAME, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    texts = splitter.split_documents(docs)
    return Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIR)

def build_chain(vectorstore):
    llm = ChatOllama(model=MODEL_ID,
                temperature=0,
                num_predict=512,
                base_url=OLLAMA_URL,
                )

    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONDENSE_PROMPT
    )

    answer_chain = create_stuff_documents_chain(llm, ANSWER_PROMPT)

    return create_retrieval_chain(history_aware_retriever, answer_chain)

def main():
    chain = build_chain(get_vectorstore())
    print("Welcome to the Dragon Ball Character Q&A! Type 'exit, quit, or bye' to quit.\n")

    while True:
        query = input("Ask a question: ").strip()

        if not query: continue

        if query.lower() in {'quit', 'exit', 'bye'}:
            print("Goodbye!")
            break

        result = chain.invoke({'input': query, 'chat_history': []})
        print('Answer: ', result['answer'], '\n')

if __name__ == '__main__':
    main()

