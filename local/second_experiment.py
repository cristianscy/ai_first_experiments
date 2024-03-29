from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

prompt_template = """Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Provide de answer in 5 lines of text.

Context: {context}

Question: {question}
Answer: """

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llama_model_path="/home/cristian/development/ai/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf"

n_gpu_layers = 8  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctxt = 2048

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=llama_model_path,
    n_gpu_layers=1,
    n_batch=n_batch,
    n_ctxt=n_ctxt,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

# Load PDF
loader = PyPDFLoader("./pdf/FINAL_REPORT.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
# # LlamaCppEmbeddings
# embeddings = LlamaCppEmbeddings(
#    model_path=llama_model_path,
#    n_gpu_layers=8,
#    n_batch=256
#)

# HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda:0'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db/chroma_db_hfembeddings")

# Retrieve and generate using the relevant snippets of the pdf
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Which antennas were used for doing the measurements?")

# cleanup
#vectorstore.delete_collection()
