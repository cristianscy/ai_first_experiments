from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

# template = """Question: {question}
#
# Answer: Let's work this out in a step by step way to be sure we have the right answer."""

template = """Question: {question}

Answer: Act very kindly as a servant."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 8  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/home/cristian/development/ai/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "How will you clean my dirty feet?"
llm_chain.run(question)
