# Run LLM locally

## 01/02/2024

1. Download model https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF

2. To be able to use the model + llama-cpp-python + langchain follow this guide https://python.langchain.com/docs/integrations/llms/llamacpp

3. Try to install running `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python`

```
Building wheels for collected packages: llama-cpp-python
  Building wheel for llama-cpp-python (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Building wheel for llama-cpp-python (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [44 lines of output]
      *** scikit-build-core 0.8.0 using CMake 3.28.1 (wheel)
      *** Configuring CMake...
      loading initial cache file /tmp/tmp0ulps2gi/build/CMakeInit.txt
      -- The C compiler identification is GNU 11.4.0
      -- The CXX compiler identification is GNU 11.4.0
      -- Detecting C compiler ABI info
      -- Detecting C compiler ABI info - done
      -- Check for working C compiler: /usr/bin/cc - skipped
      -- Detecting C compile features
      -- Detecting C compile features - done
      -- Detecting CXX compiler ABI info
      -- Detecting CXX compiler ABI info - done
      -- Check for working CXX compiler: /usr/bin/c++ - skipped
      -- Detecting CXX compile features
      -- Detecting CXX compile features - done
      -- Found Git: /usr/bin/git (found version "2.34.1")
      -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
      -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
      -- Found Threads: TRUE
      -- Could not find nvcc, please set CUDAToolkit_ROOT.
      CMake Warning at vendor/llama.cpp/CMakeLists.txt:377 (message):
        cuBLAS not found


      -- CUDA host compiler is GNU
      CMake Error at vendor/llama.cpp/CMakeLists.txt:764 (get_flags):
        get_flags Function invoked with incorrect arguments for function named:
        get_flags


      -- Warning: ccache not found - consider installing it or use LLAMA_CCACHE=OFF
      -- CMAKE_SYSTEM_PROCESSOR: x86_64
      -- x86 detected
      CMake Warning (dev) at CMakeLists.txt:21 (install):
        Target llama has PUBLIC_HEADER files but no PUBLIC_HEADER DESTINATION.
      This warning is for project developers.  Use -Wno-dev to suppress it.

      CMake Warning (dev) at CMakeLists.txt:30 (install):
        Target llama has PUBLIC_HEADER files but no PUBLIC_HEADER DESTINATION.
      This warning is for project developers.  Use -Wno-dev to suppress it.

      -- Configuring incomplete, errors occurred!

      *** CMake configuration failed
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for llama-cpp-python
Failed to build llama-cpp-python
ERROR: Could not build wheels for llama-cpp-python, which is required to install pyproject.toml-based projects
```

4. Read https://installati.one/install-libcublas11-ubuntu-22-04/ and install libcublas11 `sudo apt -y install libcublas11` but same error

5. Read https://forums.developer.nvidia.com/t/installing-cuda-on-ubuntu-22-04-with-rtx-4090/232556 and install `sudo apt-get -y install nvidia-cuda-toolkit` and reboot computer

6. Rerun `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python` and now installed successfully

7. Try to run https://github.com/cristianscy/ai_first_experiments/blob/2f18fd0dfaf9d1ccf1cbef66bc8e9a68bc04e545/rag_pipeline_example.ipynb but error

```
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes
llama_model_loader: loaded meta data with 24 key-value pairs and 995 tensors from /home/cristian/development/ai/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = cognitivecomputations_dolphin-2.7-mix...
llama_model_loader: - kv   2:                       llama.context_length u32              = 32768
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   9:                         llama.expert_count u32              = 8
llama_model_loader: - kv  10:                    llama.expert_used_count u32              = 2
llama_model_loader: - kv  11:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  12:                       llama.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  13:                          general.file_type u32              = 15
llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,32002]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  16:                      tokenizer.ggml.scores arr[f32,32002]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  17:                  tokenizer.ggml.token_type arr[i32,32002]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 32000
llama_model_loader: - kv  20:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  21:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  22:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  23:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type  f16:   32 tensors
llama_model_loader: - type q8_0:   64 tensors
llama_model_loader: - type q4_K:  833 tensors
llama_model_loader: - type q6_K:    1 tensors
llm_load_vocab: special tokens definition check successful ( 261/32002 ).
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32002
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 32768
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 8
llm_load_print_meta: n_expert_used    = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 32768
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 46.70 B
llm_load_print_meta: model size       = 24.62 GiB (4.53 BPW)
llm_load_print_meta: general.name     = cognitivecomputations_dolphin-2.7-mixtral-8x7b
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 32000 '<|im_end|>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.76 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 25145.56 MiB on device 0: cudaMalloc failed: out of memory
llama_model_load: error loading model: failed to allocate buffer
llama_load_model_from_file: failed to load model
AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 |
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
Cell In[5], line 5
      2 n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
      4 # Make sure the model path is correct for your system!
----> 5 llm = LlamaCpp(
      6     model_path="/home/cristian/development/ai/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
      7     n_gpu_layers=n_gpu_layers,
      8     n_batch=n_batch,
      9     callback_manager=callback_manager,
     10     verbose=True,  # Verbose is required to pass to the callback manager
     11 )

File ~/development/pyenvs/ia_gft_pyenv/lib/python3.10/site-packages/langchain_core/load/serializable.py:107, in Serializable.__init__(self, **kwargs)
    106 def __init__(self, **kwargs: Any) -> None:
--> 107     super().__init__(**kwargs)
    108     self._lc_kwargs = kwargs

File ~/development/pyenvs/ia_gft_pyenv/lib/python3.10/site-packages/pydantic/v1/main.py:341, in BaseModel.__init__(__pydantic_self__, **data)
    339 values, fields_set, validation_error = validate_model(__pydantic_self__.__class__, data)
    340 if validation_error:
--> 341     raise validation_error
    342 try:
    343     object_setattr(__pydantic_self__, '__dict__', values)

ValidationError: 1 validation error for LlamaCpp
__root__
  Could not load Llama model from path: /home/cristian/development/ai/models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf. Received error  (type=value_error)
```

8. Downgrade `pip install --upgrade llama-cpp-python==0.2.26` and rerun but same error.

9. Download model https://huggingface.co/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF but same error.

10. Raised post https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF/discussions/3

11. Could run setting parameter `n_gpu_layers = 8`

```
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

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
question = "What are the steps to cook spaghettis?"
llm_chain.run(question)
```

## 02/02/2024

1. Try to follow this mini-project https://www.e2enetworks.com/blog/implementing-a-rag-pipeline-with-mixtral-8x7b

2. I don't understand what I am doing... :(

## 03/02/2024

1. As I don't understand what I was doing I decided to first follow this guide https://python.langchain.com/docs/use_cases/question_answering/ though it is paid, to understand the basics blocks of LLM Chains and afterwards be able to do something similar locally in my computer.

2. I found this URL https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa that is very interesting for implementing QA RAG using local models.
