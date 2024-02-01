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

7.
