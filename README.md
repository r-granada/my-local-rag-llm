# My local RAG application

Simple repository to play around with local RAG implementations.

The objetive was to have a better understanding of the available frameworks at different levels of abstraction.
From high level like llama-index, to lower level building the code to interact with the model,
adding the necessary code to plug in the Vector database.

# Raw notes

Motives:
  - I know I won't get a better result locally than ChatGPT or any other service
  - I wanted to understand how RAG works and how to develop one
  - I wanted the experience of working with a model and how the optimizations work

Results:
  - Local Llama3 / nvidia-llama3 (8B) working on local from the command line
    - Initially was interested in having a UI, but decided it falls outside scope. Would only introduce noise.
  - Currently working on:
    - Keeping context in the conversation
    - RAG

What I have learned:
  - There are many optimizations and parameters to tune. Can be overwhelming in the beginning
  - My NVIDIA RTX 2060 doesn't eat Llama3-8B out of the box
  - Mostly been fighting to make it fit in my GPU
  - There is a lot of information out there, but at the same time it is not so easy to find a tutorial that directly fits your GPU

Links:
  - Chat tutorial: https://huggingface.co/docs/transformers/main/en/conversations
  - Generation of response algorithms and optimizations: https://huggingface.co/blog/how-to-generate
    - Greedy seach
    - Beam seach
    - Sampling
    - Top-k
    - Top-p
    - Best? --> Combination of Sampling + Top-k + Top-p
  - End-to-end training + inference: https://www.datacamp.com/tutorial/llama3-fine-tuning-locally
  - RAG example: https://github.com/leoneversberg/llm-chatbot-rag

