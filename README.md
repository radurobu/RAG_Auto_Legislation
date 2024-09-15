# RAG_Auto_Legislation
A RAG implementation for Romanian language

This repository contains a local implementation of a Retrieval Augmented Generation (RAG) architecture for Romanian language. 
Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. The knowledge base for this implementation is the Romanian auto legislation and laws, the architecture should be capable of answering questions related to automotive laws and regulations.
As pre-trained models I used "dumitrescustefan/bert-base-romanian-uncased-v1" as the embedding model and as the generative model I used Llama3 ("meta-llama/Meta-Llama-3.1-8B-Instruct"). 

There are 2 main scripts in this codebase:
1) "main_rag_auto_pytorch.ipynb" wich is a local RAG implementation using pytorch, huggingface transformers library and a pytorch vectorestore.
2) "main_rag_auto_langchain.ipynb" wich is a local RAG implementation using huggingface transformers library, langchain and pinecone as the vectorestore.

Bellow you can see an example of the "RAG Auto Legislation" answering a legislation question.

![image](https://github.com/user-attachments/assets/ce2649a1-4ca6-4715-8dd7-c9bd32160b40)


