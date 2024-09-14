import os
import fitz
import time
import torch
from tqdm.auto import tqdm
from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain import PromptTemplate

import warnings
warnings.filterwarnings("ignore")

############################################################################ Low level Functions ##############################################################################################

def load_embedding_model(model_id: str, device: str) -> SentenceTransformer:
    embedding_model = SentenceTransformer(model_name_or_path=model_id, 
                                        device=device) # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    return embedding_model

def load_pdf_text(pdf_path):
  if not os.path.exists(pdf_path):
    print("File doesn't exist!")

  else:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
      text+=page.get_text()
  return text

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
    # Other potential text formatting functions can go here
    return cleaned_text

def chunk_text(text: str ,chunk_size: int, chunk_overlap: int) -> list:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        separators = ["(", ". ", ";"]
    )

    return text_splitter.create_documents([text])

def connect_to_pinecone() -> Pinecone:

    # configure client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud="aws", region="us-east-1")

    index_name = 'auto-rag-v3'

    existing_indexes = [
        index_info["name"] for index_info in pc.list_indexes()
    ]

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in existing_indexes:
        # if does not exist, create index
        pc.create_index(
            index_name,
            dimension=768,  # dimensionality of ada 002
            metric='cosine',
            spec=spec
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # connect to index
    index = pc.Index(index_name)
    time.sleep(1)

    return index

def upload_data_to_vectorstore(index: Pinecone, text_chunks: list, embedding_model, batch_size=100) -> None:

    print("Upoloading data to pinecone vectorestore...")
    for i in tqdm(range(0, len(text_chunks), batch_size)):
        i_end = min(len(text_chunks), i+batch_size)
        # get batch of data
        batch = text_chunks[i:i_end]
        # generate unique ids for each chunk
        ids = [str(n) for n in range(i, i_end)]
        # get text to embed
        texts = [x.page_content for x in batch]
        metadata = [x.metadata for x in batch]
        # embed text
        embeds = embedding_model.encode(texts)
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))

def query_vectorstore(query: str, index: Pinecone, text_chunks, embedding_model) -> list:
    context = []
    matches = index.query(vector = [embedding_model.encode(query).tolist()], top_k=10)
    for i in range(len(matches['matches'])):
        context.append(text_chunks[int(matches['matches'][i]['id'])].page_content)

    context = "- " + "\n- ".join([item for item in context])
    return context


def load_generative_model(model_id: str):
    # 1. Create quantization config for smaller model loading (optional)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)

    # 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
    print(f"[INFO] Using LLM model_id: {model_id}")

    # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model) 
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    # 4. Instantiate the model
    print('Loading LLM model.')
    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                    torch_dtype=torch.float16, # datatype to use, we want float16
                                                    quantization_config=quantization_config,
                                                    low_cpu_mem_usage=True, # use full memory 
                                                    attn_implementation="sdpa", # which attention version to use
                                                    token = os.getenv("Huggingface_token")) # need to pas your huggingface token for model access, AutoModelForCausalLM takes care of moving model lt cuda
        
    pipe = pipeline("text-generation", 
               model=llm_model, 
               tokenizer=tokenizer, 
               max_new_tokens=256
               )

    hf = HuggingFacePipeline(pipeline=pipe)
    return hf

def create_prompt():
    base_prompt = """Based on the following context items, please answer the query in romanian language.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""

    prompt = PromptTemplate(
        input_variables=['context', 'query'],
        template=base_prompt,
    )
    return prompt


############################################################################ High level Functions ##############################################################################################

def create_file_embedings(pdf_path: str, embedding_model_id: str):

    try:
        if next(embedding_model.parameters()).is_cuda: # Check if model allrady loaded
            print('Embedding model allrady on GPU.')
    except:
        print('Loading embedding model.')
        embedding_model = load_embedding_model(model_id=embedding_model_id, device="cuda")

    text = load_pdf_text(pdf_path)
    text = text_formatter(text)
    text_chunks = chunk_text(text=text, chunk_size=400, chunk_overlap=30)

    vectorstore = connect_to_pinecone()
    upload_data_to_vectorstore(vectorstore, text_chunks, embedding_model)
    return vectorstore, text_chunks, embedding_model

class Context:
    def __init__(self, input_query, vectorstore, text_chunks, embedding_model):
        self.input_query = input_query
        self.vectorstore = vectorstore
        self.text_chunks = text_chunks
        self.embedding_model = embedding_model

    def get_context(self, *args, **kwargs):
        # Process input arguments if needed (or just ignore them)
        return query_vectorstore(self.input_query, self.vectorstore, self.text_chunks, self.embedding_model)

def return_answer(input_query: str, generative_model_id: str, vectorstore, text_chunks, embedding_model) -> str:
    # Initialize context object
    cntx = Context(input_query, vectorstore, text_chunks, embedding_model)
    
    # Create prompt (ensure this is defined elsewhere)
    prompt = create_prompt()
    
    # Load generative model (ensure this is defined and returns the right model)
    generative_model = load_generative_model(generative_model_id)
    
    # Create a chain of operations using LangChain
    rag_chain = (
        RunnableParallel(context=cntx.get_context, query=RunnablePassthrough())
        | prompt
        | generative_model
        | StrOutputParser()
    )
    
    # Invoke the chain with the input query and return the output
    return rag_chain.invoke(input_query)

############################################################################ MAIN level Functions ##############################################################################################

def call_auto_rag_qa(input_query: str,
                     pdf_path: str,
                     embedding_model_id: str,
                     generative_model_id: str) -> None:
    
    vectorstore, text_chunks, embedding_model = create_file_embedings(pdf_path=pdf_path, embedding_model_id=embedding_model_id)
    answer = return_answer(input_query=input_query,
                            generative_model_id=generative_model_id,
                            vectorstore=vectorstore,
                            text_chunks=text_chunks,
                            embedding_model=embedding_model)
    
    print(f'The question is: {input_query}')
    print(f'Answer:')
    print(answer)