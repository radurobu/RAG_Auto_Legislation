
import os
import fitz
from tqdm.auto import tqdm
import re
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import warnings
warnings.filterwarnings("ignore")

############################################################################ Low level Functions ##############################################################################################

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

def find_first_number(s): 
    match = re.search(r'\d+', s) 
    return int(match.group(0)) if match else None 

def open_and_read_pdf(text: str) -> list[dict]:
    split_text = []
    chapters = re.split("CAPITOLUL ", text)
    for i, chapter in enumerate(chapters):
        articols = re.split("Art.", chapter)
        for articol in articols:
            articol = text_formatter(articol)
            split_text.append({"Capitol": i,
                            "Articol": find_first_number(articol),
                            "text": articol})
    return split_text[2:]

def load_embedding_model(model_id: str, device: str) -> SentenceTransformer:
    embedding_model = SentenceTransformer(model_name_or_path=model_id, 
                                        device=device) # choose the device to load the model to (note: GPU will often be *much* faster than CPU)
    return embedding_model

def embed_text(text: list[dict], model: object) -> list[dict]:
    # Create embeddings one by one on the device
    for item in tqdm(text):
        item["embedding"] = model.encode(item["text"])
    return text

def save_embeddings_csv(text: list[dict], embeddings_df_save_path: str) -> None:
    # Save embeddings to file
    df_split_text = pd.DataFrame(text)
    df_split_text.to_csv(embeddings_df_save_path, index=False)

def import_embedding_csv(embeddings_df_save_path: str) -> pd.DataFrame:
    # Import saved file and view
    df_split_text = pd.read_csv(embeddings_df_save_path)
    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    df_split_text["embedding"] = df_split_text["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    embeddings = torch.tensor(np.array(df_split_text['embedding'].tolist()), dtype=torch.float32).to('cuda')
    split_text = df_split_text.to_dict(orient='records')
    return split_text, embeddings

def retrieve_relevant_indeces(input_query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer,
                                n_resources_to_return: int=5):
    
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Sanitze querry:
    input_query = text_formatter(input_query)

    # Embed the query
    query_embedding = model.encode(input_query,
                                    convert_to_tensor=True)

    # 3. Get similarity scores with the dot product (we'll time this for fun)
    dot_scores = util.cos_sim(a=query_embedding, b=embeddings)[0]

    # 4. Get the top-k results (we'll keep this to 5)
    scores, indices = torch.topk(input=dot_scores, 
                                 k=n_resources_to_return)
    return scores, indices

def retrieve_relevant_context(split_text, indices):
    # Create a list of context items
    context_items = [split_text[i]['text'] for i in indices]
    return context_items

def load_generative_model(model_id: str):
    # 1. Create quantization config for smaller model loading (optional)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)

    # 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
    print(f"[INFO] Using LLM model_id: {model_id}")

    # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model) 
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

    # 4. Instantiate the model
    try:
        if next(llm_model.parameters()).is_cuda: # Check if model allrady loaded
            print('LLM model allrady loaded in GPU.')
    except:
        print('Loading LLM model.')
        llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                        torch_dtype=torch.float16, # datatype to use, we want float16
                                                        quantization_config=quantization_config,
                                                        low_cpu_mem_usage=True, # use full memory 
                                                        attn_implementation="sdpa", # which attention version to use
                                                        token = os.getenv("Huggingface_token")) # need to pas your huggingface token for model access
    return tokenizer, llm_model

def make_query_prompt(query: str, 
                     context_items: list[dict],
                     tokenizer: SentenceTransformer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query in romanian language.
    Give yourself room to think by extracting relevant passages from the context before answering the query.
    Don't return the thinking, only return the answer.
    Make sure your answers are as explanatory as possible.
    \nNow use the following context items to answer the user query:
    {context}
    \nRelevant passages: <extract relevant passages from the context here>
    User query: {query}
    Answer:"""

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]

    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

def call_generative_model(prompt: str, tokenizer: object, llm_model: SentenceTransformer) -> str:
    # Tokenize the input text (turn it into numbers) and send it to GPU
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate outputs passed on the tokenized input
    outputs = llm_model.generate(**input_ids,
                                max_new_tokens=256) # define the maximum number of new tokens to create

    # Decode the output tokens to text
    outputs_decoded = tokenizer.decode(outputs[0])
    return outputs_decoded

############################################################################ High level Functions ##############################################################################################

def create_file_embedings(pdf_path: str, embedding_model_id: str):

    try:
        if next(embedding_model.parameters()).is_cuda: # Check if model allrady loaded
            print('Embedding model allrady on GPU.')
    except:
        print('Loading embedding model.')
        embedding_model = load_embedding_model(model_id=embedding_model_id, device="cuda")
    
    print(f"[INFO] Using embedding model_id: {embedding_model_id}")

    if not os.path.isfile("data/df_auto_embeding.csv"):
        text = load_pdf_text(pdf_path)
        split_text = open_and_read_pdf(text)
        split_text = embed_text(text=split_text, model=embedding_model)
        save_embeddings_csv(text=split_text, embeddings_df_save_path="data/df_auto_embeding.csv")
        
    split_text, embeddings = import_embedding_csv(embeddings_df_save_path="data/df_auto_embeding.csv")

    return split_text, embeddings, embedding_model

def get_context_items(input_query: str,
                      split_text: dict,
                      embeddings: SentenceTransformer,
                      embedding_model: SentenceTransformer,
                      n_resources_to_return: int):
    
    scores, indices = retrieve_relevant_indeces(input_query=input_query,
                                                embeddings=embeddings,
                                                model=embedding_model,
                                                n_resources_to_return=n_resources_to_return)
    
    context_items = retrieve_relevant_context(split_text, indices)

    return context_items

def generatre_answer(input_query: str,
                    context_items: list,
                    generative_model_id: str):
    
    llm_tokenizer, llm_model = load_generative_model(model_id=generative_model_id)
    query_prompt = make_query_prompt(query=input_query, context_items=context_items, tokenizer=llm_tokenizer)
    answer = call_generative_model(prompt=query_prompt, tokenizer=llm_tokenizer, llm_model=llm_model)

    print(answer)

############################################################################ MAIN level Functions ##############################################################################################

def call_auto_rag_qa(input_query: str,
                     pdf_path: str,
                     embedding_model_id: str,
                     generative_model_id: str) -> None:
    
    split_text, embeddings, embedding_model = create_file_embedings(pdf_path, embedding_model_id)
    context_items = get_context_items(input_query, split_text, embeddings, embedding_model, n_resources_to_return=5)
    generatre_answer(input_query, context_items, generative_model_id)