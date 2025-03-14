import os
import pickle
from tqdm import tqdm

import torch

import huggingface_hub
from datasets import load_dataset, Dataset
import transformers
from .general_utils import get_inp_tgt_lang
from .utils_dataset import transform_to_WNT_style

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# BERT embeddings generation
def load_bert():
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = transformers.BertModel.from_pretrained("bert-base-multilingual-uncased")
    return tokenizer, model

def predict_bert(tokenizer, model, sentence):
    with torch.no_grad():
        encoded_input = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
        output = model(**encoded_input.to(device))
    return output

def get_bert_embedding(tokenizer, model, sentence):
    out = predict_bert(tokenizer, model, sentence) # [1, ntokens, 768]
    out = out[0][:,  1:-1, :].cpu() # Remove CLS and SEP tokens -> [1, ntokens-2, 768]
    out = out.mean(dim=1) # [1, 768]
    return out

# Retrieval
def get_sorted_affinity_index(list_sentence):
    tokenizer, model = load_bert()
    model.to(device)
    bert_embeddings = []
    for sentence_to_translate in tqdm(list_sentence):
        bert_embeddings.append(get_bert_embedding(tokenizer, model, sentence_to_translate)) # [1, emb_size]
    model.cpu(); del model
    bert_embeddings = torch.cat(bert_embeddings, dim=0) # [n_sentences, emb_size]
    affinity = bert_embeddings @ bert_embeddings.T # [n_sentences, n_sentences]
    affinity -= torch.diagflat(torch.ones(affinity.shape[0])*float("Inf")) # Cancel self affinities

    sorted_affinity_index = []
    for i in range(len(list_sentence)):
        sorted_idx = torch.argsort(affinity[i],  descending = True).tolist()
        sorted_affinity_index.append(sorted_idx)
    return sorted_affinity_index

# Wrapper for generation
def get_sorted_affinity_index_path(direction, dataset):
    return f"./rag_retrieval_index/sorted_affinity_index_{dataset}_{direction}.pkl"

def generate_sorted_affinity_index(direction, dataset_name):
    # Load coresponding dataset
    if dataset_name == "flores":
        from credentials import hf_token
        huggingface_hub.login(token = hf_token)
        ds_flores = load_dataset("openlanguagedata/flores_plus")["devtest"]

    input_language, target_language = get_inp_tgt_lang(direction)
        
        # Getting the right split corresponding to the translation direction
    if dataset_name == "flores":
        ds = transform_to_WNT_style(ds_flores, lang=target_language, lang_start=input_language)
    elif dataset_name == "wnt23":
        if direction != "cs-en":
            ds = load_dataset("haoranxu/WMT23-Test", direction)["test"]
        else:
            ds = load_dataset("haoranxu/WMT23-Test", "en-cs")["test"]
            ds = Dataset.from_dict({f"cs-en": ds["en-cs"][::-1]}) # Reverse list to avoid having same sentences (if reduce_size not None)
    
    # Generate BERT embeddings, compute cross-similarity and rank them for each sequence
    list_source_sentences = [example[input_language] for example in ds[f"{input_language}-{target_language}"]]
    sorted_affinity_index = get_sorted_affinity_index(list_source_sentences) # list of a list (one per sentence) of ranked sentence index (based on decreasing similarities)
    
    sort_aff_idx_savepath = get_sorted_affinity_index_path(direction, dataset_name)
    if not os.path.exists("./rag_retrieval_index"):
        os.makedirs("./rag_retrieval_index")
    with open(sort_aff_idx_savepath, "wb") as f:
        pickle.dump(sorted_affinity_index, f, pickle.HIGHEST_PROTOCOL)

def get_closest_sentences(nb_sentences, sentence_idx, dataset, sorted_affinity_index):
    closest_sentences = []
    for i in sorted_affinity_index[sentence_idx][:nb_sentences]:
        closest_sentences.append(dataset[i])
    return closest_sentences