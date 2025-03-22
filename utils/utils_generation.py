import os
import re
import pickle
from tqdm import tqdm
from typing import Union

import torch

import huggingface_hub
from datasets import load_dataset, Dataset
import transformers
from transformers import BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

from .eval_params import num_beams, temperature, max_new_tokens, top_p
from .utils_dataset import transform_to_WNT_style,  reduce_dataset
from .general_utils import get_inp_tgt_lang, get_translations_filename

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


#################################   NLLB

def get_input_targets_NLLB(dataset_wnt_format, source_lang, target_lang):
    inputs = [example[source_lang] for example in dataset_wnt_format[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset_wnt_format[f"{source_lang}-{target_lang}"]]
    return inputs, inputs, targets

def translate_list_of_str_NLLB(list_str, tokenizer, model, to_laguage):
    """
    Returns a list containing str corresponding to translation of the inputted
    """
    equivalence_language_to_FLORES = {"en": "eng_Latn", "de": "deu_Latn", "ru": "rus_Cyrl", "is": "isl_Latn", "zh": "zho_Hans", "cs": "ces_Latn"}
    with torch.no_grad():
        inputs = tokenizer(list_str, return_tensors="pt", padding=True)
        language_tgt_FLORES = equivalence_language_to_FLORES[to_laguage]
        translated = model.generate(inputs["input_ids"].to(device),
                                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(language_tgt_FLORES),
                                    num_beams=num_beams, max_length=max_new_tokens, early_stopping=True,
                                    ).cpu()
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text

def translate_batched_NLLB(inputs, model, tokenizer, batch_size, target_language):
    """
    For 8GB VRAM, use batch_size = 4
    For 16GB VRAM, use batch_size = 8 (better working with unbatch version to avoid pad noise).
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_NLLB(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   ALMA

def get_input_targets_ALMA(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [(
        f"Translate from {source_lang_name} to {target_lang_name}:"
        + f"\n{source_lang_name}: {example.get(source_lang)} \n{target_lang_name}:")
        for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def translate_list_of_str_ALMA(list_str, tokenizer, model, target_language):
    """
    Returns a list containing str corresponding to translation of the inputted
    """
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    with torch.no_grad():
        inputs = tokenizer(list_str, return_tensors="pt", padding=True)
        translated = model.generate(inputs["input_ids"].to(device),
                                    num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=temperature, top_p=top_p
                                    ).cpu()
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        tgt_language_name = language_name[target_language]
        translated_text = [t.split(f"{tgt_language_name}:")[2] for t in translated_text] # Remove prompt
    return translated_text

def translate_batched_ALMA(inputs, model, tokenizer, batch_size, target_language):
    """
    For 8GB VRAM, use batch_size=1
    For 16GB VRAM, use batch_size=3
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_ALMA(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   Llama 3

def get_input_targets_Llama3(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [
        [{"role": "system", "content": "You are a translator, you output only the translation in the desired language."},
         {"role": "user",
        "content": f"Translate from {source_lang_name} to {target_lang_name}:"
        + f"\n{source_lang_name}: {example.get(source_lang)} \n{target_lang_name}:"
        }] for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def extract_translation_Llama3(translated_prompt):
    answer = translated_prompt.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1]
    translation_only = answer.split("<|end_of_text|>")[0]
    translation_only = translation_only.split("<|eot_id|><|start_header_id|>assistant\n")[-1]
    translation_only = translation_only.split("<|eot_id|><|start_header_id|>")[-1]
    return translation_only

def translate_list_of_str_Llama3(list_str, tokenizer, model, target_language=None):
    with torch.no_grad():
        instruct_messages = tokenizer.apply_chat_template(list_str, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(instruct_messages, padding=True, padding_side='left', return_tensors="pt")
        out_tokens = model.generate(**tokens.to(device),
                                    num_beams=num_beams, do_sample=True,
                                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        translations = tokenizer.batch_decode(out_tokens)
        translations = [extract_translation_Llama3(trans) for trans in translations]
        return translations
    
def translate_batched_Llama3(inputs, model, tokenizer, batch_size, target_language):
    """
    For 8GB VRAM use
        batch_size=20 with Llama3 1B,
        batch_size=4 with Llama3 3B
    For 16 GB VRAM use 
        batch_size=40 with Llama3 1B,
        batch_size=10 with Llama3 3B,
        batch_size=5 with Llama3 8B,
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_Llama3(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language=None)
        preds.extend(tslt)
    return preds

#################################   Falcon 3 (Normal + Mamba)

def get_input_targets_Falcon3(dataset, source_lang, target_lang):
    """
    This function is valid for Falcon 3 and it mamba version
    """
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [
        [{"role": "system", "content": "You are a translator, you output only the translation in the desired language."},
         {"role": "user",
          "content": f"Translate from {source_lang_name} to {target_lang_name}:"
          + f"{example.get(source_lang)}"
        }] for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def extract_translation_Falcon3Mamba(translated_prompt):
    answer = translated_prompt.split("<|im_end|>\n<|im_start|>assistant\n")[-1]
    translation_only = answer.split("<|im_end|>")[0]
    return translation_only

def translate_list_of_str_Falcon3Mamba(list_str, tokenizer, model, target_language=None):
    with torch.no_grad():
        instruct_messages = tokenizer.apply_chat_template(list_str, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(instruct_messages, padding=True, padding_side='left', return_tensors="pt").to(model.device)
        out_tokens = model.generate(**tokens,
                                    num_beams=num_beams, do_sample=True,
                                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        translations = tokenizer.batch_decode(out_tokens)
        translations = [extract_translation_Falcon3Mamba(trans) for trans in translations]
        return translations
    
def translate_batched_Falcon3Mamba(inputs, model, tokenizer, batch_size, target_language=None):
    """
    For 16GB VRAM, use
        batch_size=4 with Falcon Mamba 7B (8 bits quantization),
        batch_size=4 with Falcon Mamba 7B (4 bits quantization),
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_Falcon3Mamba(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

def extract_translation_Falcon3(translated_prompt):
    answerpadded = translated_prompt.split("\n<|assistant|>\n")[-1]
    answer = answerpadded.split("<|pad|>")[-1]
    translation_only = answer.split("<|endoftext|>")[0]
    translation_only = re.sub(r"^[^a-zA-Z0-9]*", "", translation_only)
    return translation_only.replace("assistant|>\n", "")

def translate_list_of_str_Falcon3(list_str, tokenizer, model, target_language=None):
    with torch.no_grad():
        instruct_messages = tokenizer.apply_chat_template(list_str, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(instruct_messages, padding=True, padding_side='left', return_tensors="pt").to(model.device)
        out_tokens = model.generate(**tokens,
                                    num_beams=num_beams, do_sample=True,
                                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        translations = tokenizer.batch_decode(out_tokens)
        translations = [extract_translation_Falcon3(trans) for trans in translations]
        return translations
    
def translate_batched_Falcon3(inputs, model, tokenizer, batch_size, target_language=None):
    """
    For 16GB VRAM, use
        batch_size=8 with Falcon 7B (8 bits quantization),
        batch_size=4 with Falcon 3B,
        batch_size=12 with Falcon 1B
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_Falcon3(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   Qwen 2.5

def get_input_targets_Qwen2_5(dataset, source_lang, target_lang):
    """
    This function is valid for Falcon 3 and it mamba version
    """
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [
        [{"role": "system", "content": "You are a translator, you output only the translation in the desired language."},
         {"role": "user",
          "content": f"Translate from {source_lang_name} to {target_lang_name}:"
          + f"{example.get(source_lang)}"
        }] for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def extract_translation_Qwen2_5(translated_prompt):
    answerpadded = translated_prompt.split("\n<|im_start|>assistant\n")[-1]
    answer = answerpadded.split("<|im_end|>")[0]
    translation_only = answer.replace("<|endoftext|>", "")
    return translation_only

def translate_list_of_str_Qwen2_5(list_str, tokenizer, model, target_language=None):
    with torch.no_grad():
        instruct_messages = tokenizer.apply_chat_template(list_str, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(instruct_messages, padding=True, padding_side='left', return_tensors="pt").to(model.device)
        out_tokens = model.generate(**tokens,
                                    num_beams=num_beams, do_sample=True,
                                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        translations = tokenizer.batch_decode(out_tokens)
        translations = [extract_translation_Qwen2_5(trans) for trans in translations]
        return translations

def translate_batched_Qwen2_5(inputs, model, tokenizer, batch_size, target_language=None):
    """
    For 16GB VRAM, use batch_size=100 (up to)
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_Qwen2_5(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   Mistral


def get_input_targets_Mistral(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [
        [{"role": "system", "content": "You are a translator, you output only the translation in the desired language."},
         {"role": "user",
          "content": f"Translate from {source_lang_name} to {target_lang_name}:"
          + f"{example.get(source_lang)}"
        }] for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def extract_translation_Mistral(translated_prompt):
    answerpadded = translated_prompt.split("[/INST] ")[-1]
    answer = answerpadded.split("</s>")[0]
    return answer

def translate_list_of_str_Mistral(list_str, tokenizer, model, target_language=None):
    with torch.no_grad():
        instruct_messages = tokenizer.apply_chat_template(list_str, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(instruct_messages, padding=True, padding_side='left', return_tensors="pt").to(model.device)
        out_tokens = model.generate(**tokens,
                                    num_beams=num_beams, do_sample=True,
                                    temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
        translations = tokenizer.batch_decode(out_tokens)
        translations = [extract_translation_Mistral(trans) for trans in translations]
        return translations

def translate_batched_Mistral(inputs, model, tokenizer, batch_size, target_language=None):
    """
    For 16GB VRAM, use batch_size=2 (up to)
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_Mistral(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   BayLing

def get_input_targets_BayLing(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [(
        f"Translate from {source_lang_name} to {target_lang_name}:"
        + f"\n{source_lang_name}: {example.get(source_lang)} \n{target_lang_name}:")
        for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def translate_list_of_str_BayLing(list_str, tokenizer, model, target_language):
    """
    Returns a list containing str corresponding to translation of the inputted
    """
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    with torch.no_grad():
        inputs = tokenizer(list_str, return_tensors="pt", padding=True)
        translated = model.generate(inputs["input_ids"].to(device),
                                    num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=temperature, top_p=top_p
                                    ).cpu()
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        tgt_language_name = language_name[target_language]
        translated_text = [t.split(f"{tgt_language_name}:")[2] for t in translated_text] # Remove prompt
    return translated_text

def translate_batched_BayLing(inputs, model, tokenizer, batch_size, target_language):
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_BayLing(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   BLOOM & BLOOMZ

def get_input_targets_BLOOM(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [(
        f"Translate from {source_lang_name} to {target_lang_name}:"
        + f"\n{source_lang_name}: {example.get(source_lang)} \n{target_lang_name}:")
        for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def translate_list_of_str_BLOOM(list_str, tokenizer, model, target_language):
    """
    Returns a list containing str corresponding to translation of the inputted
    """
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    with torch.no_grad():
        inputs = tokenizer(list_str, return_tensors="pt", padding=True)
        translated = model.generate(inputs["input_ids"].to(device),
                                    num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=temperature, top_p=top_p
                                    ).cpu()
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        tgt_language_name = language_name[target_language]
        translated_text = [t.split(f"{tgt_language_name}:")[2] for t in translated_text] # Remove prompt
    return translated_text

def translate_batched_BLOOM(inputs, model, tokenizer, batch_size, target_language):
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_BLOOM(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   OPT & OPT Instruct

def get_input_targets_OPT(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [(
        f"Translate from {source_lang_name} to {target_lang_name}:"
        + f"\n{source_lang_name}: {example.get(source_lang)} \n{target_lang_name}:")
        for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def translate_list_of_str_OPT(list_str, tokenizer, model, target_language):
    """
    Returns a list containing str corresponding to translation of the inputted
    """
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    with torch.no_grad():
        inputs = tokenizer(list_str, return_tensors="pt", padding=True)
        translated = model.generate(inputs["input_ids"].to(device),
                                    num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=temperature, top_p=top_p
                                    ).cpu()
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        tgt_language_name = language_name[target_language]
        translated_text = [t.split(f"{tgt_language_name}:")[2] for t in translated_text] # Remove prompt
        translated_text = [t.split(f"\n[END]")[0] for t in translated_text]
    return translated_text

def translate_batched_OPT(inputs, model, tokenizer, batch_size, target_language):
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_OPT(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   MPT

def get_input_targets_MPT(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use the instruct template
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [(
        "Below is an instruction that describes a task. Write a response that appropriately completes the request. \n### Instruction:"
        + f"Translate from {source_lang_name} to {target_lang_name}: {example.get(source_lang)}"
        + "\n### Response:")
        for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def translate_list_of_str_MPT(list_str, tokenizer, model, target_language=None):
    """
    Returns a list containing str corresponding to translation of the inputted
    """
    with torch.no_grad():
        inputs = tokenizer(list_str, return_tensors="pt", padding=True)
        translated = model.generate(inputs["input_ids"].to(device),
                                    num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=temperature, top_p=top_p
                                    ).cpu()
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translated_text = [t.split("\n### Response:")[-1] for t in translated_text] # Remove prompt
    return translated_text

def translate_batched_MPT(inputs, model, tokenizer, batch_size, target_language):
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_MPT(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds

#################################   Finetuned LLAMA NI
def get_input_targets_LLAMA_finetuned(dataset, source_lang, target_lang):
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    source_lang_name = language_name[source_lang]
    target_lang_name = language_name[target_lang]
    # Use base formulation "Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
    sources = [example[source_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    inputs = [(
        f"Translate the following text from {source_lang_name} to {target_lang_name}:"
        + f"\n{source_lang_name}: {example.get(source_lang)} \n{target_lang_name}:")
        for example in dataset[f"{source_lang}-{target_lang}"]]
    targets = [example[target_lang] for example in dataset[f"{source_lang}-{target_lang}"]]
    return sources, inputs, targets

def translate_list_of_str_LLAMA_finetuned(list_str, tokenizer, model, target_language):
    """
    Returns a list containing str corresponding to translation of the inputted
    """
    language_name = {"en": "English", "de": "German", "ru": "Russian", "is": "Islandic", "zh": "Chinese", "cs": "Czech"}
    with torch.no_grad():
        inputs = tokenizer(list_str, return_tensors="pt", padding=True)
        translated = model.generate(inputs["input_ids"].to(device),
                                    attention_mask=inputs["attention_mask"].to(device),
                                    num_beams=num_beams, 
                                    do_sample=True,
                                    temperature=temperature, 
                                    top_p=top_p,
                                    pad_token_id=tokenizer.pad_token_id
                                    ).cpu()
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        tgt_language_name = language_name[target_language]
        translated_text = [t.split(f"{tgt_language_name}:")[2] for t in translated_text] # Remove prompt
    return translated_text

def translate_batched_LLAMA_finetuned(inputs, model, tokenizer, batch_size, target_language):
    """
    For 8GB VRAM, use batch_size=1
    For 16GB VRAM, use batch_size=3
    """
    preds = []
    for i in tqdm(range(len(inputs)//batch_size)):
        tslt = translate_list_of_str_LLAMA_finetuned(inputs[i*batch_size : (i+1)*batch_size], tokenizer, model, target_language)
        preds.extend(tslt)
    return preds


def load_model_benchmark(model_name: str, model_size: Union[str, None] = None) -> tuple:
    """
    Load model and tokenizer for the models considered in the benchmark
    Returns (tokenizer, model)
    """
    if model_name == "alma":
        tokenizer = transformers.LlamaTokenizer.from_pretrained("haoranxu/ALMA-7B", padding_side='left')
        Q_config = BitsAndBytesConfig(load_in_8bit=True) 
        model = transformers.AutoModelForCausalLM.from_pretrained("haoranxu/ALMA-7B", torch_dtype="auto", device_map=device, quantization_config=Q_config)
        
    elif model_name == "nllb":
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype="auto", device_map=device)

    elif model_name == "llama3":
        from credentials import hf_token
        huggingface_hub.login(token = hf_token)
        if model_size=="1B" or model_size=="3B":
            tokenizer = transformers.AutoTokenizer.from_pretrained(f"meta-llama/Llama-3.2-{model_size}-Instruct")
            model = transformers.AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-3.2-{model_size}-Instruct", torch_dtype="auto", device_map=device)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            nQ_cofig = BitsAndBytesConfig(load_in_8bit=True)
            model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", torch_dtype="auto", device_map=device, quantization_config=Q_config)
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    elif model_name == "llama3-NI-4bit":
        from credentials import hf_token
        huggingface_hub.login(token = hf_token)
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        Q_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=getattr(torch, "float16"),
                                      bnb_4bit_use_double_quant=False)
        model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", torch_dtype="auto", device_map=device, quantization_config=Q_config)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
    elif model_name == "falcon3-mamba":
        tokenizer = transformers.AutoTokenizer.from_pretrained("tiiuae/Falcon3-Mamba-7B-Instruct")
        Q_config = BitsAndBytesConfig(load_in_8bit=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-Mamba-7B-Instruct", torch_dtype="auto", device_map=device, quantization_config=Q_config)
    
    elif model_name == "falcon3":
        if model_size=="1B" or model_size=="3B":
            tokenizer = transformers.AutoTokenizer.from_pretrained(f"tiiuae/Falcon3-{model_size}-Instruct")
            model = transformers.AutoModelForCausalLM.from_pretrained(f"tiiuae/Falcon3-{model_size}-Instruct", torch_dtype="auto", device_map=device)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Instruct")
            Q_config = BitsAndBytesConfig(load_in_8bit=True)
            model = transformers.AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Instruct", torch_dtype="auto", device_map=device, quantization_config=Q_config)
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        
    elif model_name == "qwen2.5":
        if model_size=="0.5B" or model_size=="1.5B" or model_size=="3B":
            tokenizer = transformers.AutoTokenizer.from_pretrained(f"Qwen/Qwen2.5-{model_size}-Instruct")
            model = transformers.AutoModelForCausalLM.from_pretrained(f"Qwen/Qwen2.5-{model_size}-Instruct", torch_dtype="auto", device_map=device)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            Q_config = BitsAndBytesConfig(load_in_8bit=True)
            model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map=device, quantization_config=Q_config)
    
    elif model_name == "mistral":
        from credentials import hf_token
        huggingface_hub.login(token = hf_token)
        tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
        tokenizer.pad_token = tokenizer.eos_token
        Q_config = BitsAndBytesConfig(load_in_8bit=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", torch_dtype="auto", device_map=device, quantization_config=Q_config)
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    elif model_name == "bayling":
        tokenizer = transformers.AutoTokenizer.from_pretrained("ICTNLP/bayling-2-7b")
        tokenizer.pad_token = tokenizer.eos_token
        Q_config = BitsAndBytesConfig(load_in_8bit=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("ICTNLP/bayling-2-7b", torch_dtype="auto", device_map=device, quantization_config=Q_config)
    
    elif model_name == "bloom":
        if model_size=="0.5B":
            tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-560m")
            model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", torch_dtype=torch.bfloat16, device_map=device)
        elif model_size=="1B":
            tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-1b7")
            model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7", torch_dtype="auto", device_map=device)
        elif model_size=="3B":
            tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-3b")
            model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloom-3b", torch_dtype="auto", device_map=device)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
            Q_config = BitsAndBytesConfig(load_in_8bit=True)
            model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", torch_dtype="auto", device_map=device, quantization_config=Q_config)

    elif model_name == "bloomz":
        if model_size=="1B":
            tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloomz-1b7")
            model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloomz-1b7", torch_dtype="auto", device_map=device)
        elif model_size=="3B":
            tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloomz-3b")
            model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloomz-3b", torch_dtype="auto", device_map=device)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
            Q_config = BitsAndBytesConfig(load_in_8bit=True)
            model = transformers.AutoModelForCausalLM.from_pretrained("bigscience/bloomz-7b1", torch_dtype="auto", device_map=device, quantization_config=Q_config)
    
    elif model_name == "opt":
        if model_size=="0.1B":
            tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-125m")
            model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto", device_map=device)
        elif model_size=="0.3B":
            tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-350m")
            model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype="auto", device_map=device)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-6.7b")
            Q_config = BitsAndBytesConfig(load_in_8bit=True)
            model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype="auto", device_map=device, quantization_config=Q_config)
    
    elif model_name == "opt-instruct":
        tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-iml-1.3b")
        model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-iml-1.3b", torch_dtype="auto", device_map=device)
    
    elif model_name == "mpt":
        tokenizer = transformers.AutoTokenizer.from_pretrained("mosaicml/mpt-7b-instruct")
        tokenizer.pad_token = tokenizer.eos_token
        Q_config = BitsAndBytesConfig(load_in_8bit=True)
        model = transformers.AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b-instruct", torch_dtype="auto", device_map=device, quantization_config=Q_config)
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    elif model_name == "finetuned_llama3-3B":
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )
        model = AutoPeftModelForCausalLM.from_pretrained(
            "results/parallelData_finetuned-Llama3.2-3B",
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        raise ValueError("Not supported model. Try:[finetuned_llama3-3B, mpt, opt-instruct, bloomz, mistral, alma, nllb...]")
        
    return tokenizer, model

def get_support_fn_benchmark(model_name: str) -> tuple:
    """
    Return functions to generate rightly formatted inputs and a function to use
    translation models in inference for the models considered in the benchmark
    """
    if model_name == "alma":
        get_input_targets_fn = get_input_targets_ALMA
        tslt_fn = translate_batched_ALMA
        
    elif model_name == "nllb":
        get_input_targets_fn = get_input_targets_NLLB
        tslt_fn = translate_batched_NLLB

    elif model_name == "llama3":
        get_input_targets_fn = get_input_targets_Llama3
        tslt_fn = translate_batched_Llama3
    
    elif model_name == "llama3-NI-4bit":
        get_input_targets_fn = get_input_targets_LLAMA_finetuned
        tslt_fn = translate_batched_LLAMA_finetuned
    
    elif model_name == "falcon3-mamba":
        get_input_targets_fn = get_input_targets_Falcon3
        tslt_fn = translate_batched_Falcon3Mamba
    
    elif model_name == "falcon3":
        get_input_targets_fn = get_input_targets_Falcon3
        tslt_fn = translate_batched_Falcon3
    
    elif model_name == "qwen2.5":
        get_input_targets_fn = get_input_targets_Qwen2_5
        tslt_fn = translate_batched_Qwen2_5
    
    elif model_name == "mistral":
        get_input_targets_fn = get_input_targets_Mistral
        tslt_fn = translate_batched_Mistral
    
    elif model_name == "bayling":
        get_input_targets_fn = get_input_targets_BayLing
        tslt_fn = translate_batched_BayLing

    elif model_name == "bloom" or model_name == "bloomz":
        get_input_targets_fn = get_input_targets_BLOOM
        tslt_fn = translate_batched_BLOOM
    
    elif model_name == "opt" or model_name == "opt-instruct":
        get_input_targets_fn = get_input_targets_OPT
        tslt_fn = translate_batched_OPT
    
    elif model_name == "mpt":
        get_input_targets_fn = get_input_targets_MPT
        tslt_fn = translate_batched_MPT
    elif model_name == "finetuned_llama3-3B":
        get_input_targets_fn = get_input_targets_LLAMA_finetuned
        tslt_fn = translate_batched_LLAMA_finetuned
    else:
        raise ValueError("Not supported model. Try:[finetuned_llama3-3B, mpt, opt-instruct, bloomz, mistral, alma, nllb...]")
    return get_input_targets_fn, tslt_fn



# Main function to generate translations
def generate_translation_different_directions(directions: list[str],
                                              dataset_name: str,
                                              model_name: str,
                                              batch_size: int,
                                              reduce_size: Union[int, None] = None,
                                              model_size: Union[str, None] = None,
                                              load_model_and_tokenizer_fn = load_model_benchmark,
                                              get_input_targets_fn = None,
                                              tslt_fn = None,
                                              translation_folder = None) -> None:
    """
    Inputs:
        - directions: list of strings
        - dataset name: str, either "flores" or "wnt23"
        - model_name: str
        - batch_size: int (advised 1 to avoid padding - or make sure your tokenizer is correctly parametrized)
        - reduce_size: int, the number of random samples to use. Samples are sampled using seed to have same
          samples for each models. If reduce_size=None, take all the dataset samples.
        - model_size: str or None

        - load_model_and_tokenizer_fn: a function returning a tuple
          SIGNATURE : load_model_and_tokenizer_fn(model_name: str, model_size: Union[str, None]) -> tokenizer, model

        - get_input_targets_fn: a function returning a tuple of three lists of str gicen the dataset and the source and target language:
          SIGNATURE : get_input_targets_fn(ds: HF_dataset, input_language: str, target_language: str) -> sources, inputs, targets: list[str], list[str], list[str]
            sources are the initial sentences (used in COMET metric)
            inputs are the complete prompts to the model (only one string, apply the instruct template in get_input_targets_fn)
            targets are the target translations

        - tslt_fn: a function prompting the model and generating the a list of translations given a list of prompt, the tokenizer and the model.
          It must include a batch_size argument (the batched processing is not necessary to implement in the function). Include also the
          target_language as argument for consistency with other functions.
          SIGNATURE : tslt_fn(inputs: list[str], model: HF_model, tokenizer: HF_tokenizer, batch_size: int, target_language: Union[str, None]) -> translation_pred: list[str]
    """
    
    # Loading full flores (if necessary)
    if dataset_name == "flores":
        from credentials import hf_token
        huggingface_hub.login(token = hf_token)
        ds_flores = load_dataset("openlanguagedata/flores_plus")["devtest"]

    # Loading corresponding model
    print("Loading model...")
    tokenizer, model = load_model_and_tokenizer_fn(model_name, model_size)
    if get_input_targets_fn is None:
        get_input_targets_fn, tslt_fn = get_support_fn_benchmark(model_name)

    for direction in directions:
        print(f"Translating {direction} with model {model_name}"
              +(f"-{model_size}" if model_size is not None else "")
              +f" for dataset {dataset_name}...")
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
        # Extracting input & targets
        sources, inputs, targets = get_input_targets_fn(ds, input_language, target_language)
        print(f"Total number of samples: {len(sources)}" + ("" if reduce_size is None else f"; reduced to {reduce_size} (numpy seed = 42)"))
        if reduce_size is not None:
            sources, inputs, targets = reduce_dataset(sources, inputs, targets, reduce_size)
        translation_pred = tslt_fn(inputs, model, tokenizer, batch_size, target_language)

        # Saving translations
        translation_folder = "evaluations" if translation_folder is None else translation_folder
        if not os.path.exists(f"./generated_translations/{translation_folder}"):
            os.makedirs(f"./generated_translations/{translation_folder}")
        translations_filename = get_translations_filename(direction, dataset_name, model_name, model_size, reduce_size, translation_folder)
        with open(translations_filename, "wb") as f:
            pickle.dump(translation_pred, f, pickle.HIGHEST_PROTOCOL)

    # De-load model from GPU to enable calling this function with another model without restarting kernel
    model.cpu()
    del model, tokenizer

# Wrappers for several models and several datasets
def generate_translation_several_models(directions, dataset_name, model_names, model_sizes, batch_size, reduce_size,
                                        load_model_and_tokenizer_fn = load_model_benchmark,
                                        get_input_targets_fn = None,
                                        tslt_fn = None,
                                        translation_folder = None) -> None:
    for model_name, model_size in zip(model_names, model_sizes):
        generate_translation_different_directions(directions=directions,
                                                dataset_name=dataset_name,
                                                model_name=model_name,
                                                model_size=model_size,
                                                batch_size=batch_size,
                                                reduce_size=reduce_size,
                                                load_model_and_tokenizer_fn = load_model_and_tokenizer_fn,
                                                get_input_targets_fn = get_input_targets_fn,
                                                tslt_fn = tslt_fn,
                                                translation_folder = translation_folder)
        
def generate_translation_several_datasets(directions, dataset_names, model_names, model_sizes, batch_size, reduce_size,
                                          load_model_and_tokenizer_fn = load_model_benchmark,
                                          get_input_targets_fn = None,
                                          tslt_fn = None,
                                          translation_folder = None) -> None:
    for dataset_name in dataset_names:
        generate_translation_several_models(directions, dataset_name, model_names, model_sizes, batch_size, reduce_size,
                                            load_model_and_tokenizer_fn = load_model_and_tokenizer_fn,
                                            get_input_targets_fn = get_input_targets_fn,
                                            tslt_fn = tslt_fn,
                                            translation_folder = translation_folder)