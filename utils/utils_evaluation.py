import os
import pickle
from typing import Union

import numpy as np

import torch

import huggingface_hub
from datasets import load_dataset, Dataset
import evaluate
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from sacrebleu.tokenizers.tokenizer_zh import TokenizerZh

from .eval_params import num_beams, temperature, max_new_tokens, top_p
from .utils_dataset import reduce_flores_to_some_languages, transform_to_WNT_style, reduce_dataset
from .general_utils import get_inp_tgt_lang, get_eval_filename, get_translations_filename
from .utils_generation import get_support_fn_benchmark

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Functions to compute metrics given:
#   the metric,
#   the list of initial sentence,
#   the list of target translations,
#   the list of translated sentences,
#   the target language

# Each metric is computed sample by sample. The output is a dictionnary containing
#   the full list of scores
#   the mean score
#   the standard deviation
#   the unbias standard deviation

def eval_rouge(metric, sources, targets, translation_infered, target_language):
    out_rouge = metric.compute(predictions=translation_infered,
                                  references=targets,
                                  use_aggregator=False)
    # For further statistical treatment
    results_rouge = {"rouge1": {},
                     "rouge2": {},
                     "rougeL": {},
                     "rougeLsum": {},}
    for key in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        results_rouge[key]["mean_score"] = np.mean(out_rouge[key]).item()
        results_rouge[key]["std_score"] = np.std(out_rouge[key]).item()
        results_rouge[key]["std_unbias_score"] = np.std(out_rouge[key], ddof=1).item()
    return results_rouge

def eval_bleu(metric, sources, targets, translation_infered, target_language):
    results_bleu = {"scores": [], "brevity_penalty": []}
    for trans, tgt in zip(translation_infered, targets):
        try:
            bleu_out = metric.compute(predictions=[trans],
                                    references=[[tgt]],
                                    tokenizer = TokenizerZh() if target_language=="zh" else Tokenizer13a())
        except ZeroDivisionError:
            bleu_out={"bleu": 0., "brevity_penalty": 0.}

        results_bleu["scores"].append(bleu_out["bleu"])
        results_bleu["brevity_penalty"].append(bleu_out["brevity_penalty"])
    # For further statistical treatment
    results_bleu["mean_score"] = np.mean(results_bleu["scores"]).item()
    results_bleu["std_score"] = np.std(results_bleu["scores"]).item()
    results_bleu["std_unbias_score"] = np.std(results_bleu["scores"], ddof=1).item()
    return {"bleu": results_bleu}

def eval_sacrebleu(metric, sources, targets, translation_infered, target_language):
    results_sacrebleu = {"scores": [], "brevity_penalty": []}
    for trans, tgt in zip(translation_infered, targets):
        try:
            sacrebleu_out = metric.compute(predictions=[trans],
                                            references=[[tgt]],
                                            tokenize = "zh" if target_language=="zh" else "13a")
        except ZeroDivisionError:
            sacrebleu_out = {"score": 0., "bp": 0.}
        results_sacrebleu["scores"].append(sacrebleu_out["score"])
        results_sacrebleu["brevity_penalty"].append(sacrebleu_out["bp"])
    # For further statistical treatment
    results_sacrebleu["mean_score"] = np.mean(results_sacrebleu["scores"]).item()
    results_sacrebleu["std_score"] = np.std(results_sacrebleu["scores"]).item()
    results_sacrebleu["std_unbias_score"] = np.std(results_sacrebleu["scores"], ddof=1).item()
    return {"sacrebleu": results_sacrebleu}

def eval_chrf_and_chrfplusplus(metric, sources, targets, translation_infered, target_language):
    results_chrf = {"scores": []}
    results_chrfplusplus = {"scores": []}
    for trans, tgt in zip(translation_infered, targets):
        try:
            chrf_out = metric.compute(predictions=[trans],
                                    references=[[tgt]],
                                    word_order=0,
                                    eps_smoothing=False)
        except ZeroDivisionError:
            chrf_out = {"score": 0.}
        try:
            chrfplusplus_out = metric.compute(predictions=[trans],
                                            references=[[tgt]],
                                            word_order=2,
                                            eps_smoothing=True)
        except ZeroDivisionError:
            chrfplusplus_out = {"score": 0.}
        results_chrf["scores"].append(chrf_out['score'])
        results_chrfplusplus["scores"].append(chrfplusplus_out['score'])
    # For further statistical treatment
    results_chrf["mean_score"] = np.mean(results_chrf["scores"]).item()
    results_chrf["std_score"] = np.std(results_chrf["scores"]).item()
    results_chrf["std_unbias_score"] = np.std(results_chrf["scores"], ddof=1).item()
    results_chrfplusplus["mean_score"] = np.mean(results_chrfplusplus["scores"]).item()
    results_chrfplusplus["std_score"] = np.std(results_chrfplusplus["scores"]).item()
    results_chrfplusplus["std_unbias_score"] = np.std(results_chrfplusplus["scores"], ddof=1).item()
    return {"chrf": results_chrf,
            "chrfplusplus": results_chrfplusplus}

def eval_comet(metric, sources, targets, translation_infered, target_language):
    results_comet = metric.compute(predictions=translation_infered,
                                         references=targets,
                                         sources=sources)
    # For further statistical treatment
    results_comet.update({"std_score": np.std(results_comet["scores"]).item(),
                          "std_unbias_score": np.std(results_comet["scores"], ddof=1).item()})
    return {"comet": results_comet}

def eval_bleurt(metric, sources, targets, translation_infered, target_language):
    results_bleurt = metric.compute(predictions=translation_infered,
                                    references=targets)
    # For further statistical treatment
    results_bleurt.update({"mean_score": np.mean(results_bleurt["scores"]).item(),
                           "std_score": np.std(results_bleurt["scores"]).item(),
                           "std_unbias_score": np.std(results_bleurt["scores"], ddof=1).item()})
    return {"bleurt": results_bleurt}

def eval_bertscore(metric, sources, targets, translation_infered, target_language):
    results_bert = metric.compute(predictions=translation_infered, references=targets, lang=target_language)
    # For further statistical treatment
    results_bert.update({"mean_score": np.mean(results_bert["f1"]).item(),
                         "std_score": np.std(results_bert["f1"]).item(),
                         "std_unbias_score": np.std(results_bert["f1"], ddof=1).item()})
    return {"bertscore": results_bert}

def eval_meteor(metric, sources, targets, translation_infered, target_language):
    results_meteor = {"scores": []}
    for trans, tgt in zip(translation_infered, targets):
        meteor_out = metric.compute(predictions=[trans],
                                    references=[tgt])
        results_meteor["scores"].append(meteor_out["meteor"])
    # For further statistical treatment
    results_meteor["mean_score"] = np.mean(results_meteor["scores"]).item()
    results_meteor["std_score"] = np.std(results_meteor["scores"]).item()
    results_meteor["std_unbias_score"] = np.std(results_meteor["scores"], ddof=1).item()
    return {"meteor": results_meteor}

def get_eval_fn(metric_name):
    if metric_name == "rouge":
        return eval_rouge
    elif metric_name == "bleu":
        return eval_bleu
    elif metric_name == "sacrebleu":
        return eval_sacrebleu
    elif metric_name == "chrf":
        return eval_chrf_and_chrfplusplus
    elif metric_name == "comet":
        return eval_comet
    elif metric_name == "bleurt":
        return eval_bleurt
    elif metric_name == "bertscore":
        return eval_bertscore
    elif metric_name == "meteor":
        return eval_meteor

def load_metric(metric_name):
    if metric_name == "rouge":
        return evaluate.load('rouge')
    elif metric_name == "bleu":
        return evaluate.load("bleu")
    elif metric_name == "sacrebleu":
        return evaluate.load("sacrebleu")
    elif metric_name == "chrf":
        return evaluate.load("chrf")
    elif metric_name == "comet":
        return evaluate.load('comet')
    elif metric_name == "bleurt":
        return evaluate.load('bleurt', 'bleurt-large-512')
    elif metric_name == "bertscore":
        return evaluate.load("bertscore")
    elif metric_name == "meteor":
        return evaluate.load('meteor')


def eval_one_metric_one_model(metric_name: str, metric, directions: list[str], dataset_name: str, model_name: str, model_size: Union[str, None], reduce_size: Union[int, None],
                              get_input_targets_fn = None,
                                translation_folder = None,
                              additionnal_name = None) -> None:
    """
    Compute the evaluation accoreding to one metric of one model on several directions
    Metric name should be in ["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum", "BLEU", "SacreBLEU", "chrF", "chrF++", "COMET", "BLEURT", "BERTscore", "METEOR"]
    metric if the huggingface metric return by evaluate.load()
    Refer to translations generation function for input_and_generate_fn

    Save directly the computed evaluations, returns None
    """
    # Getting right evaluation function
    metric_eval_fn = get_eval_fn(metric_name)

    # Loading full flores (if necessary)
    if dataset_name == "flores":
        from credentials import hf_token
        huggingface_hub.login(token = hf_token)
        ds_flores = load_dataset("openlanguagedata/flores_plus")["devtest"]
        ds_flores = reduce_flores_to_some_languages(ds_flores, directions)

    for direction in directions:
        print(f"Evaluating translations {direction} with model {model_name}"
              +(f"-{model_size}" if model_size is not None else "")
              +f" for dataset {dataset_name}...")
        input_language, target_language = get_inp_tgt_lang(direction)

        # Loading previous eval if existing
        eval_filename = get_eval_filename(direction, dataset_name, model_name, model_size, reduce_size, additionnal_name)
        if not os.path.exists(f"./evaluations"):
            os.makedirs(f"./evaluations")
        if os.path.exists(eval_filename):
            with open(eval_filename, "rb") as f:
                complete_eval = pickle.load(f)
        else:
            complete_eval = {}
        
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
        if get_input_targets_fn is None:
            get_input_targets_fn, _ = get_support_fn_benchmark(model_name)
        sources, inputs, targets = get_input_targets_fn(ds, input_language, target_language)
        print(f"Total number of samples: {len(sources)}" + ("" if reduce_size is None else f"; reduced to {reduce_size} (numpy seed = 42)"))
        if reduce_size is not None:
            # /!\ Use same reduce size and same seed to ensure sources and previous inputs are the same /!\
            sources, inputs, targets = reduce_dataset(sources, inputs, targets, reduce_size)

        # Loading precomputed translations
        translations_filename = get_translations_filename(direction, dataset_name, model_name, model_size, reduce_size, translation_folder)
        with open(translations_filename, "rb") as f:
            translation_pred = pickle.load(f)
        
        # Evaluation translation for this direction
        eval_dict = metric_eval_fn(metric, sources, targets, translation_pred, target_language)
        complete_eval.update(eval_dict)

        with open(eval_filename, "wb") as f:
            pickle.dump(complete_eval, f, pickle.HIGHEST_PROTOCOL)

# Wrapper to perform several evaluations
def eval_one_metric(metric_name, directions, dataset_names, model_names, model_sizes, reduce_sizes,
                    get_input_targets_fn = None,
                    translation_folder = None,
                    additionnal_name = None):
    print(f"Computing evaluations with {metric_name}...")
    metric = load_metric(metric_name)
    for dataset_name, reduce_size in zip(dataset_names, reduce_sizes):
        for model_name, model_size in zip(model_names, model_sizes):
            eval_one_metric_one_model(metric_name, metric, directions, dataset_name, model_name, model_size, reduce_size,
                                      get_input_targets_fn, translation_folder, additionnal_name)

def eval_metrics(metric_names, directions, dataset_names, model_names, model_sizes, reduce_sizes,
                 get_input_targets_fn = None,
                 translation_folder = None,
                 additionnal_name = None):
    for metric_name in metric_names:
        eval_one_metric(metric_name, directions, dataset_names, model_names, model_sizes, reduce_sizes,
                        get_input_targets_fn, translation_folder, additionnal_name)