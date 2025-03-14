import numpy as np
from datasets import load_dataset, Dataset

def reduce_dataset(inputs: list[str], sources: list[str], targets, final_nb: list[str]) -> tuple[list[str], list[str], list[str]]:
    """
    Selects randomly the samples of the evaluation corpus
    """
    idx = np.arange(len(inputs))
    np.random.seed(42)
    idx = np.random.choice(idx, final_nb)
    return [inputs[i] for i in idx], [sources[i] for i in idx], [targets[i] for i in idx]

def reduce_flores_to_some_languages(ds_flores, directions: list[str]):
    """
    Extracts a subpart of FLORES dataset to group computations. Keep only the languages
    presents in directions
    Returns a dataset
    """
    print("Extracting all languages in directions from FLORES...")
    list_languages = []
    for direction in directions:
        lang1, lang2 = direction[0:2], direction[3:5]
        if lang1 not in list_languages:
            list_languages.append(lang1)
        if lang2 not in list_languages:
            list_languages.append(lang2)

    language_to_iso = {"en": "eng", "de": "deu", "cs": "ces", "is": "isl", "zh": "cmn", "ru": "rus"}
    ds_list = []
    for elem in ds_flores:
        for lang in list_languages:
            if elem["iso_639_3"] == language_to_iso[lang]:
                if lang == "zh":
                    if elem["glottocode"] == "beij1234":
                        ds_list.append(elem)
                else:
                    ds_list.append(elem)
    return Dataset.from_list(ds_list)

def transform_to_WNT_style(ds_flores, lang, lang_start="en"):
    """
    Convert FLORES dataset (or a fraction of it) to a dataset formatted as WNT23
    Returns a dataset
    """
    language_to_iso = {"en": "eng", "de": "deu", "cs": "ces", "is": "isl", "zh": "cmn", "ru": "rus"}
    list_sentence_lang, list_sentence_lang_start = [], []
    for elem in ds_flores:
        if elem["iso_639_3"] == language_to_iso[lang]:
            if lang == "zh":
                if elem["glottocode"] == "beij1234":
                    list_sentence_lang.append(elem["text"])
            else:
                list_sentence_lang.append(elem["text"])

        elif elem["iso_639_3"] == language_to_iso[lang_start]:
            if lang_start == "zh":
                if elem["glottocode"] == "beij1234":
                    list_sentence_lang_start.append(elem["text"])
            else:
                list_sentence_lang_start.append(elem["text"])
    assert len(list_sentence_lang) == len(list_sentence_lang_start)
    #print(f"Number of samples: {len(list_sentence_lang)}")
    final_text_list = []
    for i in range(len(list_sentence_lang)):
        final_text_list.append({f"{lang_start}": list_sentence_lang_start[i],
                                f"{lang}": list_sentence_lang[i],})
    return Dataset.from_dict({f"{lang_start}-{lang}": final_text_list})