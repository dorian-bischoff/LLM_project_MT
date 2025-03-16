from typing import Union


def get_inp_tgt_lang(direction: str) -> tuple[str, str]:
    """
    Return source and target language given a direction xx-yy
    """
    return direction[0:2], direction[3:5]

def get_translations_filename(direction: str, dataset_name: str, model_name: str, model_size: Union[str, None], reduce_size: Union[int, None], translation_folder: Union[str, None] = None) -> str:
    """
    Generate the pkl filename where to save the generated translations
    """
    mod_size = "-"+model_size if model_size is not None else ""
    translation_folder = "evaluations" if translation_folder is None else translation_folder
    return f"./generated_translations/{translation_folder}/{dataset_name}_{model_name}{mod_size}_{direction}_red-{reduce_size}.pkl"

def get_eval_filename(direction: str, dataset_name: str, model_name: str, model_size: Union[str, None], reduce_size: Union[int, None], additionnal_name: Union[str, None] = None) -> str:
    """
    Generate the pkl filename where to save the computed metrics
    """
    mod_size = "-"+model_size if model_size is not None else ""
    additionnal_name = "" if additionnal_name is None else f"_{additionnal_name}"
    return f"./evaluations/raw_{dataset_name}_{model_name}{mod_size}_{direction}_red-{reduce_size}{additionnal_name}.pkl"

def get_full_model_name(model_name, model_size, additionnal_name):
    return f"{model_name}"+(f"-{model_size}" if model_size is not None else "")+(f" - {additionnal_name}" if additionnal_name is not None else "")