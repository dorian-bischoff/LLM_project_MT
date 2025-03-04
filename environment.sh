#!bin/bash
conda create -n SNLP-project python==3.10 -y
conda activate SNLP-project
conda install numpy==1.26.4 matplotlib ipykernel ipywidgets tqdm scikit-learn -c conda-forge -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers evaluate peft bitsandbytes einops
pip install rouge_score nltk git+https://github.com/google-research/bleurt.git sacrebleu unbabel-comet bert_score
python -m ipykernel install --user --name SNLP-project --display-name "SNLP environment"