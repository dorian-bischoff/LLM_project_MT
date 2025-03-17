#!bin/bash
conda create -n SNLP-project python==3.10 -y
conda activate SNLP-project
conda install numpy==2.2.4 matplotlib seaborn ipykernel ipywidgets tqdm scikit-learn -c conda-forge -y
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.49.0 evaluate peft==0.14.0 bitsandbytes accelerate==1.5.2 einops trl tf-keras
pip install rouge_score nltk git+https://github.com/google-research/bleurt.git sacrebleu unbabel-comet bert_score seaborn
python -m ipykernel install --user --name SNLP-project --display-name "SNLP environment"