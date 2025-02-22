# General organisation
Since we all work on the same github (no branching), we need to organize to avoid having merging nightmare. The repo is organized as follows:
- /data folder (if necessary)
- /generated_translations : if you need to save resulting translation (whatever the end-use) - create an explicitly-named folder inside this one and dump your results in.
- /report : for our later-to-redact .tex report
- /results : to save our end-results (that we will include in the report)
- /personnal_nb : Folder for personnal notebooks - start your nb here and once finished, put or copy it in the root so everyone can access it. Ideally, notebooks in the root should not be modified (otherwise we will enter the merging nightmare). Note: this folder is in the .gitignore (precisely to avoid merging conflicts...).

In the root folder:
- notebooks containing our general pipelines. If we want to run them, best to copy in our personnal_nb folder and then run it. At least, copy it with a name including your initials so no one will run the same notebook at the same time on two different machines...
- create a credentials.py file and put your huggingface token in the first line `hf_token = "my_hf_token"`. This file is in the .gitignore (for obvious reasons) and the \_\_pycache\_\_ file as well. Once you will have requested access for the different Llama models (ask Llama 2, Llama 3.1 Instruct 1B) and for the FLORES+ dataset, you will be able to run the different cell using these objects.