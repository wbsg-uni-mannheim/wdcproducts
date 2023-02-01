## Requirements
* Python 3.7.7
* PyTorch 1.9
* HuggingFace Transformers 4.9.2
* Spacy with the ``en_core_web_lg`` models

Install required packages
```
conda env create -f ditto_env.yml
conda activate ditto_env
python -m spacy download en_core_web_lg
```

## Activate corresponding environment 
```
conda deactivate 
conda activate ditto_env
```

## To train and test ditto
```
python all_runs.py
```