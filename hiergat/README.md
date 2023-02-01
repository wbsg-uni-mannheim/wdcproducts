## Requirements 
* Python 3.7
* PyTorch 1.4
* HuggingFace Transformers
* NLTK (for 1-N ER problem)

Install required packages
```
conda env create -f hiergat_env.yml
```

## Activate corresponding environment 
```
conda deactivate 
conda activate hiergat_env
```

## To train and test hiergat
```
python all_runs.py
```