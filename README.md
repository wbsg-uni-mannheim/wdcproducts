# WDC Products: A Multi-Dimensional Entity Matching Benchmark

This repository contains the code and data download links to reproduce building the WDC Products benchmark. The benchmark files are available for direct download [here](http://webdatacommons.org/largescaleproductcorpus/wdc-products/)

* **Requirements**

    [Anaconda3](https://www.anaconda.com/products/individual)

    Please keep in mind that the code is not optimized for portable or even non-workstation devices. Some of the scripts may require large amounts of RAM (64GB+) and GPUs. It is advised to use a powerful workstation or server when experimenting with some of the larger files.

    The code has only been used and tested on Linux (CentOS) servers.

* **Building the conda environment**

    To build the exact conda environment used for the creation of WDC Products, navigate to the project root folder where the file *environment.yml* is located and run ```conda env create -f environment.yml```

* **Downloading the raw data files**

    Navigate to the *src/data/* folder and run ```python download_datasets.py``` to automatically download the source files into the correct locations.
    You can find the data at *data/interim/wdc-lspc/corpus/*

* **Building WDC Products**

    To reproduce building WDC Products, run the following notebooks in order. Please keep in mind that some notebooks may run for multiple hours.
    
    1. *notebooks/processing/benchmark2020/langdetect-and-clean.ipynb*
    2. *notebooks/processing/benchmark2020/dbscan-clustering.ipynb*
    3. *notebooks/processing/benchmark2020/generate-sets-final.ipynb*
	
* **Experiments**
	
	You need to install the project as a package. To do this, activate the environment, navigate to the root folder of the project, and run ```pip install -e .```
	
    * **Preparing the data**:

    To prepare the data for the experiments, run the first script below and then any of the others to prepare the data for the respective experiments. Make sure to navigate to the respective folders first. If you did not run the previous benchmark creation steps, you can obtain the needed data files on the WDC Products [page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/): 
    
    - *src/processing/preprocess/preprocess_wdcproducts.py*

    - *src/processing/contrastive/prepare-data.py*
    - *src/processing/process-magellan/process_to_magellan.py*
    - *src/processing/process-wordcooc/process-to-wordcooc.py*
    - *src/processing/process-wordocc/process-to-wordocc-multi.py*
    
    To prepare data for Ditto and HierGAT input run
    
    - *preprocess_data_for_ditto_hiergat.py*

    * **Running the experiments**:
	
		* **Ditto**:
	    First install the Ditto environment from the *ditto/ditto_env.yml* then run *ditto/all_runs.py*
	
		* **HierGAT**:
	    First install the HierGAT environment from the *hiergat/hiergat_env.yml* then run *hiergat/all_runs.py*
	
        * **Magellan**:
            Navigate to *src/models/magellan/* and run the script *run_magellan.py*

        * **Word Coocurrence**:
            Navigate to *src/models/wordcooc/* and run the script *run_wordcooc.py*

        * **Word Occurrence**:
            Navigate to *src/models/wordocc* and run the script *run_wordocc_multi.py*
	    

        * **Transformer**:

            Navigate to src/contrastive/
            
            To fine-tune a Transformer, run any of the fine-tuning scripts, e.g. for pair-wise:

            ```CUDA_VISIBLE_DEVICES="GPU_ID" bash lspc/run_finetune_baseline.sh roberta-base True 64 5e-05 wdcproducts80cc20rnd000un large```

            You need to specify model, usage of gradient checkpointing, batch size, learning rate, dataset and development size as arguments here.

            Analogously for fine-tuning a multi-class Transformer: 

            ```CUDA_VISIBLE_DEVICES="GPU_ID" bash lspc/run_finetune_baseline_multi.sh roberta-base True 64 5e-05 wdcproductsmulti80cc20rnd000un large```

        * **R-SupCon**:

            Navigate to src/contrastive/

            * **Contrastive Pre-training**:
	
                To run contrastive pre-training use e.g.

                ```CUDA_VISIBLE_DEVICES="GPU_ID" bash lspc/run_pretraining.sh roberta-base True 1024 5e-05 0.07 wdcproducts80cc20rnd000un large```

                You need to specify model, usage of gradient checkpointing, batch size, learning rate, temperature, dataset and development size as arguments here.

            * **Cross-entropy Fine-tuning**:
            
                Finally, to use the pre-trained models for fine-tuning, run any of the fine-tuning scripts, e.g. for pair-wise:

                ```CUDA_VISIBLE_DEVICES="GPU_ID" bash lspc/run_finetune_siamese.sh roberta-base True 1024 5e-05 0.07 frozen wdcproducts80cc20rnd000un large``` 

                Analogously for fine-tuning multi-class R-SupCon: 

                ```CUDA_VISIBLE_DEVICES="GPU_ID" bash lspc/run_finetune_multi.sh roberta-base True 1024 5e-05 0.07 frozen wdcproductsmulti80cc20rnd000un large```
		
	

    
    Result files can subsequently be found in the *reports* folder.


	
--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
