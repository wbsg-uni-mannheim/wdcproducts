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

    The code for running [Ditto](https://github.com/megagonlabs/ditto) and [HierGAT](https://github.com/CGCL-codes/HierGAT) is available in the respective repositories.
	
	Code to reproduce the word-(co)occurrence, Magellan, RoBERTa and R-SupCon experiments will be available here soon.

	
--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
