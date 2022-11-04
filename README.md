# Exploring Subgroup Performance in End-to-End Speech Models
Code associated with the paper "Exploring Subgroup Performance in End-to-End Speech Models".

In this repository, you will find the code to replicate our experiments.  
We do not include the dataset used in the paper as it is publicly available and downloadable from the official site. 

## Get Started
Our code was tested on Python 3.10.4. To make it work, you will need:
- a working environment with the libraries listed in `requirements.txt`;
- a functioning `torch` installation in the same environment.

## Running the Experiments
Use the `ic_fsc.ipynb` notebook to run the inference of the selected models on the FSC dataset, and to extract demographic, signal- and dataset- related metadata, building a `.csv` file such as the ones in the `data_precomputed` folder.

To reproduce the experiments of the paper, you can directly run the `divexplorer_analysis.ipynb` notebook, which leverages the files already computed in `data_precomputed`. 
