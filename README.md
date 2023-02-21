# The landscape of biomedical research
###### Rita González-Márquez, Luca Schmidt, Ben Schmidt, Philipp Berens & Dmitry Kobak

In this repository you can find the code associated to the paper "The landscape of biomedical research" (add URL).

![alt text](https://github.com/berenslab/pubmed-landscape/blob/main/results/figures/fig_1_general_embedding.png?raw=true)


### How to use this repository

The notebooks `01-15` contain the code to reproduce all the experiments and analyses performed in the paper. The notebooks `01-11` in the `figures` folder contain the code to generate the final figures included in the paper. All figures generated with the notebooks will be stored in the `results/figures` folder. All intermediate results will be saved in `results/variables` folder.

In order to be able to run `01-rgm-data-parse.ipynb`, the dataset needs to be first downloaded into the `data` directory from https://www.nlm.nih.gov/databases/download/pubmed_medline.html and then unziped. PubMed releases a new snapshot of their database every year; they call it a `baseline'. The data used in the paper is the 2020 baseline (download date: 26.01.2021, not available anymore) supplemented with additional files from the 2021 baseline (download date: 27.04.2022, not available anymore). However, any files from the currently available baseline could be used instead. For that the notebook `00_download_data.ipynb` (CHANGE NAME) can be used (it should be noted that the path to the pubmed baseline should be changed accordingly to which baseline you want to download).

As one runs the notebooks, the computed variables and intermediate results will be stored in the `results/variables` directory. As the dataset and the obtained variables are too large, they are not included in the repository. However, you can find the t-SNE embedding along with some metadata in here: https://zenodo.org/record/6684762.
