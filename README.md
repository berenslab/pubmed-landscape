# The landscape of biomedical research
###### Rita González-Márquez, Luca Schmidt, Benjamin M. Schmidt, Philipp Berens & Dmitry Kobak

In this repository you can find the code associated to the paper ["The landscape of biomedical research"](https://doi.org/10.1016/j.patter.2024.100968).

Here you can find the interactive visualization: [https://static.nomic.ai/pubmed.html](https://static.nomic.ai/pubmed.html)

![alt text](https://github.com/berenslab/pubmed-landscape/blob/main/results/figures/final/fig_1_general_embedding.png?raw=true)


## How to use this repository

The notebooks `01-18` contain the code to reproduce all the experiments and analyses performed in the paper. In particular, the notebooks `02-03` contain all the steps for generating the 2D embedding from raw text. The notebooks `01-16` in the `scripts/figure-scripts/` folder contain the code to generate the final figures included in the paper. All figures generated with the notebooks will be stored in the `results/figures/final` folder. The notebooks `01-09` in the `scripts/BERT-based-embeddings/` folder contain the comparisons between different BERT-based models and TF-IDF in a subset of the data $(n=1$ M) and some additional analyses.

In order to be able to run `01-rgm-data-parse.ipynb`, the dataset needs to be first downloaded into the `data/` directory from https://www.nlm.nih.gov/databases/download/pubmed_medline.html and then unziped. PubMed releases a new snapshot of their database every year; they call it a "baseline". The data used in the paper is the 2020 baseline (download date: 26.01.2021, not available anymore) supplemented with additional files from the 2021 baseline (download date: 27.04.2022, not available anymore). However, any files from the currently available baseline could be used instead. For that the notebook `00-rgm-data-download.ipynb` can be used (it should be noted that the path to the pubmed baseline should be changed accordingly to which baseline you want to download). Also, the FTP connection breaks after downloading several hundreds of files and one needs to reestablish the connection.

As one runs the notebooks, the computed variables and intermediate results will be stored in the `results/variables/` directory. As the dataset and the obtained variables are too large, they are not included in the repository. However, we made available some of the data in here: (https://zenodo.org/record/7695389). We share some metadata we parsed from PubMed, including article title, journal, PMID, publication year, and abstract; as well as some produced by us, including the PubMedBERT embeddings of the abstracts, t-SNE embedding, labels, and colors.

**Update:** we updated the dataset by downloading the latest annual PubMed snapshot (2023 baseline) contaning papers from the years 2022-2023, originally not in the dataset we used in the paper. The interactive visualization has the latest data and we plan on updating it yearly using annual PubMed releases. The code for updating and analyzing the latest data can be found in the directory `scripts/2024-baseline`.


## Installation
This project depends on Python ($\geq$ 3.7) and R ($\geq$ 4.0.0). The project script can be installed via `pip install .` in the project root, i.e.:
```
git clone https://github.com/berenslab/pubmed-landscape
cd pubmed-landscape
pip install -e .
```


## Detailed description of scripts

Here there is a more detailed description on what you can find in the different notebooks in the `scripts/` folder.

Notebooks in `scripts/`:
- `01-02`: obtain and prepare the data.
- `03-05`: pipelines to transform the abstracts from raw text to 2D embedding, both BERT-based and TF-IDF ones.
- `06-08`: metrics calculations (kNN recall, kNN accuracy and isolatedness).
- `09`: analysis of Covid-19 (Section 2.1 from the paper).
- `10`: analysis of neurscience (Section 2.2 from the paper).
- `11`: analysis of machine learning (Section 2.3 from the paper).
- `12-14`: analysis of author's gender (Section 2.4 from the paper).
- `15`: analysis of retracted papers (Section 2.5 from the paper).
- `16`: whitening experiment (see Methods, section 4.3).
- `17`: Covid-19 ablation experiment (see supplementary figure S7).
- `18`: analysis of affiliation countries (see supplementary figure S8).

Notebooks and Python files in `scripts/BERT-based-embeddings/` (Table 3, section 4.3 of the paper):
- `01-rgm-ls-malteos.ipynb`: compute the SciNCL embeddings of the abstracts and the kNN accuracy of the representation.
- `02-rgm-ls-SBERT.ipynb`: compute the SBERT embeddings of the abstracts and the kNN accuracy of the representation.
- `03-rgm-ls-PubMedBERT.ipynb`: compute the PubMedBERT embeddings of the abstracts and the kNN accuracy of the representation.
- `04-rgm-ls-BERT.ipynb`: compute the BERT embeddings of the abstracts and the kNN accuracy of the representation.
- `05-rgm-pipeline-TFIDF-1M.ipynb`: compute the TF-IDF rerpesentation and its kNN accuracy of the corresponding 1M subset used for the BERT-based models comparison.
- `06-rgm-pipeline-BERT-models.ipynb`: compute the t-SNE embedding of the different BERT-based models representations.
- `bert-models.py`: compute the embeddings of the abstracts and the kNN accuracy for the rest of the BERT-based models.
- `07-rgm-comparison-tSNE-UMAP-1M.ipynb`: comparison t-SNE vs. UMAP in the 1M subset.
- `08-rgm-SVD-L2-experiment.ipynb`: effect of $L_2$ normalizing in SVD.
- `09-rgm-TF-IDF-vocabulary-experiment.ipynb`: experiment of combining PubMedBERT tokenizer and TF-IDF representation (see Section 4.4 from the paper).
