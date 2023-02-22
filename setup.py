from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="pubmed_landscape_src",
    description="Python functions used for the analysis of the PubMed landscape",
    author=["Rita González-Márquez", "Lucha Schmidt", "Dmitry Kobak", "Philipp Berens"],
    author_email="rita.gonzalez-marquez@uni-tuebingen.de",
    install_requires=required,
    packages=find_packages(),
)