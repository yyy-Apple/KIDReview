<img src="./files/kid.jpeg" width="150" class="center"><img src="./files/font.png" width="500" class="left">

This is the repo for the paper [KID-REVIEW: Knowledge-GuIdeD Scientific Review Generation with Oracle Pre-training]()

# Requirements
Requirements for all the libraries are in the **requirements.txt**, please use the right version for each library in order to reproduce our results.

# Dataset
Run the command below to download our dataset.
```
sh prepare_data.sh
```


# Finetune
To finetune the baseline model using oracle extraction, use the following command. The default save directory is *trained_models*, the default save model name is *bart.pthbest*
```python
python finetune.py --source oracle --gpu 2
```
To finetune the citation graph model, use the following command. The default save directory is *trained_models*, the default save model name is *bart_cite.pthbest* 
```python
python finetune.py --source oracle --gpu 2 --citation_graph --prepend 
```
To finetune the concept graph model, use the following command. The default save directory is *trained_models*, the default save model name is *bart_concept.pthbest*
```python
python finetune.py --source oracle --gpu 2 --concept_graph
```
To finetune the model using both citation graph and concept graph, we first reload the model trained by using pure concept graph, and then finetune. Please use the following command. The default save directory is *trained_models*, the default save model name is *bart_cite_concept.pthbest*
```
python finetune.py --source oracle --citation_graph --prepend --concept_graph --reload_from_saved trained_models/bart_concept.pthbest
```
All arguments are in **args.py** file, please check it for more settings.

# Generate
Using vanilla model to generate text. The default output directory is *output*, the default predictions are in the created *pred.txt*.
```python
python generate.py --source oracle --gpu 1 --model_name bart.pthbest 
```
Using citataion graph to generate text.
```python
python generate.py --prepend --source oracle --gpu 1 --citation_graph --model_name bart_cite.pthbest
```
Using concept graph to generate text.
```python
python generate.py --source oracle --gpu 1 --concept_graph --model_name bart_concept.pthbest
```
Using both concept graph and citation graph to generate text.
```python
python generate.py --prepend --source oracle --gpu 1 --citation_graph --concept_graph --model_name bart_cite_concept.pthbest 
```


# Bib
```
@article{Yuan_Liu_2022, 
    title={KID-Review: Knowledge-Guided Scientific Review Generation with Oracle Pre-training}, 
    volume={36}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/21418}, DOI={10.1609/aaai.v36i10.21418}, 
    abstractNote={The surge in the number of scientific submissions has brought challenges to the work of peer review. In this paper, as a first step, we explore the possibility of designing an automated system, which is not meant to replace humans, but rather providing a first-pass draft for a machine-assisted human review process. Specifically, we present an end-to-end knowledge-guided review generation framework for scientific papers grounded in cognitive psychology research that a better understanding of text requires different types of knowledge. In practice, we found that this seemingly intuitive idea suffered from training difficulties. In order to solve this problem, we put forward an oracle pre-training strategy, which can not only make the Kid-Review better educated but also make the generated review cover more aspects. Experimentally, we perform a comprehensive evaluation (human and automatic) from different perspectives. Empirical results have shown the effectiveness of different types of knowledge as well as oracle pre-training. We make all code, relevant dataset available: https://github.com/Anonymous4nlp233/KIDReview as well as the Kid-Review system: http://nlpeer.reviews.}, 
    number={10}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Yuan, Weizhe and Liu, Pengfei}, 
    year={2022}, 
    month={Jun.}, 
    pages={11639-11647} 
}
```