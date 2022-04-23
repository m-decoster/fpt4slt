# Frozen Pretrained Transformers for Neural Sign Language Translation

This is the implementation for the paper [Frozen Pretrained Transformers for Neural Sign Language Translation](https://users.ugent.be/~mcdcoste/assets/SLT_DeCoster2021Frozen.pdf) presented at [AT4SSL 2021](https://sites.google.com/tilburguniversity.edu/at4svl2021/home), and the implementation of the article [Leveraging Frozen Pretrained Written Language Models for Neural Sign Language Translation ](https://www.mdpi.com/2078-2489/13/5/220) published in [Information's special issue "Frontiers in Machine Translation"](https://www.mdpi.com/journal/information/special_issues/frontiers_machine_translation).

This code is based on the implementation of [Sign Language Transformers: Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf), available [here](https://github.com/neccam/slt).

 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

Choose the configuration file that you want to reproduce and update the `data.data_path` and `training.model_dir` configuration entries
to the path where your data resides (default: `data/PHOENIX2014T`) and the path where you want the experiment logs and checkpoints to be saved.

  `python -m signjoey train configs/$CONFIG.yaml` 

For the mBART-50 model, you will first need to tokenize the corpus using the mBART-50 tokenizer. You can use the `tokenization/tokenize_mbart50.py` script for this.

## Citation

If you use this code in one of your projects, please cite

```
@InProceedings{De_Coster_2021_AT4SSL,
    author    = {De Coster, Mathieu and D'Oosterlinck, Karel and Pizurica, Marija and Rabaey, Paloma and Verlinden, Severine and Van Herreweghe, Mieke and Dambre, Joni},
    title     = {Frozen Pretrained Transformers for Neural Sign Language Translation},
    booktitle = {1st International Workshop on Automated Translation for Signed and Spoken Languages},
    month     = {August},
    year      = {2021},
}
```

and/or

```
@article{info13050220,
	Article-Number = {220},
	Author = {De Coster, Mathieu and Dambre, Joni},
	Doi = {10.3390/info13050220},
	Issn = {2078-2489},
	Journal = {Information},
	Number = {5},
	Title = {Leveraging Frozen Pretrained Written Language Models for Neural Sign Language Translation},
	Url = {https://www.mdpi.com/2078-2489/13/5/220},
	Volume = {13},
	Year = {2022}
}
```
