# TAKG
The official implementation of ACL 2019 paper ["**T**opic-**A**ware Neural **K**eyphrase **G**eneration for Social Media Language" (**TAKG**)](https://arxiv.org/pdf/1906.03889.pdf).
This is a joint work with [NLP Center at Tencent AI Lab](https://ai.tencent.com/ailab/nlp/).

## Data
Due to the copyright issue of TREC 2011 Twitter dataset, we only release the Weibo dataset (in `data/Weibo`) and StackExchange dataset (in `data/StackExchange`). For more details about the Twitter dataset, please contact [Yue Wang](yuewang-cuhk.github.io) or [Jing Li](https://girlgunner.github.io/jingli/).

### Data format
* The dataset is randomly splited into three segments (80% training, 10% validation, 10% testing).
* For each segment (train/valid/test), we have the source posts (stored in `".*_src.txt"`) and target keyphrases (stored in `".*_trg.txt"`). One line is for each instance.
* For multiple keyphrases for one post, keyphrases are seperated by a semicolon `";"`.


## Model
Our model allows joint modeling of latent topics and keyphrase generation. It consists of a neural topic model to induce the latent topics and a neural seq2seq-based generation model to produce keyphrases. The overall architecture is depicted below:
![alt text](https://github.com/yuewang-cuhk/TAKG/blob/master/model.PNG "The overall architecture")

## Code
This code is mainly adapted from [KenChan's keyphrase generation code](https://github.com/kenchan0226/keyphrase-generation-rl) and [Zengjichuan's TMN](https://github.com/zengjichuan/TMN).

### Prepocessing
To preprocess the source data, run:
`python preprocess.py -data_dir ../data/Weibo`

Some common arguments are listed below:
```
data_dir: The source file of the data
vocab_size: Size of the source vocabulary
bow_vocab: Size of the bow dictionary
max_src_len: Max length of the source sequence
max_trg_len: Max length of the target sequence
```

### Run Model
TBA

## Citation
If you use either the code or data in your paper, please kindly star this repo and cite our paper:
```
@inproceedings{conf/acl/yuewang19,
  author    = {Yue Wang and
               Jing Li and
               Hou Pong Chan and
               Irwin King and
               Michael R. Lyu and                              
               Shuming Shi},
  title     = {Topic-Aware Neural Keyphrase Generation for Social Media Language},
  booktitle = {Proceedings of ACL},
  year      = {2019}
}
```
