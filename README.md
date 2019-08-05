# TAKG
The official implementation of ACL 2019 paper "[**T**opic-**A**ware Neural **K**eyphrase **G**eneration for Social Media Language](https://www.aclweb.org/anthology/P19-1240)" (**TAKG**).
This is a joint work with [NLP Center at Tencent AI Lab](https://ai.tencent.com/ailab/nlp/). Some scripts for drawing figures in the paper can be found [here](https://github.com/yuewang-cuhk/DrawFigureForPaper).


## Dataset
Due to the copyright issue of TREC 2011 Twitter dataset, we only release the Weibo dataset (in `data/Weibo`) and StackExchange dataset (in `data/StackExchange`). For more details about the Twitter dataset, please contact [Yue Wang](https://yuewang-cuhk.github.io/) or [Jing Li](https://girlgunner.github.io/jingli/).

### Data format
* The dataset is randomly splited into three segments (80% training, 10% validation, 10% testing).
* For each segment (train/valid/test), we have the source posts (stored in `".*_src.txt"`) and target keyphrases (stored in `".*_trg.txt"`). One line is for each instance.
* For multiple keyphrases for one post, keyphrases are seperated by a semicolon `";"`.

### Data statistics
We first show the statistics of the source posts. (**KP** refers to keyphrase)

Datasets | # of posts | Avg len of posts | # of KP per post | Source Vocabulary
--- | --- | --- | --- | --- 
Twitter | 44,113 | 19.52 | 1.13 | 34,010
Weibo | 46,296 | 33.07 | 1.06 | 98,310
StackExchange | 49,447 | 87.94 | 2.43 | 99,775

We then show the detailed statistics of keyphrases:

Datasets | Size of KP | Avg len of KP | % of absent KP | Target Vocabulary
--- | --- | --- | --- | --- 
Twitter | 4,347 | 1.92 | 71.35 | 4,171
Weibo | 2,136 | 2.55 | 75.74 | 2,833
StackExchange | 12,114 | 1.41 | 54.32 | 10,852


## Model
Our model allows joint modeling of latent topics and keyphrase generation. It consists of a neural topic model to induce the latent topics and a neural seq2seq-based generation model to produce keyphrases. The overall architecture is depicted below:

<p align="center">
  <img src="https://github.com/yuewang-cuhk/TAKG/blob/master/model.PNG" alt="The overall architecture" width="600"/>
</p>

## Code
Here we give some representative commands illustrating how to preprocess data, train, test, and evaluate our model. For more detailed configurations, please refer to `config.py`. In `train.py` and `predict.py`, I hard code some default arguments in the function `process_opt(opt)` to simplify each running command. We also provide our model's sample predictions for the three datasets in `my_sample_prediction`.

This code is mainly adapted from [KenChan's keyphrase generation code](https://github.com/kenchan0226/keyphrase-generation-rl) and [Zengjichuan's TMN](https://github.com/zengjichuan/TMN). Thank my colleagues (Ken and Zeng) very much for their support.


### Dependencies
* Python 3.5+
* Pytorch 0.4


### Prepocessing
To preprocess the source data, run:
`python preprocess.py -data_dir data/Weibo`

It will output the processed data to the folder `processed_data`. The filename of the processed data will be used as the argument `data_tag` in the following process. 

Some common arguments are listed below:
```
data_dir: The source file of the data
vocab_size: Size of the source vocabulary
bow_vocab: Size of the bow dictionary
max_src_len: Max length of the source sequence
max_trg_len: Max length of the target sequence
```

### Training
Only train the seq2seq model, run: `python train.py -data_tag Weibo_s100_t10`. To train with copy mechanism, add `-copy_attention`.

Only train the neural topic model, run: `python train.py -data_tag Weibo_s100_t10 -only_train_ntm -ntm_warm_up_epochs [epoch_num, e.g., 100]`.

Jointly train the TAKG model, run: `python train.py -data_tag Weibo_s100_t10  -copy_attention -use_topic_represent -load_pretrain_ntm  -joint_train  -topic_attn -check_pt_ntm_model_path [the warmed up ntm model path]`.

There are some common arguments about different joint training strategies and model variants:
```
joint_train: whether to jointly train the model, True or False
joint_train_strategy: how to jointly train the model, choices=['p_1_iterate', 'p_0_iterate', 'p_1_joint', 'p_0_joint'], iterate: iteratively train each module, joint: jointly train both modules, the digit (0/1): the epoch number of pretraining the seq2seq 
topic_type: use latent variable z or g as topic, g is the default
topic_dec: add topic in decoder input
topic_attn: add topic in context vector
topic_copy: add topic in copy switch
topic_attn_in: add topic in computing attn score
add_two_loss: use the sum of two losses as the objective 
```
Different ways of adding topic can be combined simultaneously. Let's say: `-topic_attn -topic_dec -topic_copy`.

### Inference
To generate the prediction, run: `python predict.py -model [seq2seq model path] (-ntm_model [ntm model path]).`

Some common arguments are listed below: 
```
batch_size: batch size during predicting. Decrease this number if it exceeds GPU memory.
beam_size: beam size. Decrease this number if it exceeds GPU memory.
n_best: default -1, pick the top n_best sequences from beam_search, if n_best < 0, then n_best=beam_size
max_length: max sequence length for each prediction.
```

It will output the prediction under the folder `pred`. For the format of prediction, please refer to `my_sample_prediction`.


### Evaluation
To evaluate the predictions, run: `python pred_evaluate.py -pred [prediction path] -src [data/Weibo/test_src.txt] -trg [data/Weibo/test_trg.txt]`

It will output the performance in different metrics including _Precision_, _Recall_, _F1 measure_, _MAP_, _NDCG_ at different top predictions. You can modify the top prediction number in the main function of `pred_evaluate.py`. It will also report results for present keyphrase, absent keyphrase, and both of them.


## Citation
If you use either the code or data in your paper, please kindly star this repo and cite our paper:
```
@inproceedings{wang-etal-2019-topic-aware,
    title = "Topic-Aware Neural Keyphrase Generation for Social Media Language",
    author = "Wang, Yue  and
      Li, Jing  and
      Chan, Hou Pong  and
      King, Irwin  and
      Lyu, Michael R.  and
      Shi, Shuming",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1240",
    pages = "2516--2526",
    abstract = "A huge volume of user-generated content is daily produced on social media. To facilitate automatic language understanding, we study keyphrase prediction, distilling salient information from massive posts. While most existing methods extract words from source posts to form keyphrases, we propose a sequence-to-sequence (seq2seq) based neural keyphrase generation framework, enabling absent keyphrases to be created. Moreover, our model, being topic-aware, allows joint modeling of corpus-level latent topic representations, which helps alleviate data sparsity widely exhibited in social media language. Experiments on three datasets collected from English and Chinese social media platforms show that our model significantly outperforms both extraction and generation models without exploiting latent topics. Further discussions show that our model learns meaningful topics, which interprets its superiority in social media keyphrase generation.",
}
```
