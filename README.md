# TAKG
The official implementation of ACL 2019 paper "[**T**opic-**A**ware Neural **K**eyphrase **G**eneration for Social Media Language](https://arxiv.org/pdf/1906.03889.pdf)" (**TAKG**).
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
Here we give some representative commands illustrating how to preprocess data, train, test, and evaluate our model. For more detailed configurations, please refer to `config.py`. We also provide our model's sample prediction for the three datasets in `my_sample_prediction`

This code is mainly adapted from [KenChan's keyphrase generation code](https://github.com/kenchan0226/keyphrase-generation-rl) and [Zengjichuan's TMN](https://github.com/zengjichuan/TMN).


### Prepocessing
To preprocess the source data, run:
`python preprocess.py -data_dir data/Weibo`

It will output the processed data to the folder `processed_data`.

Some common arguments are listed below:
```
data_dir: The source file of the data
vocab_size: Size of the source vocabulary
bow_vocab: Size of the bow dictionary
max_src_len: Max length of the source sequence
max_trg_len: Max length of the target sequence
```

### Training
Only train the seq2seq model, run: `python train.py -data_tag Weibo_s100_t10`.

Only train the neural topic model, run: `python train.py -data_tag Weibo_s100_t10 -only_train_ntm -ntm_warm_up_epochs 100`.

Jointly train the TAKG model, run: `python train.py -data_tag Weibo_s100_t10  -use_topic_represent -load_pretrain_ntm  -joint_train  -topic_attn -check_pt_ntm_model_path [the warmed up ntm model path]`

There are some common arguments about different training strategies and model variants:
```
joint_train: whether to jointly train the model, True or False
joint_train_strategy: how to jointly train the model, choices=['p_1_iterate', 'p_0_iterate', 'p_1_joint', 'p_0_joint'], iterate: iteratively train each module, joint: jointly train both modules, the digit (0/1): the epoch number of pretraining the seq2seq 
topic_type: use latent variable z or g as topic
topic_dec: add topic in decoder input
topic_attn: add topic in context vector
topic_copy: add topic in copy switch
topic_attn_in: add topic in computing attn score
add_two_loss: use the sum of two losses as the objective 
```


### Inference
To generate the prediction, run: `python predict.py -model [seq2seq model path] (-ntm_model [ntm model path]`).

It will output the prediction under the folder `pred`.


### Evaluation
To evaluate the predictions, run: `python pred_evaluate.py -pred [prediction path] -src_file_path [data/Weibo/test_src.txt] -trg_file_path [data/Weibo/test_trg.txt]`


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
