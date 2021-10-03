# active-learning-transformer

A hands-on tutorial on how to use Active Learning with Transformer models.

This repository contains the code for this article: TBD



# Usage

You can either use the Google colab notbook or run it on you own infra as follow


Install the requirement
```
pip intall -r requirements.txt
```

To run the experiment, use `python main.py`. See the article above for details.

```
python main.py -h
usage: main.py [-h] [--do_al] [--target_score TARGET_SCORE] [--task_name TASK_NAME] [--random_seed RANDOM_SEED] [--initial_train_dataset_size INITIAL_TRAIN_DATASET_SIZE]
               [--query_samples_count QUERY_SAMPLES_COUNT]

optional arguments:
  -h, --help            show this help message and exit
  --do_al
  --target_score TARGET_SCORE
  --task_name TASK_NAME
  --random_seed RANDOM_SEED
  --initial_train_dataset_size INITIAL_TRAIN_DATASET_SIZE
  --query_samples_count QUERY_SAMPLES_COUNT
 ```
 
 Example, the followin will run the active learning experiment ont mrpc dataset
 
 ```
 pyton main.py --do_al --taget_score 85.7 --task_name mrpc --random_seed 123 --initial_train_dataset_size 0.3 --query_samples_count 64
 ```
