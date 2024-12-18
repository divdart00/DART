# DART: Diversified and Accurate Long-Tail Recommendation.

This project is a pytorch implementation of 'DART: Diversified and Accurate Long-Tail Recommendation'.
This project provides executable source code with adjustable arguments and preprocessed datasets used in the paper.
The backbone model of DART is SASRec, and we built our implementation using the following repository: https://github.com/pmixer/SASRec.pytorch

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/) 1.13.1 

## Datasets
We use 3 datasets in our work: Books, MovieLens, and Yelp.
The preprocessed dataset is included in the repository: `./data`.

## Repository Architecture
There are 5 folders and each consists of:
- data: preprocessed datasets
- result: cluster files, trained models
- shell: script files
- src: source codes

## Running the code

### 1)  Only inference
We uploaded the pretrained model for books and ml to `./result/{datasetname}/model/DART/{dataset_name}_DART.pth.`

For specific hyperparameters, refer to `./result/{dataset_name}/model/DART/args.txt`. 

To perform only inference with this pretrained model, run the following code.

```
bash shell/test.sh
```

### 2)  Full training

1.  Before training DART, you need to train Neural Matrix Factorization first. Therefore, run the following code first.
    ```
    python src/nmf.py --dataset={dataset_name}
    ```

2.  You can train the model DART by following code. 
    The different hyperparameters for each dataset are set in `main.py`.
    ```
    python src/main.py --dataset={dataset_name}
    ```
    Running the above code will initiate clustering. 
    By default, the code runs based on the uploaded clusters.
    If you want to train the model with new clustering, set `--exist_cluster=no` when executing the code.

    If you want to train the model using an already uploaded cluster, set `--exist_cluster=yes` when executing the code.
    Alternatively, you can run the script `./shell/train.py`. 



