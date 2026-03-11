# Exploring Gradual and Abrupt Preference Transition Processes for Sequential Recommendation

This is the implementation of the submission "Exploring Gradual and Abrupt Preference Transition Processes for Sequential Recommendation".

## Environment

The hardware and software we used are listed below to facilitate the environment's configuration.

- Hardware:
  - GPU: GeForce RTX 4090 GPU
  - CUDA: 11.8

You can run the following command to a reproduce environment:

```shell
conda create -y -n HPT python=3.9
conda activate HPT
pip install -r HPT/requirements.txt
```

## Recommendation Dataset

Dataset Download Options:

- You can directly download the dataset using the links provided by RecBole here: https://github.com/RUCAIBox/RecBole/blob/master/recbole/properties/dataset/url.yaml。

- Alternatively, you can modify the `dataset` field in `model/config.yaml` to one of the following dataset names: `amazon-baby`, `amazon-beauty`, `amazon-sports-outdoors`,  `amazon-video-games-18`, `ml-100k` or `lastf,`. The dataset will be downloaded automatically when running the training script.

  ```yaml
  dataset: {datset}
  ```

## Granular ball Construction

### Pretraining

We use SASRec as the backbone model. You can follow the steps below to obtain user and item embeddings:

1. Generate the ID mapping for user and item tokens:

   ```shell
   python token2id.py --dataset {dataset}
   ```

2. Prepare the dataset in the format required by SASRec:

   - Run the remapping script to generate `inter_remapped.txt`:

     ```shell
     python dataset/remapping.py --dataset {dataset}
     ```

   - Run the extraction script to generate the SASRec-compatible dataset file `inter.txt`:

     ```shell
     python dataset/extract.py --dataset {dataset}
     ```

3. Train the SASRec model to obtain user and item embeddings:
    Outputs will be `itm_emb_sasrec_seq.pkl` and `usr_emb_sasrec_seq.pkl`.

   - ```shell
     bash experiments/general.bash {dataset}
     ```

### Granular ball

Then, run the following command to generate the corresponding granular ball representation:

```shell
python dataset/granular_ball.py --dataset {dataset} --p {p}
```

## Training

Finally, you can run follow script for dataset:

```shell
python run.py --model HPT --dataset amazon-baby --exp_type Overall --alpha 0.1 --beta 0.1 --zeta 0.5 --rho 0.3 --L 3 --p 50
```


