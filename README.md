# Show, Translate and Tell

This repo contains code for training and evaluation of a multi-task model which performs image captioning, cross modal retrieval and sentence paraphrasing.
The paper and results can be found at <a href="https://arxiv.org/abs/1903.06275"> Show, Translate and Tell</a>. This work has been accepted at <a href="http://2019.ieeeicip.org/index.php"> ICIP 2019</a>.
The proposed architecture is as shown in the figure
![Alt text](figures/stt.PNG?raw=true)

## Generate Data
In the data folder, you can find scripts for generating TF-records for mscoco dataset.
Update (11/20/2019): `prepare_mscoco_pairs.py` is updated in the repo. It uses the `captions_train2014.json` annotations of MSCOCO to build paraphrases. This script should be used to generate `train_enc.txt` and `train_dec.txt` which are basically paraphrases.
Using 5 captions, it creates 20 permutations of paraphrases and writes them in the TF record (using coco_data_loader.py) along with the associated image. This script is not cleaned up and should only be used for reference (it might not be the final script that we used).
Checkout command line arguments in the scripts for setting paths
* To generate TF-records for MSCOCO
```
python -m data.coco_data_loader --num 10000
```
Args:
* `--num` : Number of images to be written in TF record. Do not specify this unless you want to generate a subset of entire dataset.

* Generate TF-records with Image, predicted caption and groundtruth caption
```

python -m data.coco_data_loader --precompute \
                                --record_path para_att_pred.tfrecord \
                                --feature_path coco_precomp/testall_ims.npy \
                                --captions_path <path_to_coco_captions>
```

* `feature_path`: This should be used if you have already extracted features for all the images. 
In this case, a sample in TF record would look like (Feature (1x2048 dim vector for an image), caption A, caption B) which are paraphrases.

## Training on COCO dataset

```
sh scripts/train_coco.sh
```

## Evaluation on COCO dataset

```
sh scripts/eval_coco.sh
```

If you find this research or codebase useful in your experiments, consider citing

```
@article{stt2019,
  title={Show, Translate and Tell},
  author={Peri, Dheeraj and Sah, Shagan and Ptucha, Raymond},
  journal={arXiv preprint arXiv:1903.06275},
  year={2019}
}
```
