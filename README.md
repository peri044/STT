# Show, Translate and Tell

This repo contains code for training and evaluation of a multi-task model which performs image captioning, cross modal retrieval and sentence paraphrasing. 
The paper and results can be found at <a href="https://arxiv.org/abs/1903.06275"> Show, Translate and Tell </a>
The proposed architecture is as shown in the figure
![Alt text](figures/stt.PNG?raw=true)

## Generate Data
In the data folder, you can find scripts for generating TF-records for mscoco dataset.
Checkout command line arguments in the scripts for setting paths
* To generate TF-records for MSCOCO
```
python -m data.coco_data_loader --num 10000
```
Args:
* `--num` : Number of images to be written in TF record. Do not specify this unless you wnat to generate a subset of entire dataset.

* Generate TF-records with Image, predicted caption and groundtruth caption
```

python -m data.coco_data_loader --precompute \
                                --record_path /shared/kgcoe-research/mil/cvs_cvpr18/para_att_pred.tfrecord \
                                --feature_path /shared/kgcoe-research/mil/peri/scan_data/data/coco_precomp/testall_ims.npy \
                                --captions_path /shared/kgcoe-research/mil/cvs_cvpr18/coco/
```

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
