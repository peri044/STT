# Show, Translate and Tell

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
mv captioning/new_captions/ic_coco_stt_para-att_12001.txt /shared/kgcoe-research/mil/cvs_cvpr18/coco/

python -m data.coco_data_loader --precompute \
                                --record_path /shared/kgcoe-research/mil/cvs_cvpr18/para_att_pred.tfrecord \
                                --feature_path /shared/kgcoe-research/mil/peri/scan_data/data/coco_precomp/testall_ims.npy \
                                --captions_path /shared/kgcoe-research/mil/cvs_cvpr18/coco/
```
