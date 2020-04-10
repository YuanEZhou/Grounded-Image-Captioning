# Moer Grounded Image Captioning by Distilling  Image-Text Matching Model

## Requirements
- Python 3.7
- Pytorch 1.2

## Prepare data
1. Please use **git clone --recurse-submodules** to clone this repository and remember to follow initialization steps in coco-caption/README.md.
Then download and place the [Flickr30k reference file](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ERf4vteh7AdMmpR5jCc2ve4BNmZJ8EfY8LJVe4D3KCR4oQ?e=8qNj1W) under coco-caption/annotations. Also, download [Stanford CoreNLP 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html) for grounding evaluation and place the uncompressed folder under the tools/ directory.
2. Download the preprocessd dataset from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/Ea0HzFuNDGNPmmTxBTVjfbwBp9ZGhIAyyQylATXV735eUA?e=yEEaI6) and extract it to data/.
3. For *Flickr30k-Entities*, please download bottom-up visual feature extracted by Anderson's [extractor](https://github.com/peteanderson80/bottom-up-attention) (Zhou's [extractor](https://github.com/facebookresearch/grounded-video-description/tree/flickr_branch)) from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/EWKJu8TLXtVPu5h3EnNRWo4BfWs_3WIBfoXXJPWFoIS5kA?e=IFSR8Q) ( [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/ES446ZSwHCZAqiPjXxXW2twB_jMa_GmAiyuOUnEsNSWeUw?e=6u3pnF)) and place the uncompressed folders  under data/flickrbu/.   For *MSCOCO*, please follow this [instruction](https://github.com/ruotianluo/self-critical.pytorch/blob/master/data/README.md#convert-from-peteanderson80s-original-file) to prepare the bottom-up features and place them under data/mscoco/.
4. Download the pretrained models from [here](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/EXnM0hYNJXVEnkoPdCKEQ3cBnBqaNJf2cjkwJ4zJFm3MWg?e=NJU2Dj) and extract them to log/.
5. Download the pretrained SCAN models from this [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1806230d_e_ntu_edu_sg/EXS3v6NaZ8hPkCxebds8QIUBMzdokCQiNu5A4-N_yudvmg?e=o3wAzH) and extract them to misc/SCAN/runs.

## Evaluation
To reproduce the results reported in the paper, just simply run

```
bash eval_flickr.sh
```
fro Flickr30k-Entities and
```
bash eval_coco.sh
```
for MSCOCO.
## Training
1.  In the first training stage, run like
```
python train.py --id CE-scan-sup-0.1kl --caption_model topdown --input_json data/flickrtalk.json --input_fc_dir data/flickrbu/flickrbu_fc --input_att_dir data/flickrbu/flickrbu_att  --input_box_dir data/flickrbu/flickrbu_box  --input_label_h5 data/flickrtalk_label.h5 --batch_size 29 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log/CE-scan-sup-0.1kl --save_checkpoint_every 1000 --val_images_use -1 --max_epochs 30  --att_supervise  True   --att_supervise_weight 0.1
```

2. In the second training stage, run like

```
python train.py --id sc-ground-CE-scan-sup-0.1kl --caption_model topdown --input_json data/flickrtalk.json --input_fc_dir data/flickrbu/flickrbu_fc --input_att_dir data/flickrbu/flickrbu_att  --input_box_dir data/flickrbu/flickrbu_box  --input_label_h5 data/flickrtalk_label.h5 --batch_size 29 --learning_rate 5e-5 --start_from log/CE-scan-sup-0.1kl --checkpoint_path log/sc-ground-CE-scan-sup-0.1kl --save_checkpoint_every 1000 --language_eval 1 --val_images_use -1 --self_critical_after 30  --max_epochs  110      --cider_reward_weight  1
--ground_reward_weight   1 
```

## Citation

```
@inproceedings{zhou2020grounded,
  title={More Grounded Image Captioning by Distilling Image-Text Matching Model},
  author={Zhou, Yuanen and Wang, Meng and Liu, Daqing and  Hu, Zhenzhen and Zhang, Hanwang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Acknowledgements
This repository is built upon [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch),   [SCAN](https://github.com/kuanghuei/SCAN) and [grounded-video-description](https://github.com/facebookresearch/grounded-video-description/tree/flickr_branch). Thanks for their released  code.

