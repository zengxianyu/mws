# Code for the paper 
Code for the paper in CVPR2019, 'Multi-source weak supervision for saliency detection' ([download the pdf file](https://arxiv.org/pdf/1904.00566.pdf))

## Results

   score/datasets  |ECSSD | HKU-IS|PASCALS|SOD    |OMRON|DUTS-test| SED1|SED2|
  ---| ---  | ---   | ---   | ---   | --- | --- | ---|---|
  max$F_\beta$|.878|.856|.790|.799|.718|.767|.902|.849|
  MAE|.096|.084|.134|.167|.114|.096|.081|.097|
  
Download result maps: [OneDrive](https://1drv.ms/u/s!AqVkBGUQ01XGhx_eNt8MfQ_HpCO0) / [GoogleDrive](https://drive.google.com/file/d/1PqsH6HI_juyE6ZeDFhJ7ByscSEakwRS0/view?usp=sharing)

## Usage
### Test

0. Environment: python2.7, pytorch'1.0'

1. [Download models](https://1drv.ms/f/s!AqVkBGUQ01XGhyeDYsaaxvAZXW7k) and put in the current folder



1. Run

```bash
python main.py \
--img_dir 'path/to/images(.jpg)' \
--gt_dir 'path/to/ground-truth(.png)'

```

### Train
Please checkout the other branch of this repo

## Citation
```
@inproceedings{zeng2019multi,
  title={Multi-source weak supervision for saliency detection},
  author={Zeng, Yu and Zhuge, Yunzhi and Lu, Huchuan and Zhang, Lihe and Qian, Mingyang and Yu, Yizhou},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```




