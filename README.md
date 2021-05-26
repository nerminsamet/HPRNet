# HPRNet: Hierarchical Point Regression for Whole-Body Human Pose Estimation
Official PyTroch implementation of HPRNet.

> [**HPRNet: Hierarchical Point Regression for Whole-Body Human Pose Estimation**](https://arxiv.org/abs/2104.06773),            
> [Nermin Samet](https://nerminsamet.github.io/), [Emre Akbas](http://user.ceng.metu.edu.tr/~emre/),        
> *Under review at IMAVIS. ([arXiv pre-print](https://arxiv.org/abs/2104.06773))*          

  
## Highlights
- HPRNet is a bottom-up, one-stage and hierarchical keypoint regression method for whole-body pose estimation.
- HPRNet has the best performance among bottom-up methods for all the whole-body parts. 
- HPRNet achieves SOTA performance for the face (*76.0* AP) and hand (*51.2* AP) keypoint estimation.
- Unlike two-stage methods, HPRNet predicts whole-body pose in a constant time independent of the number of people in an image.


## COCO-WholeBody Keypoint Estimation Results

| Model                    |   Body AP        | Foot AP        | Face AP  |  Hand AP     |  Whole-body AP       | Download |
|--------------------------|--------------------|-----------|-----------|-----------|-----------|-----------|
|[HPRNet (DLA)](../experiments/wholebody_hprnet_dla.sh)   | 55.2 /  57.1 | 49.1 / 50.7 | 74.6 / 75.4 | 47.0 / 48.4 |  31.5 / 32.7|[model](https://drive.google.com/file/d/1LQShniDCkTNJDfvyfbU8QXMy_Uqz-_C2/view?usp=sharing) |
|[HPRNet (Hourglass)](../experiments/wholebody_hprnet_hourglass.sh) |59.4 / **61.1** | 53.0 / **53.9** | 75.4 / **76.0** | 50.4 / **51.2** | 34.8 / **34.9** | [model](https://drive.google.com/file/d/1qcE7ac_I_M4qvXV2TH2KO8314K3Q7zIV/view?usp=sharing) |

- Results are presented without and with test time flip augmentation respectively.
- All models are trained on COCO-WholeBody `train2017` and evaluated on `val2017`.
- The models can be downloaded directly from [Google drive](https://drive.google.com/drive/u/1/folders/1yKxQVRxjicvDDM_p1-uKdAewRaSIaY4P).


## Installation


0. [Optional but recommended] create a new conda environment.

    ~~~
    conda create --name HPRNet python=3.7
    ~~~
    And activate the environment.

    ~~~
    conda activate HPRNet
    ~~~

1. Clone the repo:

    ~~~
    HPRNet_ROOT=/path/to/clone/HPRNet
    git clone https://github.com/nerminsamet/HPRNet $HPRNet_ROOT
    ~~~

2. Install PyTorch 1.4.0:

    ~~~
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    ~~~

3. Install the requirements:

    ~~~
    pip install -r requirements.txt
    ~~~


5. Compile DCNv2 (Deformable Convolutional Networks):

    ~~~
    cd $HPRNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
    ~~~

## Dataset preparation

- Download the images (2017 Train, 2017 Val) from [coco website](http://cocodataset.org/#download).
- Download [train](https://drive.google.com/file/d/1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf/view) and [val](https://drive.google.com/file/d/1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb/view) annotation files.
  
  ~~~
  ${COCO_PATH}
  |-- annotations
      |-- coco_wholebody_train_v1.0.json
      |-- coco_wholebody_val_v1.0.json
  |-- images
      |-- train2017
      |-- val2017 
  ~~~
  

## Evaluation and Training


- You could find all the evaluation and training scripts in the [experiments](../experiments) folder.
- For evaluation, please download the [pretrained models](https://drive.google.com/drive/folders/1yKxQVRxjicvDDM_p1-uKdAewRaSIaY4P?usp=sharing) you want to evaluate and put them in `HPRNet_ROOT/models/`.
- In the case that you don't have 4 GPUs, you can follow the [linear learning rate rule](https://arxiv.org/abs/1706.02677) to adjust the learning rate.
- If the training is terminated before finishing, you can use the same command with `--resume` to resume training. 


## Acknowledgement

The numerical calculations reported in this paper were fully performed at TUBITAK ULAKBIM,  High Performance and Grid Computing Center (TRUBA resources). 
 
## License

HPRNet is released under the MIT License (refer to the [LICENSE](readme/LICENSE) file for details). 

## Citation

If you find HPRNet useful for your research, please cite our paper as follows:

> N. Samet, E. Akbas, "HPRNet: Hierarchical Point Regression for Whole-Body Human Pose Estimation",
> arXiv, 2021.

BibTeX entry:
 
```
@misc{hprnet,
      title={HPRNet: Hierarchical Point Regression for Whole-Body Human Pose Estimation}, 
      author={Nermin Samet and Emre Akbas},
      year={2021}, 
}
```
