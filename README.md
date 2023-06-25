# GEM - Generalizing Event-Based Motion Deblurring in Real-World Scenarios 
## Paper | Supp | Video

Event-based motion deblurring has shown promising results by exploiting low-latency events. However, current approaches are limited in their practical usage, as they assume the same spatial resolution of inputs and specific blurriness distributions. This work addresses these limitations and aims to generalize the performance of event-based deblurring in real-world scenarios. We propose a scale-aware network that allows flexible input spatial scales and enables learning from different temporal scales of motion blur. A two-stage self-supervised learning scheme is then developed to fit real-world data distribution. By utilizing the relativity of blurriness, our approach efficiently ensures the restored brightness and structure of latent images and further generalizes deblurring performance to handle varying spatial and temporal scales of motion blur in a self-distillation manner. Our method is extensively evaluated, demonstrating remarkable performance, and we also introduce a real-world dataset consisting of multi-scale blurry frames and events to facilitate research in event-based deblurring.


## Environment setup

- Python 3.7
- Pytorch-Lightning 1.6.0
- NVIDIA GPU + CUDA
- numpy, argparse, opencv-python

You can create a new [Anaconda](https://www.anaconda.com/products/individual) environment as follows.
<br>
```
conda create -n gem python=3.7
conda activate gem
```
Clone this repository.
```
git clone git@github.com:XiangZ-0/GEM.git
```
Install the above dependencies and [Deformable Convolution V2](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0)
```
cd GEM
pip install -r requirements.txt
cd codes/model/DCN_v2/
sh make.sh
```

## Download model and data
[Pretrained models](https://1drv.ms/f/s!AgjOZB4WHoLei2Kz5horpPI6A1aP?e=dNZypa) and [datasets](https://1drv.ms/f/s!AgjOZB4WHoLei2PMmIQza0cPaWZA?e=acFSD4) can be downloaded via One Drive.
<br>
In our paper, we conduct experiments on three types of data:
- **Ev-REDS** contains synthetic 1280x640 blurry images and synthetic 320x160 events. We first convert [REDS](https://seungjunnah.github.io/Datasets/reds.html) into high frame rate videos using [RIFE](https://github.com/hzwer/arXiv2021-RIFE), and then obtain blurry images by averaging sharp frames and generate events from down-sampled images via [VID2E](https://github.com/uzh-rpg/rpg_vid2e).
- **HS-ERGB** contains synthetic blurry images and real-world events from [HS-ERGB](https://github.com/uzh-rpg/rpg_timelens). We first convert HS-ERGB into high frame rate videos using [Time Lens](https://github.com/uzh-rpg/rpg_timelens) and then synthesize blurry images by averaging sharp frames. Since only the test set of HS-ERGB is available, we choose 4 sequences (*far-bridge_lake_01*, *close-fountain_schaffhauserplatz_02*, *close-spinning_umbrella*, and *close-water_bomb_floor_01*) for testing and leave the rest for training. We mannually filter the static frames in the HS-ERGB dataset (where no motion blur occurs) to ensure valid evaluation of deblurring performance.
- **MS-RBD** contains real-world blurry images and real-world events collected by ourselves. A beam splitter connecting a FLIR BlackFly S RGB camera and a DAVIS346 event camera is built for data collection. In total, our MS-RBD contains 32 sequences composed of 22 indoor and 10 outdoor scenes, where each sequence consists of 60 RGB 1152x768 blurry frames and the concurrent 288x192 events. For self-supervised methods, we select 5 and 3 sequences from the indoor and outdoor scenes for testing and leave the rest for training. For supervised approaches, all sequences can be used for qualitative evaluation of deblurring performance in real-world scenarios.

<div align=center> <img src="figs/camera_setup.jpg" height="200"> </div>
<div align=center> MS-RBD capture system </div>

<div align=center> <img src="figs/dataset_detail.png" width="600"> </div>
<div align=center> Overview of MS-RBD </div>

<div align=center> <img src="figs/dataset_example.png" width="600"> </div>
<div align=center> Examples of MS-RBD </div>

## Quick start
### Initialization
- Change the parent directory to './codes/'
```
cd codes
```
- Download pretrained model to directory './checkpoint/'
- Download the datasets to directory './datasets/'

TODO =============================

### Test
- Test on GoPro data
```
python Test.py --test_ts=0.5 --model_path=./PreTrained/EVDI-GoPro.pth --test_path=./Database/GoPro/ --save_path=./Result/EVDI-GoPro/ 
```
- Test on HQF data
```
python Test.py --test_ts=0.5 --model_path=./PreTrained/EVDI-HQF.pth --test_path=./Database/HQF/ --save_path=./Result/EVDI-HQF/ 
```
- Test on RBE data
```
python Test.py --test_ts=0.5 --model_path=./PreTrained/EVDI-RBE.pth --test_path=./Database/RBE/ --save_path=./Result/EVDI-RBE/
```
- Test on GoPro-Color data
```
python Test.py --test_ts=0.5 --model_path=./PreTrained/EVDI-GoPro-Color.pth --test_path=./Database/GoPro-Color/ --save_path=./Result/EVDI-GoPro-Color/ --color_flag=1
```
**Main Parameters:**
- `--test_ts` : reconstruction timestamp, normalized in \[0,1\].
- `--model_path` : path of pretrained model.
- `--test_path` : path of test dataset.
- `--save_path` : path of reconstruction results.
- `--color_flag` : use color model or gray model.

### Train
If you want to train your own model, please prepare the blurry images and events in the following directory structure (an example data is provided in './Database/Raw/' for reference):
```
<project root>
  |-- Database
  |     |-- Raw
  |     |     |-- Events.txt
  |     |     |-- Exposure_start.txt
  |     |     |-- Exposure_end.txt
  |     |     |-- Blur
  |     |     |     |-- 000000.png
  |     |     |     |-- 000001.png
  |     |     |     |-- ...
```
- Events.txt contains event data in (t,x,y,p) format with t in ns and p in {-1, 1}.
- Exposure_start.txt contains the start timestamp of each blurry image in ns.
- Exposure_end.txt contains the end timestamp of each blurry image in ns.

After arranging the raw data into the above structure, please pack them into training pairs by running 
```
python Prepare_data.py --input_path=./Database/Raw/ --save_path=./Database/train/ --color_flag=0
```
Please set --color_flag=1 if you want to use color images. Finally, modify the parameters in 'Train.py' according to your need and run
```
python Train.py
```
**Main Parameters:**
- `--model_path` : model save path.
- `--train_path` : path of train datasets.
- `--num_epoch` : number of epoch.
- `--loss_wei` : weights for loss functions \[blur-sharp, blur-event, sharp-event\].
- `--num_frames` : the number of reconstructions per input, related to 'M' in paper (recommended >= 25).
- `--bs` : batch size.
- `--lr` : initial learning rate.
- `--color_flag` : use color or gray model.

## Citation
If you find our work useful in your research, please cite:

```
@inproceedings{zhang2022generalizingc,
  title={Generalizing Event-Based Motion Deblurring in Real-World Scenarios},
  author={Zhang, Xiang and Yu, Lei and Yang, Wen and Liu, Jianzhuang and Xia, Gui-Song},
  year={2023},
  booktitle={ICCV},
}
```

