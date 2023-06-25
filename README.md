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
- **Ev-REDS** (TBD) contains synthetic blurry images and synthetic events. We first convert [REDS](https://seungjunnah.github.io/Datasets/reds.html) into high frame rate videos using [RIFE](https://github.com/hzwer/arXiv2021-RIFE), and then obtain blurry images by averaging sharp frames and generate events by [ESIM](https://github.com/uzh-rpg/rpg_vid2e).
- **HS-ERGB** (TBD) contains synthetic blurry images and real-world events from [HQF](https://timostoff.github.io/20ecnn), where blurry images are generated using the same manner as GoPro.
- **MS-RBD** (TBD) contains real-world blurry images and real-world events from [RBE](https://github.com/xufangchn/Motion-Deblurring-with-Real-Events).


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
@inproceedings{zhang2022unifying,
  title={Unifying Motion Deblurring and Frame Interpolation with Events},
  author={Zhang, Xiang and Yu, Lei},
  year={2022},
  booktitle={CVPR},
}
```

