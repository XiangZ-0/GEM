Please put the datasets here. 
## Data Format
Events and images are stored as npz files in our work, where each npz file contains two consecutive blurry images (to facilitate synthesizing large blur) 
and the corresponding sharp ground-truth images (note, MS-RBD do not contain sharp images) as well as events. 
Specifically, each npz file contains the following keys:
- **blur1**: the first blurry image in (C, H, W) format.
- **exp_start1**: the exposure start timestamp of the first blurry image in integer format.
- **exp_end1**: the exposure end timestamp of the first blurry image in integer format.
- **blur2**: the second blurry image in (C, H, W) format.
- **exp_start2**: the exposure start timestamp of the second blurry image in integer format.
- **exp_end2**: the exposure end timestamp of the second blurry image in integer format.
- **sharp_imgs**: sharp ground-truth images of the two blurry frames in (N, C, H, W) format, where N is the total number of ground-truth images.
- **sharp_timestamps**: the timestamps of each sharp ground-truth images in (N) format. 
- **events**: a dictionary containing the spatial coordinates ('x', 'y'), timestamps ('t'), and polarities ('p') of the events triggered between exp_start1 and exp_end2. 
- **scale_factor**: the spatial resolution ratio of images over events, e.g., scale_factor=4 means the spatial resolution of images is 4 times that of events.
