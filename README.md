# StarGAN2 for practice

<p align='center'><img src='_out/blink-35.jpg' /></p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eps696/stargan2/blob/master/StarGAN2_colab.ipynb)

This version of [StarGAN2] (coined as 'Post-modern Style Transfer') is intended mostly for fellow artists, who rarely look at scientific metrics, but rather need a working creative tool. At least, this is what I use nearly daily myself.  
Here are few pieces, made with it: [Terminal Blink](http://www.aiartonline.com/highlights-2020/vadim-epstein), [Ghosts](https://vimeo.com/633172534), [Occurro](https://vimeo.com/527118906), [etc.](https://vimeo.com/445930853)  
Tested on Pytorch 1.4-1.8. Sequence-to-video conversions require [FFMPEG]. For more explicit details refer to the original implementation. 

## Features
* streamlined workflow, focused on practical tasks
* cleaned up and simplified code for better readability
* stricter memory management to fit bigger batches on consumer GPUs
* models mixing (SWA) for better stability

## Presumed file structure

| stargan2 | root
| :--- | :----------
| &boxvr;&nbsp; **_in** | input data for processing
| &boxvr;&nbsp; **_out** | generation output (sequences & videos)
| &boxvr;&nbsp; **data** | datasets for training
| &boxv;&nbsp; &boxur;&nbsp; afhq | [example] some dataset
| &boxv;&nbsp; &nbsp;&nbsp; &boxvr;&nbsp; cats | [example] images for training
| &boxv;&nbsp; &nbsp;&nbsp; &boxv;&nbsp; &boxur;&nbsp; test | [example] images for validation
| &boxv;&nbsp; &nbsp;&nbsp; &boxvr;&nbsp; dogs | [example] images for training
| &boxv;&nbsp; &nbsp;&nbsp; &boxv;&nbsp; &boxur;&nbsp; test | [example] images for validation
| &boxv;&nbsp; &nbsp;&nbsp; &boxur;&nbsp; &#x22ef; | 
| &boxvr;&nbsp; **models** | trained models for inference/processing
| &boxv;&nbsp; &boxur;&nbsp;  afhq-256-5-100.pkl | [example] trained model file
| &boxvr;&nbsp; **src** | source code
| &boxur;&nbsp; **train** | training folders
| &ensp;&ensp; &boxur;&nbsp;  afhq.. | [example] auto-created training folder


## Training

* Prepare your multi-domain dataset as shown above. Main directory should contain folders with images of different domains (e.g. cats, dogs, ..); every such folder must contain `test` subfolder with validation subset. Such structure allows easy data recombination for experiments. The images may be of any sizes (they'll be randomly cropped during training), but not smaller than `img_size` specified for training (default is `256`).  

* Train StarGAN2 on the prepared dataset (e.g. `afhq`):
```
 python src/train.py --data_dir data/afhq --model_dir train/afhq --img_size 256 --batch 8
```
This will run training process, according to the settings in `src/train.py` (check and explore those!). Models are saved under `train/afhq` and named as `dataset-size-domaincount-kimgs`, e.g. `afhq-256-5-100.ckpt` (required for resuming). 

* Resume training on the same dataset from the iteration 50 (thousands), presuming there's corresponding complete 3-models set (with `nets` and `optims`) in `train/afhq`:
```
 python src/train.py --data_dir data/afhq --model_dir train/afhq --img_size 256 --batch 8 --resume 50
```

* Make an averaged model (only for generation) from the directory of those, e.g. `train/select`:
```
 python src/swa.py -i train/select 
```

#### Few personal findings

1. Batch size is crucial for this network! Official settings are `batch=8` for size `256`, if you have large GPU RAM. One can fit batch 3 or 4 on 11gb GPU; those results are interesting, but less impressive. Batches of 2 or 1 are for the brave only.. Size is better kept as `256`; the network has auto-scaling layer count, but I didn't manage to get comparable results for size `512` with batches up to 7 (max for 32gb).
2. Model weights may seriously oscillate during training, especially for small batches (typical for Cycle- or Star- GANs), so it's better to save models frequently (there may be jewels). The best selected models can be mixed together with `swa.py` script for better stability. By default, Generator network is saved every 1000 iterations, and the full set - every 5000 iterations. 100k iterations (few days on a single GPU) may be enough; 200-250k would give pretty nice overfit.
3. Lambda coefficients `lambda_ds` (diversity), `lambda_cyc` (reconstruction) and `lambda_sty` (style) may be increased for smaller batches, especially if the goal is stylization, rather than photo-realistic macro transformation. The videos above, for instance, were made with these lambdas equal 3. The reference-based generation is nearly lost with such settings, but latent-based one can make nice art.
4. The order of domains in the training set matters a lot! I usually put photoreal first (as it will be the main source imagery), and something similar as second; but other approaches may go well too (and your mileage may vary).
5. I particularly love this network for its' failures. Even the flawed results (when the batches are small, the lambdas are wrong, etc.) are usually highly expressive and "inventive", just the kind of "AI own art", which is so spoken about. Experimenting with such aesthetics is a great fun.

## Generation

### Single images

* Transform image `test.jpg` with AFHQ model (can be downloaded [here](https://www.dropbox.com/s/etwm810v25h42sn/100000_nets_ema.ckpt?dl=0)):
```
python src/test.py --source test.jpg --model models/100000_nets_ema.ckpt
```
This will produce 3 images (one per trained domain in the model) in the `_out` directory.  
If `source` is a directory, every image in it will be processed accordingly. 

* Generate output for the domain(s), referenced by number(s):
```
python src/test.py --source test.jpg --model models/100000_nets_ema.ckpt --ref 0-1-2
```

* Generate output with reference image for domain 1 (ref filename must start with that number):
```
python src/test.py --source test.jpg --model models/100000_nets_ema.ckpt --ref 1-ref.jpg
```

### Animation

The commands below would output frame sequences and mp4 videos to the `_out` directory.  

* Process image `_in/test.jpg` with `mymodel.ckpt` model, interpolating between referenced domains, with total duration 100 frames:
```
python src/process.py --source _in/test.jpg --model models/mymodel.ckpt --frames 100 --ref 0-1-2
ffmpeg -y -v warning -i _out/test/%06d.jpg _out/test-mymodel.mp4
```

* Process video `_in/test.mp4` with `mymodel.ckpt` model, interpolating between referenced domains:
```
mkdir _in/test-tmp
ffmpeg -y -v warning -i _in/test.mp4 _in/test-tmp/%06d.jpg
python src/process.py --source _in/test-tmp --model models/mymodel.ckpt --ref 0-1-2
ffmpeg -y -v warning -i _out/test-tmp/%06d.jpg _out/test-mymodel.mp4
```
or, if you're on Windows:
```
process.bat mymodel.ckpt test.mp4 0-1-2
```

### Recursive generation 

* Generate video sequence `_out/recurs.mp4` with `mymodel.ckpt` model, interpolating between referenced domains (0,1,2) in a feedback loop for 100 frames, switching domain every 25 frames:
```
python src/process.py --model models/mymodel.ckpt --refs 0-1-2 --size 1280-720 --frames 100 --fstep 25 --out_dir _out/recurs --recurs 1
ffmpeg -y -v warning -i _out/recurs/%06d.jpg _out/recurs.mp4
```
(to start from existing `image.jpg`, replace `--size 1280-720` with `--source image.jpg`)

* Generate similar video sequence, drawing over contour mask `mapping.jpg` (useful for videomapping projections):
```
python src/process.py --model models/mymodel.ckpt --refs 0-1-2 --source mapping.jpg --frames 100 --fstep 25 --out_dir _out/mapping --recurs 0.4
ffmpeg -y -v warning -i _out/mapping/%06d.jpg _out/mapping.mp4
```

* corresponding batch commands on Windows:

`recurs.bat mymodel.ckpt 0-1-2 1280-720 100 --fstep 25`

`recurs.bat mymodel.ckpt 0-1-2 _in/mapping.jpg 100 --fstep 25` (with mask)

* To add some motion, apply `--move` argument, and edit `scale/shift/angle/shear` parameters if needed. 


## Credits

[StarGAN2]  
Copyright � 2020, NAVER Corp. All rights reserved.  
Made available under [Creative Commons BY-NC 4.0] license.  
Original paper: https://arxiv.org/abs/1912.01865  

[Creative Commons BY-NC 4.0]: <https://github.com/clovaai/stargan-v2/blob/master/LICENSE>
[StarGAN2]: <https://github.com/clovaai/stargan-v2>
[FFMPEG]: <https://ffmpeg.org/download.html>
