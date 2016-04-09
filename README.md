# Visual7W QA Models

![Visual7W QA samples](http://web.stanford.edu/~yukez/images/img/visual7w_examples.png "Visual7W example QAs")

## Introduction

[Visual7W](http://web.stanford.edu/~yukez/visual7w/) is a large-scale visual question answering (QA) dataset, with object-level groundings and multimodal answers.
Each question starts with one of the seven Ws, *what*, *where*, *when*, *who*, *why*, *how* and *which*. Please check out [our CVPR'16 paper](http://web.stanford.edu/~yukez/papers/cvpr2016.pdf) for more details. This repository provides a [torch](http://torch.ch/) implementation of the attention-based QA model from our paper. Part of the code is adapted from [neuraltalk2](https://github.com/karpathy/neuraltalk2).

## Dataset Overview
The [Visual7W](http://web.stanford.edu/~yukez/visual7w/) dataset is collected on 47,300 COCO images. In total, it has 327,939 QA pairs, together with 1,311,756 human-generated multiple-choices and 561,459 object groundings from 36,579 categories. In addition, we provide complete grounding annotations that link the object mentions in the QA sentences to their bounding boxes in the images and therefore introduce a new QA type with image regions as the visually grounded answers. We refer to questions with textual answers
as *telling* QA and to such with visual answers as *pointing* QA. The figure above shows some examples in the [Visual7W](http://web.stanford.edu/~yukez/visual7w/) dataset, where the first row shows *telling* QA examples, and the second row shows *pointing* QA examples.

[Visual7W](http://web.stanford.edu/~yukez/visual7w/) constitutes a part of the [Visual Genome](http://visualgenome.org/) project. Visual Genome contains 1.7 million QA pairs of the 7W question types, which offers the largest visual QA collection to date for training models. The QA pairs in [Visual7W](http://web.stanford.edu/~yukez/visual7w/) are a subset of the 1.7 million QA pairs from Visual Genome. Moreover, [Visual7W](http://web.stanford.edu/~yukez/visual7w/) includes extra annotations such as object groundings, multiple choices and human experiments, making it a clean and complete benchmark for evaluation and analysis.

## Dependencies
1.  **Python 2.7**
    - required: [h5py](http://www.h5py.org/), [numpy](http://www.numpy.org/), [skimage](http://scikit-image.org/)
2.  **Lua 5.2**
    -  required: [torch](http://torch.ch/), [nn](https://github.com/torch/nn), [nngraph](https://github.com/torch/nngraph), [hdf5](https://github.com/deepmind/torch-hdf5), [loadcaffe](https://github.com/szagoruyko/loadcaffe), [cjson](https://github.com/mpx/lua-cjson), [image](https://github.com/torch/image)
    -  optional: [cutorch](https://github.com/torch/cutorch), [cunn](https://github.com/torch/cunn), [cudnn](https://github.com/soumith/cudnn.torch)  (for GPU support)

To install these dependencies after installing Python and Lua:
```Shell
# Install Python dependencies using pip
pip install numpy h5py scikit-image

# Install most Lua dependencies using luarocks
luarocks install torch
luarocks install nn
luarocks install nngraph
luarocks install lua-cjson
luarocks install image # required for demo.lua

# Install torch-hdf5 from git repo
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec

# (Optional) Install packages for GPU support, which require CUDA 6.5 or higher.
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

## How to Use
In this section, we describe the steps to set up the codebase for training new QA models as well as evaluating their performances. You can use similar procedures to develop new models, and test on your own data.

**Step 1**: Get the code base and submodules (using the --recursive flag).
```bash
git clone --recursive https://github.com/yukezhu/visual7w-qa-models.git
```

**Step 2**: Simply run the downloading script in the root folder. It takes care of downloading everything needed to run the whole pipeline, including the QA data, images and a pretrained CNN model (VGGNet-16).
```bash
./download_data.sh
```

**Step 3**: Process the raw dataset into a single hdf5 file that is easy to parse by torch. By default, it will create `qa_data.h5` and `qa_data.json` in the `data` folder. Make sure the QA data and images are in the right place (from Step 2) before runing this script.
```bash
python prepare_dataset.py
```

**Step 4**: We are all set. Now let's have fun training and evaluating.
```bash
# Training Mode
# the default parameters work with the default setup
# we strongly recommend you to use GPU mode for training
# use flag -h to see helper infomation
th train_telling.lua -h

# default command for training the model on GPU #0 without finetuning the CNN
# it should train a model that has very similar performances as reported in our paper
th train_telling.lua -gpuid 0 -mc_evaluation -verbose -finetune_cnn_after -1

# Evaluation Mode
# you need to specify which model you want to evaluate
# use flag -h to see helper infomation
th eval_telling.lua -model <path-to-model> -mc_evaluation
```

## Model Zoo
To make it easy, we have released a list of pre-trained QA models for you to play with.
These models are trained on the *telling* QA tasks, using the [Visual7W](http://web.stanford.edu/~yukez/visual7w/) dataset and the larger Visual Genome dataset. You can download these models in both CPU and GPU modes below.

Dataset                       | Num. QA  | What  | Where | When  | Who  | Why  | How  | Overall |
----------------------------- |-------------------| ------| ------| ------| -----| -----| -----| --------|
Visual7W telling ([gpu](http://vision.stanford.edu/yukezhu/model_visual7w_telling_gpu.t7)\|[cpu](http://vision.stanford.edu/yukezhu/model_visual7w_telling_cpu.t7)) | 139,868 | 0.529	| 0.560	| 0.743	| 0.602	| 0.522	| 0.466	| 0.541 |
Visual Genome telling ([gpu](http://vision.stanford.edu/yukezhu/model_visualgenome_telling_gpu.t7)\|[cpu](http://vision.stanford.edu/yukezhu/model_visualgenome_telling_gpu.t7))    | 1,359,108 | - | - | - | - | - | - | - |

**Note:**
- Visual7W QA is a subset of Visual Genome QA, but has additional annotations (such as *multiple choices* and *object groundings*) for evaluation and analysis. The numbers are multiple-choice accuracies reported on the Visual7W test set.
- You can use the script `gpu_to_cpu.lua` to convert a GPU model to a CPU copy.

## Visual QA Demo
We have provided a demo script for you to run a pretrained QA model on your own image and ask your own questions. `demo.lua` has provided a pipeline for answering a list of sample questions (written in `demo.lua`) on a [demo image](https://raw.githubusercontent.com/yukezhu/visual7w-qa-models/master/data/demo.jpg). Use the following commands to run the QA demo.
```
# run demo script on GPU mode
wget http://vision.stanford.edu/yukezhu/model_visual7w_telling_gpu.t7 -P checkpoints
th demo.lua -model checkpoints/model_visual7w_telling_gpu.t7 -gpuid 0

# alternatively, run demo script on CPU mode
wget http://vision.stanford.edu/yukezhu/model_visual7w_telling_cpu.t7 -P checkpoints
th demo.lua -model checkpoints/model_visual7w_telling_cpu.t7 -gpuid -1
```

You will see the QA model produces reasonable answers on the [demo image](https://raw.githubusercontent.com/yukezhu/visual7w-qa-models/master/data/demo.jpg) below. Feel free to try your own images or ask your own questions :)

![Visual7W QA demo](https://raw.githubusercontent.com/yukezhu/visual7w-qa-models/master/data/demo.jpg "Visual7W QA demo")

```
** QA demo on data/demo.jpg **

Q: how many people are there ?
A: two .

Q: what animal can be seen in the picture ?
A: elephant .

Q: who is wearing a red shirt ?
A: the man on the right .

Q: where color is the elephant ?
A: gray .

Q: when is the picture taken ?
A: daytime .
```

## Reference
Please acknowledge the our CVPR'16 paper if you are using this code.
```
@InProceedings{zhu2016cvpr,
  title = {{Visual7W: Grounded Question Answering in Images}},
  author = {Yuke Zhu and Oliver Groth and Michael Bernstein and Li Fei-Fei},
  booktitle = {{IEEE Conference on Computer Vision and Pattern Recognition}},
  year = 2016,
}
```
