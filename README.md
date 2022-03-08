# CTGANs

## Set up

Only CTLearn installation is required:

1. `git clone https://github.com/ctlearn-project/ctlearn`
2. `cd ctlearn`
3. `git checkout tf_update`
4. `conda env create -f environment-gpu.yml`
5. `conda activate ctlearn_tf2`
6. `pip install .`

To plot model graphs:

8. `conda install pydot`

## Usage

First, update `config.yml` paths and make sure that the images have the same dimensions as the generator output (see `generator.py`) and the discriminator input (see `discriminator.py`). Then, simply run `main.py`.

## The model

<img title="CTGANs arquitecture" src="images/CTGANs.PNG">

## Results

Generated images after 1 epoch of training with a dataset of over 2.000.000 images:
<img title="generated_images" src="images/generated_images.png">
The real images corresponding to same labels are:
<img title="real_images" src="images/real_images.png">