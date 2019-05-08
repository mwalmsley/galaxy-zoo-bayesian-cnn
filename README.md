[![DOI](https://zenodo.org/badge/185564932.svg)](https://zenodo.org/badge/latestdoi/185564932)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

# "Zoobot" Galaxy Zoo Bayesian CNN and Active Learning - Public Release

This repository contains code to simulate training a Bayesian CNN using active learning on Galaxy Zoo, for the paper:

> Galaxy Zoo: Probabilistic Morphology through Bayesian CNNs and Active Learning, M. Walmsley et al, Monthly Notices of the Royal Society, 2019 (submitted)

**This code is released for reproducibility only**. The code is an exact copy of the code used for the paper. As a research best practice, the code is documented, tested, and (we hope) fairly readable. However, we **strongly suggest** any users interested in further original research **wait for the launch of Yarin Gal's Bayesian Deep Learning Competition** (Q3 2019), which will include much of this code in a friendlier form.

## Installation

The core features are provided as the `zoobot` package. Install as per Python convention:
`pip install -r requirements.txt`
`pip install -e zoobot`

## Usage

We provide example scripts for using the `zoobot` package in the root directory. 

- `run_zoobot_on_panoptes.py` trains and evaluates a model on a train/test split of either GZ2 or GZ DECALS galaxies. This is useful for developing models. 
- `panoptes_to_tfrecord.py` and `gz2_to_tfrecord.py` convert the respective download catalogs into tfrecords suitable for models.

To simulate active learning, see the readme in `zoobot/active_learning`.

*Note that the Galaxy Zoo 2 and Nasa Sloan Atlas galaxy catalogs required are available externally from [here](data.galaxyzoo.org) and [here](https://www.sdss.org/dr13/manga/manga-target-selection/nsa/).*

## Zoobot Folders
- `estimators` includes custom TensorFlow models (and associated training routines and input utilities) for galaxy classification. 
- `active_learning` applies those models and routines in an active learning loop. Currently, previous classifications are used as an oracle. **See the Readme in this folder for more details.**
- `get_catalogs` is used to download GZ2 classifications. Panoptes classifications have been refactored to the repo `gz-panoptes-reduction`.
- `tests` contains unit tests for the package. Look here for examples!
- `tfrecord` has various useful utilities for reading and writing galaxy catalogs to tfrecords.
- `uncertainty` has code to find the coverage fractions of trained models


## Contact Us

This code was written by [Mike Walmsley](walmsley.dev). Please get in touch by [email](mailto:mike.walmsley@physics.ox.ac.uk).

## Legal Stuff

Copyright (C) 2019 - Mike Walmsley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.