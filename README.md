# ppo-dice

## Please use hyper parameters from this readme. With other hyper parameters things might not work (it's RL after all)!

This repo contains a PyTorch implementation for the paper

[Stable Policy Optimization via Off-Policy Divergence Regularization. Ahmed Touati, Amy Zhang, Joelle Pineau and Pascal Vincent. UAI2020](https://arxiv.org/abs/2003.04108)

```
@article{touati2020stable,
  title={Stable Policy Optimization via Off-Policy Divergence Regularization},
  author={Touati, Ahmed and Zhang, Amy and Pineau, Joelle and Vincent, Pascal},
  journal={arXiv preprint arXiv:2003.04108},
  year={2020}
}
```

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash
# PyTorch
conda install pytorch torchvision -c soumith

# Baselines for Atari preprocessing
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

## Training

### Atari

```
 ./run_local_atari.sh
```

### Deepmind Control

```
 ./run_local.sh
```

# LICENSE

[Attribution-NonCommercial 4.0 International](/LICENSE)
