# Neural Fields &ndash; Old Idea, New Glory

[![license][license-badge]][license]
[![docs][docs-stable-badge]][docs-stable]
[![docs][docs-latest-badge]][docs-latest]
[![pre-commit][pre-commit-badge]][pre-commit]
[![bandit][bandit-badge]][bandit-hp]
[![isort][isort-badge]][isort-hp]
[![black][black-badge]][black]
[![ci][ci-badge]][ci]
[![tests][tests-badge]][tests]
[![coverage][coverage-badge]][coverage]

## About

In 1977, Shun-ichi Amari introduced _neural fields_, a class of potential-based recurrent neural networks [1].
This architecture was developed as a simplistic model of the activity of neurons in a (human) brain.
It's main characteristic is the lateral in-/exhibition of neurons though their accumulated potential.
Due to its simplicity and expressiveness, Amari’s work was highly influential and led to several follow-up papers such
as [2, 3, 4, 5] to only name a few.

## Citing

If you use code or ideas from this repository for your projects or research, **please cite and star** it.
It does not cost you anything, and would support me for putting in the effort of providing a clean state-of-the-art
implementation to you.

```
@misc{Muratore_neuralfields,
  author = {Fabio Muratore},
  title = {neuralfields - A type of potential-based recurrent neural networks implemented with PyTorch},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/famura/neuralfields}}
}
```

## Features

* There are two variants of the neural fields implemented in this repository: one called `NeuralField` that matches
  the model of Amari closely using 1D convolutions, as well as another one called `SimpleNeuralField` that replaces the
  convolutions and introduces custom potential dynamics function.
* Both implementations have by modern standards very few, i.e. typically less than 1000, parameters. I suggest that you
  start with the `NeuralField` class since it is more expressive. However, the `SimpleNeuralField` has the benefit of
  operating with typically less than 20 parameters, which allows you to use optimizers that otherwise might not scale.
* Both model classes are subclasses of `torch.nn.Module`, hence able to process batched data and run on GPUs.
* This repository is a spin-off from [SimuRLacra](https://github.com/famura/SimuRLacra) where the neural fields have
  been used as the backbone for control policies. In `SimuRLacra`, the focus is on reinforcement learning for
  sim-to-real transfer. However, the goal of this repository is to make the implementation **as general as possible**,
  such that it could for example be used as generative model.

## Getting Started

To install this package, simply run

```sh
pip install neuralfields
```

For further information, please have a look at the [getting started guide][docs-getting-started].
In the documentation, you can also find the [complete reference of the source code][docs-code-reference].

---
[1] Shun-ichi Amari. _Dynamics of pattern formation in lateral-inhibition type neural fields_. Biological Cybernetics, 27(2):77–87, 1977.<br />
[2] K. Kishimoto and Shun-ichi Amari. _Existence and stability of local excitations in homogeneous neural fields_. Journal of Mathematical Biology, 7(4):303–318, 1979.<br />
[3] Wolfram Erlhagen and Gregor Schöner. _Dynamic field theory of movement preparation_. Psychological Review, 109(3):545–572, 2002.<br />
[4] Shun-ichi Amari, Hyeyoung Park, and Tomoko Ozeki. _Singularities affect dynamics of learning in neuromanifolds_. Neural Computation, 18(5):1007–1065, 2006.<br />
[5] T. Luksch, M. Gineger, M. Mühlig, T. Yoshiike, _Adaptive Movement Sequences and Predictive Decisions based on Hierarchical Dynamical Systems_. International Conference on Intelligent Robots and Systems, 2012.


<!-- URLs -->
[bandit-badge]: https://img.shields.io/badge/security-bandit-green.svg
[bandit-hp]: https://github.com/PyCQA/bandit
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black]: https://github.com/psf/black
[ci-badge]: https://github.com/famura/neuralfields/actions/workflows/ci.yaml/badge.svg
[ci]: https://github.com/famura/neuralfields/actions/workflows/ci.yaml
[coverage-badge]: https://famura.github.io/neuralfields/latest/exported/coverage/badge.svg
[coverage]: https://famura.github.io/neuralfields/latest/exported/coverage/report
[docs-stable-badge]: https://img.shields.io/badge/docs-stable-informational
[docs-latest-badge]: https://img.shields.io/badge/docs-latest-informational
[docs-code-reference]: https://famura.github.io/neuralfields/stable/reference
[docs-getting-started]: https://famura.github.io/neuralfields/stable/getting_started
[docs-stable]: https://famura.github.io/neuralfields/stable
[docs-latest]: https://famura.github.io/neuralfields/latest
[isort-badge]: https://img.shields.io/badge/imports-isort-green
[isort-hp]: https://pycqa.github.io/isort/
[license-badge]: https://img.shields.io/badge/license-MIT--v4-informational
[license]: https://github.com/famura/neuralfields/LICENSE.txt
[pre-commit-badge]: https://img.shields.io/badge/pre--commit-enabled-green
[pre-commit]: https://github.com/pre-commit/pre-commit
[tests-badge]: https://famura.github.io/neuralfields/latest/exported/tests/badge.svg
[tests]: https://famura.github.io/neuralfields/latest/exported/tests/report
