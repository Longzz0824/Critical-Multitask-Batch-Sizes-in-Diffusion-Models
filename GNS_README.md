This repo features an improved PyTorch implementation for the paper [**Scalable Diffusion Models with Transformers**](https://www.wpeebles.com/DiT).

It contains:

* 🪐 An improved PyTorch [implementation](models.py) and the original [implementation](train_options/models_original.py) of DiT
* ⚡️ Pre-trained class-conditional DiT models trained on ImageNet (512x512 and 256x256)
* 💥 A self-contained [Hugging Face Space](https://huggingface.co/spaces/wpeebles/DiT) and [Colab notebook](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb) for running pre-trained DiT-XL/2 models
* 🛸 An improved DiT [training script](train.py) and several [training options](train_options)
