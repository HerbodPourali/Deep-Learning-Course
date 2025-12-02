# Deep Learning — Notebook Collection

This repository contains a structured set of notebooks developed throughout an advanced graduate-level Deep Learning course.  
The material progresses from foundational concepts to modern architectures, covering core neural-network mechanics, key computer-vision tasks, sequence models and LLMs, generative modeling, and self-supervised/text–image techniques.

The repository is organized into five parts.

---

## Part 1 — Core Foundations

**Notebooks:**
- `P1_01_Basics.ipynb`
- `P1_02_NN_Scratch.ipynb`
- `P1_03_Optimization.ipynb`
- `P1_04_Lazy_Gradient.ipynb`
- `pytorch_basic.py`

**Topics include:**
- Tensor manipulation, PyTorch basics, and autograd  
- A full feed-forward neural network implemented **from scratch** using NumPy  
- Manual forward/backward passes with gradient checking  
- First-order and second-order optimization algorithms  
- Lazy gradient evaluation and related performance considerations  

---

## Part 2 — Computer Vision

**Notebooks:**
- `P2_01_Classification.ipynb`
- `P2_02_Segmentation.ipynb`
- `P2_03_Detection.ipynb`

**Topics include:**
- Image classification (training workflows, metrics, visualization)  
- Semantic segmentation using paired image–mask datasets  
- Object detection with bounding boxes and non-maximum suppression  
- Evaluation metrics such as accuracy, mIoU, and mAP  
- Qualitative visualization and analysis of model predictions  

---

## Part 3 — Sequence Modeling & LLMs

**Notebooks:**
- `P3_01_RNN.ipynb`
- `P3_02_GPT2.ipynb`
- `P3_03_PEFT.ipynb`
- `P3_04_Reasoning.ipynb`

**Topics include:**
- Classical sequence models: RNNs, LSTMs, and GRUs  
- GPT-2 causal language modeling (training, sampling, and evaluation)  
- Parameter-efficient fine-tuning (LoRA / PEFT)  
- Inference-time reasoning strategies: Chain-of-Thought, self-consistency, best-of-n  
- Perplexity analysis and text-generation evaluation  

---

## Part 4 — Generative Models

**Notebooks:**
- `P4_01_VAE.ipynb`
- `P4_02_DDPM.ipynb`

**Topics include:**
- Variational Autoencoders (ELBO, β-VAE, latent-space traversals)  
- Denoising Diffusion Probabilistic Models (DDPM)  
- Forward diffusion, reverse sampling, and UNet-based denoisers  
- Visualization and analysis of generative behavior  

---

## Part 5 — Self-Supervision & Text–Image Methods

**Notebooks:**
- `P5_01_DINO.ipynb`
- `P5_02_CLIP.ipynb`
- `P5_03_StableDiffusion.ipynb`

**Topics include:**
- DINO self-supervised learning (EMA teacher–student, centering, multi-crop augmentation)  
- CLIP-guided image optimization and prompt engineering  
- Stable Diffusion for text-to-image, image-to-image, and inpainting pipelines  
- Practical inference settings, sampling parameters, and qualitative evaluation  

---

## Academic Context

These notebooks were prepared as part of the **Deep Learning course at SUT (MSc)**.  
They reflect hands-on implementations designed to build practical intuition for foundational and modern deep-learning methods.

---

## 📄 License

This repository is licensed under the **MIT License**.  
See the `LICENSE` file for details.
