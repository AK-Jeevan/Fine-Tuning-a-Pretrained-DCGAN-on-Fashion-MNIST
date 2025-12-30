# 🧠 Fine-Tuning a Pretrained DCGAN on Fashion-MNIST  
**TensorFlow / Keras · Hugging Face**

This repository demonstrates how to **fine-tune a pretrained Deep Convolutional Generative Adversarial Network (DCGAN)**—originally trained on the **CelebA** dataset—using **Fashion-MNIST** as a new target domain.

Instead of training a GAN from scratch, this project focuses on **GAN transfer learning**, showing how pretrained generative models can be adapted to new datasets using low learning rates and proper preprocessing.

---

## 🚀 What This Project Covers

- Loading **pretrained Generator and Discriminator** from Hugging Face  
- Adapting Fashion-MNIST to DCGAN input requirements  
- Image preprocessing:
  - Normalization to `[-1, 1]`
  - Resizing from `28×28 → 64×64`
  - Grayscale → RGB channel conversion  
- Building a `tf.data` pipeline with shuffling and batching  
- Fine-tuning using adversarial loss  
- Custom training loop with `tf.GradientTape`  
- Visualizing generated images after training  

---

## 🧠 Why Fine-Tune a Pretrained DCGAN?

This project helps you:

- Understand **GAN fine-tuning vs training from scratch**
- Reuse learned generative features from large datasets
- Improve training stability
- Follow **real-world industry workflows** for generative modeling

Fine-tuning is especially useful when:
- Training data is limited
- Pretrained GANs already capture meaningful visual structure

---

## 🏗️ Training Architecture

### 🔹 Generator (Pretrained)
- Originally trained on **CelebA**
- Input: Random noise vector (`latent_dim = 100`)
- Output: `64 × 64 × 3` RGB image
- Updated during fine-tuning

### 🔹 Discriminator (Pretrained)
- Binary classifier (real vs fake)
- Input: `64 × 64 × 3` images
- Trained jointly with the generator

---

## 🧪 Dataset Preparation

Fashion-MNIST images are transformed to match DCGAN expectations:

- Pixel values scaled from `[0, 255] → [-1, 1]`
- Images resized from `28×28` to `64×64`
- Grayscale images converted to RGB
- Data shuffled and batched using `tf.data.Dataset`

---

## 📉 Loss Functions

**Binary Cross Entropy**

### Generator Loss
- Encourages the discriminator to classify generated images as **real**

### Discriminator Loss
- Real images → label **1**
- Generated images → label **0**

### Optimizers
- Adam optimizer for both networks  
- Learning rate: `1e-5`  
- `beta_1 = 0.5` for stable adversarial training  

---

## 🔍 Key Concepts Explained in Code

- GAN fine-tuning and transfer learning  
- Domain adaptation for generative models  
- Custom training loops in TensorFlow  
- Adversarial loss balancing  
- Training stability considerations  

---

## 🖼️ Visualization

After training, the script:

- Generates **16 images**
- Displays them in a **4 × 4 grid**
- Rescales images from `[-1, 1] → [0, 1]` for visualization

This provides quick qualitative feedback on fine-tuning performance.

---

## 🧑‍🎓 Who Should Use This Repo?

- Learners exploring **DCGANs**
- Students studying **deep generative models**
- TensorFlow users interested in **GAN transfer learning**
- Anyone moving from **basic GANs → convolutional GANs**

---

## ⚠️ Important Notes

- This project **fine-tunes** a pretrained model — it does not train from scratch
- Image quality depends on pretrained weights and dataset compatibility
- Low learning rates are essential to avoid destroying pretrained features
- Fashion-MNIST is structurally different from CelebA, so domain mismatch artifacts are expected

---

## 📜 License

MIT License

---

## ⭐ Support

If this repository helped you:

⭐ Star the repo  
🧠 Share it with other ML learners  
