"""
Title: Fine-Tuning a Pretrained DCGAN on a New Dataset (Fashion-MNIST)

Description:
This script demonstrates how to fine-tune a pretrained Deep Convolutional
Generative Adversarial Network (DCGAN) originally trained on the CelebA dataset
using a new target dataset (Fashion-MNIST).

Key Steps:
1. Load pretrained Generator and Discriminator models from Hugging Face.
2. Preprocess the new dataset to match the pretrained DCGAN requirements
   (normalization, resizing, and channel alignment).
3. Define adversarial loss functions and optimizers with a low learning rate
   for stable fine-tuning.
4. Continue adversarial training using custom training loops in TensorFlow.
5. Generate and visualize images after fine-tuning.

Purpose:
- To understand practical fine-tuning of GANs rather than training from scratch.
- To align theoretical GAN knowledge with real-world, industry-style workflows.

Frameworks Used:
- TensorFlow / Keras
- Hugging Face Hub
- Matplotlib
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

gen_path = hf_hub_download(repo_id="hussamalafandi/DCGAN_CelebA", filename="generator.h5")
disc_path = hf_hub_download(repo_id="hussamalafandi/DCGAN_CelebA", filename="discriminator.h5")

generator = tf.keras.models.load_model(gen_path)
discriminator = tf.keras.models.load_model(disc_path)

print("✅ Pretrained DCGAN loaded")


# ------------------------------
# Prepare the Fashion-MNIST dataset to match pretrained DCGAN inputs
# ------------------------------

# Load Fashion-MNIST dataset; we only need images so ignore labels with _
(x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

# Convert integer pixels to float32 for math operations
x_train = x_train.astype("float32")
# Scale pixels from [0,255] to [-1,1] because DCGAN expects that range
x_train = (x_train - 127.5) / 127.5
# Add a channel dimension to make shape (28,28,1) from (28,28)
x_train = np.expand_dims(x_train, axis=-1)
# Resize images from 28x28 to 64x64 because the pretrained generator
# expects 64x64 images (CelebA resolution). tf.image.resize returns floats.
x_train = tf.image.resize(x_train, (64, 64))
# Convert grayscale images to RGB by repeating the single channel 3 times
x_train = tf.image.grayscale_to_rgb(x_train)

# Define constants for dataset pipeline buffering and batch size
BUFFER_SIZE = 60000
BATCH_SIZE = 128

# Create a tf.data.Dataset from the NumPy/Tensor array
dataset = tf.data.Dataset.from_tensor_slices(x_train)
# Shuffle the dataset to randomize sample order and then batch it
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# ------------------------------
# Loss functions and optimizers used for fine-tuning
# ------------------------------

# Use binary cross-entropy since GAN outputs are probabilities (sigmoid)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def generator_loss(fake_output):
    # Compare discriminator's output on fake images to 1s (wants them seen as real)
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    # Discriminator should classify real images as ones
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Discriminator should classify fake images as zeros
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # Total discriminator loss is sum of both parts
    return real_loss + fake_loss


# Use small learning rates for fine-tuning to avoid destroying pretrained weights
generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)


# ------------------------------
# Training step: single minibatch update for both networks
# ------------------------------

# Dimension of the random noise vector fed to generator
noise_dim = 100


@tf.function
def train_step(real_images):
    # Get batch size dynamically from input real images tensor
    batch_size = tf.shape(real_images)[0]
    # Sample random normal noise vectors for the whole batch
    noise = tf.random.normal([batch_size, noise_dim])

    # Use two gradient tapes: one for generator, one for discriminator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images from random noise using the generator model
        fake_images = generator(noise, training=True)

        # Discriminator prediction on real images
        real_output = discriminator(real_images, training=True)
        # Discriminator prediction on generated (fake) images
        fake_output = discriminator(fake_images, training=True)

        # Compute generator loss encouraging generator to fool discriminator
        gen_loss = generator_loss(fake_output)
        # Compute discriminator loss on both real and fake samples
        disc_loss = discriminator_loss(real_output, fake_output)

    # Compute gradients of generator loss w.r.t generator's trainable variables
    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # Compute gradients of discriminator loss w.r.t discriminator's variables
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply generator gradients to update generator weights
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    # Apply discriminator gradients to update discriminator weights
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

    # Return scalar losses so caller can log or monitor training progress
    return gen_loss, disc_loss


# ------------------------------
# Training loop: run multiple epochs over the dataset
# ------------------------------

EPOCHS = 20  # Number of passes over the whole dataset

for epoch in range(EPOCHS):
    # Iterate over all batches in the dataset
    for image_batch in dataset:
        # Perform one training step and get losses
        g_loss, d_loss = train_step(image_batch)

    # After each epoch, print a short summary of current losses
    print(f"Epoch {epoch+1}/{EPOCHS} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")


# ------------------------------
# Utility: generate a grid of images and show them with matplotlib
# ------------------------------

def generate_and_plot():
    # Sample 16 random noise vectors to generate a 4x4 grid
    noise = tf.random.normal([16, noise_dim])
    # Produce images with generator (inference mode)
    images = generator(noise, training=False)
    # Convert images from [-1,1] back to [0,1] for display
    images = (images + 1) / 2.0

    # Create a matplotlib figure to hold the grid
    plt.figure(figsize=(6, 6))
    # Loop over generated images and put each on a subplot
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        # Display the i-th image in the grid
        plt.imshow(images[i])
        # Turn off axis ticks and labels for a cleaner look
        plt.axis("off")
    # Show the plotted figure on screen
    plt.show()

generate_and_plot()
