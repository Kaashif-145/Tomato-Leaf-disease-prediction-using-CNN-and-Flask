#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils.ipynb

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import random

def build_model(base_model, input_shape, num_classes, fine_tune_at=None):
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)  # Output shape depends on model

    # Check if output is 4D (CNN) or 2D (ViT)
    if len(x.shape) == 4:
        x = layers.GlobalAveragePooling2D()(x)  # Only for CNN backbones

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model



def augmix_image(image):
    """AugMix implementation (simplified)."""
    ws = np.float32(np.random.dirichlet([1.]*3))
    m = np.float32(np.random.beta(1., 1.))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(3):
        image_aug = image.copy()
        op = random.choice([
            lambda x: tf.image.random_brightness(x, max_delta=0.2),
            lambda x: tf.image.random_contrast(x, lower=0.5, upper=1.5),
            lambda x: tf.image.random_flip_left_right(x),
        ])
        image_aug = op(image_aug)
        mix += ws[i] * image_aug

    mixed = (1 - m) * image + m * mix
    return tf.clip_by_value(mixed, 0, 1)


# In[ ]:




