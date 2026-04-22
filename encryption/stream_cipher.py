import numpy as np

def generate_key(shape):
    return np.random.randint(0, 256, shape, dtype=np.uint8)

def encrypt(img, key):
    return np.bitwise_xor(img, key)

def decrypt(enc, key):
    return np.bitwise_xor(enc, key)