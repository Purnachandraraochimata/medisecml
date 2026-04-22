import numpy as np

def calculate_psnr(original, decrypted):
    mse = np.mean((original - decrypted) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_entropy(image):
    hist = np.histogram(image.flatten(), bins=256)[0]
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))