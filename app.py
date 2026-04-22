import os
import cv2
import numpy as np
import time
import csv

from encryption.stream_cipher import encrypt, decrypt
from metrics import calculate_psnr, calculate_entropy
from ml_keygen import generate_ml_key
from attack_detector import detect_attack

input_folder = "Dataset"
enc_folder = "results/encrypted"
dec_folder = "results/decrypted"

os.makedirs(enc_folder, exist_ok=True)
os.makedirs(dec_folder, exist_ok=True)

# CSV file
csv_file = open("results/metrics.csv", "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["Image", "PSNR", "Entropy", "Enc_Time", "Dec_Time", "Security"])

total_enc_time = 0
total_dec_time = 0
count = 0

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)

    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.resize(img, (224, 224))

    # 🔹 ML-based Key Generation
    key = generate_ml_key(img.shape)

    # 🔹 Encryption
    start = time.time()
    encrypted = encrypt(img, key)
    enc_time = time.time() - start

    # 🔹 Decryption
    start = time.time()
    decrypted = decrypt(encrypted, key)
    dec_time = time.time() - start

    total_enc_time += enc_time
    total_dec_time += dec_time
    count += 1

    # Save images
    cv2.imwrite(os.path.join(enc_folder, filename), encrypted)
    cv2.imwrite(os.path.join(dec_folder, filename), decrypted)

    # Validation
    if not np.array_equal(img, decrypted):
        print("Mismatch:", filename)

    # 🔹 Metrics
    psnr = calculate_psnr(img, decrypted)
    entropy = calculate_entropy(encrypted)

    # 🔹 Attack Detection
    security_status = detect_attack(decrypted)

    # Print output
    print(f"{filename} | PSNR: {psnr:.2f} | Entropy: {entropy:.2f} | Status: {security_status}")

    # Save to CSV
    writer.writerow([filename, psnr, entropy, enc_time, dec_time, security_status])

# Close CSV
csv_file.close()

# Average times
if count > 0:
    print("Average Encryption Time:", total_enc_time / count)
    print("Average Decryption Time:", total_dec_time / count)

print("All images processed successfully ✅")