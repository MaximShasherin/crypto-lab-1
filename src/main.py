import os
import json
import csv
import hashlib
import numpy as np
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def xorshift32(seed, length):
    state = seed
    keystream = bytearray()
    for _ in range(length):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        keystream.append(state & 0xFF)
    return bytes(keystream)


def xor_crypt(data, key_str):
    seed = int(hashlib.md5(key_str.encode()).hexdigest()[:8], 16)
    ks = xorshift32(seed, len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


def aes_ecb_encrypt(data, key):
    cipher = AES.new(key, AES.MODE_ECB)
    return cipher.encrypt(pad(data, AES.block_size))


def aes_ecb_decrypt(data, key):
    cipher = AES.new(key, AES.MODE_ECB)
    return unpad(cipher.decrypt(data), AES.block_size)


def aes_cbc_encrypt(data, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct = cipher.encrypt(pad(data, AES.block_size))
    return cipher.iv + ct


def aes_cbc_decrypt(data, key):
    iv = data[:16]
    ct = data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)


def calc_entropy(data):
    _, counts = np.unique(list(data), return_counts=True)
    probs = counts / len(data)
    return -np.sum(probs * np.log2(probs))


def calc_correlation(img_array, direction='h'):
    arr = img_array.astype(float)
    if direction == 'h':
        x = arr[:, :-1].flatten()
        y = arr[:, 1:].flatten()
    else:
        x = arr[:-1, :].flatten()
        y = arr[1:, :].flatten()
    sample = min(5000, len(x))
    return pearsonr(x[:sample], y[:sample])[0]


def npcr_uaci(c1, c2):
    c1_arr = np.frombuffer(c1, dtype=np.uint8)
    c2_arr = np.frombuffer(c2, dtype=np.uint8)
    diff = (c1_arr != c2_arr)
    npcr = np.sum(diff) / len(c1_arr) * 100
    uaci = np.mean(np.abs(c1_arr.astype(int) - c2_arr.astype(int)) / 255.0) * 100
    return npcr, uaci


if __name__ == "__main__":

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    imgs_dir = os.path.join(base_dir, "imgs")
    results_dir = os.path.join(base_dir, "results")

    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    img_path = os.path.join(imgs_dir, "my_photo.png")

    if not os.path.exists(img_path):
        print(f"Ошибка: Положи картинку {img_path}")
        exit()

    img = Image.open(img_path).convert('RGB')
    if img.size[0] > 512 or img.size[1] > 512:
        img = img.resize((512, 512))

    img_bytes = img.tobytes()
    img_np = np.array(img)

    key = "secretkey123"
    aes_key = hashlib.sha256(key.encode()).digest()[:16]

    print("Шифрование...")

    xor_ct = xor_crypt(img_bytes, key)
    xor_pt = xor_crypt(xor_ct, key)
    assert xor_pt == img_bytes

    ecb_ct = aes_ecb_encrypt(img_bytes, aes_key)
    ecb_pt = aes_ecb_decrypt(ecb_ct, aes_key)
    assert ecb_pt == img_bytes

    cbc_ct = aes_cbc_encrypt(img_bytes, aes_key)
    cbc_pt = aes_cbc_decrypt(cbc_ct, aes_key)
    assert cbc_pt == img_bytes

    print("Все тесты на обратимость пройдены!")

    Image.frombytes('RGB', img.size, xor_ct).save(os.path.join(imgs_dir, "xor_cipher.png"))
    Image.frombytes('RGB', img.size, xor_pt).save(os.path.join(imgs_dir, "xor_decrypted.png"))

    with open(os.path.join(imgs_dir, "ecb_cipher.bin"), "wb") as f:
        f.write(ecb_ct)
    Image.frombytes('RGB', img.size, ecb_pt).save(os.path.join(imgs_dir, "ecb_decrypted.png"))

    with open(os.path.join(imgs_dir, "cbc_cipher.bin"), "wb") as f:
        f.write(cbc_ct)
    Image.frombytes('RGB', img.size, cbc_pt).save(os.path.join(imgs_dir, "cbc_decrypted.png"))

    print("\nСчитаем метрики...")
    n = img_np.size

    methods = {
        "Original": img_bytes,
        "XOR": xor_ct[:n],
        "AES-ECB": ecb_ct[:n],
        "AES-CBC": cbc_ct[:n]
    }

    metrics_list = []
    print(f"{'Метод':<15} | {'Энтропия':<10} | {'Corr H':<10} | {'Corr V':<10}")
    print("-" * 55)

    for name, data in methods.items():
        ent = calc_entropy(data)
        arr = np.frombuffer(data, dtype=np.uint8)[:n].reshape(img_np.shape)
        corr_h = calc_correlation(arr, 'h')
        corr_v = calc_correlation(arr, 'v')
        print(f"{name:<15} | {ent:<10.4f} | {corr_h:<10.4f} | {corr_v:<10.4f}")

        metrics_list.append({
            "method": name,
            "entropy": round(ent, 4),
            "corr_h": round(corr_h, 4),
            "corr_v": round(corr_v, 4)
        })

    key2 = "secretkey124"
    aes_key2 = hashlib.sha256(key2.encode()).digest()[:16]
    ecb_ct2 = aes_ecb_encrypt(img_bytes, aes_key2)
    npcr, uaci = npcr_uaci(ecb_ct, ecb_ct2)
    print(f"\nNPCR (AES-ECB): {npcr:.2f}%")
    print(f"UACI (AES-ECB): {uaci:.2f}%")

    with open(os.path.join(results_dir, "metrics.csv"), "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["method", "entropy", "corr_h", "corr_v"])
        writer.writeheader()
        writer.writerows(metrics_list)

    meta_data = {
        "source_image": "my_photo.png",
        "algorithm": "AES-CBC",
        "iv": cbc_ct[:16].hex(),
        "npcr_aes_ecb": round(npcr, 2),
        "uaci_aes_ecb": round(uaci, 2)
    }
    with open(os.path.join(results_dir, "meta.json"), "w", encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)

    plt.figure(figsize=(12, 8))
    for i, (name, data) in enumerate(methods.items()):
        plt.subplot(2, 2, i + 1)
        sample_data = np.random.choice(list(data), size=min(50000, len(data)), replace=False)
        plt.hist(sample_data, bins=256, color='black', alpha=0.7)
        plt.title(f"Гистограмма: {name}")
        plt.xlabel("Значение байта")
        plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "histograms.png"))
    print("\nФайлы сохранены!")