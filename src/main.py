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

    indices = np.random.choice(len(x), size=min(10000, len(x)), replace=False)
    return pearsonr(x[indices], y[indices])[0]


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

    np.random.seed(42)

    images = ["my_photo.png", "gradient.png", "checkerboard.png"]

    key = "secretkey123"
    aes_key = hashlib.sha256(key.encode()).digest()[:16]

    aes_key2_arr = bytearray(aes_key)
    key2 = "secretkey124"
    aes_key2 = hashlib.sha256(key2.encode()).digest()[:16]

    all_metrics = []
    npcr_rows = []
    meta_all = {}

    for img_name in images:
        img_path = os.path.join(imgs_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Пропуск: {img_path} не найден")
            continue

        print(f"\n=== Обработка {img_name} ===")

        img = Image.open(img_path).convert('RGB')
        if img.size[0] > 512 or img.size[1] > 512:
            img = img.resize((512, 512))

        img_bytes = img.tobytes()
        img_np = np.array(img)
        n = img_np.size

        xor_ct = xor_crypt(img_bytes, key)
        assert xor_crypt(xor_ct, key) == img_bytes

        ecb_ct = aes_ecb_encrypt(img_bytes, aes_key)
        assert aes_ecb_decrypt(ecb_ct, aes_key) == img_bytes

        cbc_ct = aes_cbc_encrypt(img_bytes, aes_key)
        assert aes_cbc_decrypt(cbc_ct, aes_key) == img_bytes

        print("  Тесты на обратимость пройдены.")

        stem = os.path.splitext(img_name)[0]

        # Сохраняем результаты шифрования этого изображения
        Image.frombytes('RGB', img.size, xor_ct).save(
            os.path.join(imgs_dir, f"{stem}_xor_cipher.png"))
        with open(os.path.join(imgs_dir, f"{stem}_ecb_cipher.bin"), "wb") as f:
            f.write(ecb_ct)
        with open(os.path.join(imgs_dir, f"{stem}_cbc_cipher.bin"), "wb") as f:
            f.write(cbc_ct)

        methods = {
            "Original": img_bytes,
            "XOR": xor_ct[:n],
            "AES-ECB": ecb_ct[:n],
            "AES-CBC": cbc_ct[16:16 + n],
        }

        for name, data in methods.items():
            ent = calc_entropy(data)
            arr = np.frombuffer(data, dtype=np.uint8)[:n].reshape(img_np.shape)
            corr_h = calc_correlation(arr, 'h')
            corr_v = calc_correlation(arr, 'v')
            print(f"  {name:<12} ent={ent:.4f} corr_h={corr_h:.4f} corr_v={corr_v:.4f}")
            all_metrics.append({
                "image": img_name,
                "method": name,
                "entropy": round(ent, 4),
                "corr_h": round(corr_h, 4),
                "corr_v": round(corr_v, 4),
            })

        ecb_ct2 = aes_ecb_encrypt(img_bytes, aes_key2)
        npcr, uaci = npcr_uaci(ecb_ct, ecb_ct2)
        print(f"  NPCR={npcr:.2f}%  UACI={uaci:.2f}%")
        npcr_rows.append({
            "image": img_name,
            "npcr": round(npcr, 2),
            "uaci": round(uaci, 2),
        })

        meta_all[img_name] = {
            "source_image": img_name,
            "algorithm": "AES-CBC",
            "iv": cbc_ct[:16].hex(),
            "npcr_method": "AES-ECB",
            "npcr": round(npcr, 2),
            "uaci": round(uaci, 2),
        }

        # гистограмма по этому изображению
        plt.figure(figsize=(12, 8))
        for i, (name, data) in enumerate(methods.items()):
            plt.subplot(2, 2, i + 1)
            sample_data = np.random.choice(
                list(data), size=min(50000, len(data)), replace=False)
            plt.hist(sample_data, bins=256, color='black', alpha=0.7)
            plt.title(f"{stem}: {name}")
            plt.xlabel("Значение байта")
            plt.ylabel("Частота")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"histograms_{stem}.png"))
        plt.close()

    # Сохраняем сводные результаты
    with open(os.path.join(results_dir, "metrics.csv"), "w",
              newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "method", "entropy", "corr_h", "corr_v"])
        writer.writeheader()
        writer.writerows(all_metrics)

    with open(os.path.join(results_dir, "npcr_uaci.csv"), "w",
              newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image", "npcr", "uaci"])
        writer.writeheader()
        writer.writerows(npcr_rows)

    with open(os.path.join(results_dir, "meta.json"), "w",
              encoding='utf-8') as f:
        json.dump(meta_all, f, indent=4)

    print("\nВсё сохранено в results/")