"""Quick smoke test to validate all modules."""

import sys
import traceback
import numpy as np

tests_pass = 0
tests_fail = 0


def test(name, fn):
    global tests_pass, tests_fail
    try:
        fn()
        print(f"  PASS: {name}")
        tests_pass += 1
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
        traceback.print_exc()
        tests_fail += 1


print("=== Module Import Tests ===")
test("crypto.kdf", lambda: __import__("crypto.kdf"))
test("crypto.aes", lambda: __import__("crypto.aes"))
test("stego.lsb_fixed", lambda: __import__("stego.lsb_fixed"))
test("stego.entropy", lambda: __import__("stego.entropy"))
test("stego.fuzzy", lambda: __import__("stego.fuzzy"))
test("stego.lsb_adaptive", lambda: __import__("stego.lsb_adaptive"))
test("analysis.metrics", lambda: __import__("analysis.metrics"))
test("analysis.steganalysis", lambda: __import__("analysis.steganalysis"))
test("analysis.robustness", lambda: __import__("analysis.robustness"))

print()
print("=== Functional Tests ===")


def test_kdf():
    from crypto.kdf import derive_key, validate_key_entropy
    bundle = derive_key("test_password", algorithm="pbkdf2")
    assert len(bundle.key) == 32
    assert validate_key_entropy(bundle.key)


test("KDF derive key", test_kdf)


def test_aes():
    from crypto.aes import encrypt, decrypt
    payload = encrypt(b"Hello Steganography!", "password123", kdf_algorithm="pbkdf2")
    recovered = decrypt(payload, "password123")
    assert recovered == b"Hello Steganography!"


test("AES-GCM encrypt/decrypt", test_aes)


def test_lsb_fixed_1bit():
    from stego.lsb_fixed import embed_fixed, extract_fixed
    cover = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    msg = b"Secret message for testing!"
    stego = embed_fixed(cover, msg, lsb_depth=1, seed=42)
    extracted = extract_fixed(stego, lsb_depth=1, seed=42)
    assert extracted == msg


test("Fixed LSB embed/extract (1-bit)", test_lsb_fixed_1bit)


def test_lsb_fixed_2bit():
    from stego.lsb_fixed import embed_fixed, extract_fixed
    cover = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    msg = b"Two-bit LSB test payload!"
    stego = embed_fixed(cover, msg, lsb_depth=2, seed=42)
    extracted = extract_fixed(stego, lsb_depth=2, seed=42)
    assert extracted == msg


test("Fixed LSB embed/extract (2-bit)", test_lsb_fixed_2bit)


def test_metrics():
    from analysis.metrics import compute_all
    a = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = (int(b[0, 0, 0]) + 1) % 256
    m = compute_all(a, b, payload_bits=8)
    assert "psnr" in m and "ssim" in m


test("Quality metrics", test_metrics)


def test_steganalysis():
    from analysis.steganalysis import run_all_detectors
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    results = run_all_detectors(img)
    assert "rs" in results and "chi_square" in results and "spa" in results


test("Steganalysis detectors", test_steganalysis)


def test_fuzzy():
    import yaml
    from stego.fuzzy import FuzzyDepthController
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    ctrl = FuzzyDepthController.from_config(cfg["stego"]["fuzzy"])
    e_map = np.random.uniform(0, 8, (32, 32))
    g_map = np.random.uniform(0, 1, (32, 32))
    depth = ctrl.infer(e_map, g_map, pressure=0.5)
    assert depth.shape == (32, 32)
    assert depth.min() >= 1 and depth.max() <= 3


test("Fuzzy depth controller", test_fuzzy)

print()
print(f"Results: {tests_pass} passed, {tests_fail} failed")
sys.exit(0 if tests_fail == 0 else 1)
