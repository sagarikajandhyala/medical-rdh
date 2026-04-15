"""security/encrypt.py — AES-256-GCM encryption"""
import os, base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

def encrypt_payload(plaintext: bytes, key: bytes = None):
    if key is None: key = get_random_bytes(32)
    cipher = AES.new(key, AES.MODE_GCM)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    packed = cipher.nonce + tag + ct
    meta = {
        "algorithm": "AES-256-GCM",
        "key_b64":   base64.b64encode(key).decode(),
        "nonce_b64": base64.b64encode(cipher.nonce).decode(),
        "tag_b64":   base64.b64encode(tag).decode(),
        "plaintext_len": len(plaintext),
    }
    print(f"  [encrypt] {len(plaintext)}B → {len(packed)}B (AES-256-GCM)")
    return packed, meta
