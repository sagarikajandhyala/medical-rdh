"""security/decrypt.py — AES-256-GCM decryption"""
import base64
from Crypto.Cipher import AES

def decrypt_payload(packed: bytes, key_b64: str) -> bytes:
    key   = base64.b64decode(key_b64)
    nonce, tag, ct = packed[:16], packed[16:32], packed[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    try:
        pt = cipher.decrypt_and_verify(ct, tag)
    except ValueError as e:
        raise ValueError(f"[decrypt] Authentication FAILED: {e}")
    print(f"  [decrypt] Authentication ✓  Recovered {len(pt)} bytes")
    return pt
