import base64
import hashlib

def get_hash_of_two_strings(string1: str, string2: str) -> str:
    """Hashes two strings together and returns the result."""

    string_hash = hashlib.sha256((string1 + string2).encode())

    return base64.b64encode(string_hash.digest()).decode("utf-8")