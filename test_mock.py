import sys
from importlib import metadata

original_version = metadata.version

def mocked_version(pkg):
    if pkg == 'bitsandbytes':
        return '0.41.1'
    return original_version(pkg)

metadata.version = mocked_version

try:
    metadata.version('bitsandbytes')
    print("Mock worked!")
except Exception as e:
    print(f"Failed: {e}")
