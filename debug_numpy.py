<<<<<<< HEAD
version https://git-lfs.github.com/spec/v1
oid sha256:59f781a9e0a764ffaf1f7184a9ca1895842e5a425a9d8fa57dd4f3a72758a013
size 621
=======

import sys
import os

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

print("\nsys.path:")
for p in sys.path:
    print(p)

# Check for a directory named 'numpy' in the current directory
if 'numpy' in os.listdir('.'):
    print("\nWARNING: A directory named 'numpy' exists in the current directory.")
    print("This can cause import conflicts.")

try:
    import numpy
    print("\nSuccessfully imported numpy")
    print(f"Numpy version: {numpy.__version__}")
    print(f"Numpy path: {numpy.__file__}")
except ImportError as e:
    print(f"\nFailed to import numpy. Error: {e}")
>>>>>>> origin/window
