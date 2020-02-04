This is the dataset for compiler identification.
When you use this dataset, decompress these .zip files, and the directory structure after decompression is as follows.
```
dataset/
 +compiler/
  +01_VC2003_32bit_none/
  +02_VC2017_32bit_none/
  +03_VC2003_32bit_max/
  +04_VC2017_32bit_max/
  +05_VC2017_64bit_none/
  +06_VC2017_64bit_max/
  +11_gcc_x86_none/
  +12_gcc_x86_O3/
  +13_gcc_64bit_none/
  +14_gcc_64bit_max/
  +21_clang_32_none/
  +22_clang_32_O3/
  +23_clang_64bit_none/
  +24_clang_64bit_max/
  +31_intel_32_none/
  +32_intel_32bit_max/
  +33_intel_64_none/
  +34_intel_64bit_max/
  +40_document
```
# elf_cof2bin
This script is a tool for making dataset and extracts code segments from object files.
Arguments are the path of the folder where the object files exist and the folder of the output destination.

```
> python3 elf_coff2bin.py [source code dirpath] [output dirpath]
```
e.g.,
```
> python3 elf_coff2bin.py ./object_files ./dataset
```
