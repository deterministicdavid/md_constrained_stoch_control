import os
from setuptools import setup, Extension
import pybind11

current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_source = os.path.join(current_dir, "csrc", "sde_core.cpp")

# Define the C++ extension module
ext_modules = [
    Extension(
        "sde_cpp",  # Name of the compiled module
        [cpp_source],  # Source files
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],  # Optimization flags
    ),
]

setup(
    name="sde_cpp",
    version="1.0",
    ext_modules=ext_modules,
)
