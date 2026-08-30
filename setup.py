from setuptools import setup, find_packages

setup(
    name="sde_md",
    version="0.1.0",
    # Automatically finds the sde_md folder and all sub-packages (envs, policies, etc.)
    packages=find_packages(),
)
