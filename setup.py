"""Setup for the project."""
from setuptools import setup

setup(
    name="bps",
    version=1.0,
    description="Basis point set (BPS) library for efficient point cloud encoding",
    setup_requires=["numpy", "sklearn", "tqdm"],
    install_requires=[
        "sklearn",
        "tqdm",
        "numpy"
    ],
    author="Sergey Prokudin",
    license="MIT-0",
    author_email="prokus@amazon.com",
    packages=["bps"]
)