from setuptools import setup

setup(
    name="dkm",
    packages=["dkm"],
    version="0.1.0",
    author="Johan Edstedt",
    install_requires=open("requirements.txt", "r").read().split("\n"),
)
