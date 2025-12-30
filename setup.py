from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="car-vision-toolkit",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for vehicle vision analysis using Qwen-VL models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/car-vision-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "car-vision=main:main",
        ],
    },
)