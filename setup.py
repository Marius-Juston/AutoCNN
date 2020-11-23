import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_cnn-mjuston",  # Replace with your own username
    version="0.0.1",
    author="Marius Juston",
    author_email="marius.juston@hotmail.fr",
    description="Automatically Designing CNN Architectures Using Genetic Algorithm for Image Classification "
                "implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Marius-Juston/AutoCNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
