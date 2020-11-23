import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_cnn",
    version="1.0",
    author="Marius Juston",
    author_email="marius.juston@hotmail.fr",
    description="Automatically designing CNN architectures using Genetic Algorithm for Image Classification "
                "implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Marius-Juston/AutoCNN",
    packages=setuptools.find_packages(),
    install_requires=['tensorflow>=2.0.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Image Processing'
    ],
    python_requires='>=3.6',
)
