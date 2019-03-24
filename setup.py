import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sknet",
    version="0.0.1",
    author="Randall Balestriero",
    author_email="randallbalestriero@gmail.com",
    description="sknet package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RandallBalestriero/sknet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
)
