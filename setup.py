import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="deeprob-kit",
    version="1.1.0",
    author="Lorenzo Loconte, Gennaro Gala",
    author_email="lorenzoloconte@outlook.it, g.gala@tue.nl",
    description="A Python Library for Deep Probabilistic Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deeprob-org/deeprob-kit",
    packages=setuptools.find_packages(exclude=['test', 'experiments', 'benchmark']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)
