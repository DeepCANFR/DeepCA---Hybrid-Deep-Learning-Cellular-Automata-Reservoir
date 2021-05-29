import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brainslicer",
    version="0.0.1",
    author="TAE Lindell",
    author_email="trymlind@oslomet.no",
    description="Distributed neuron simulation in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepCANFR/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir",
    project_urls={
        "Bug Tracker": "https://github.com/DeepCANFR/DeepCA---Hybrid-Deep-Learning-Cellular-Automata-Reservoir/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "../brainslicer"},
    packages=setuptools.find_packages(where="../brainslicer"),
    python_requires=">=3.6",
)