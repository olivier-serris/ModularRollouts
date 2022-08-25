from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "matplotlib>=3.3.4",
    "omegaconf",
    "hydra-core",
]
setup(
    name="modular rollouts",
    version="0.0.1",
    author="Olivier Serris",
    author_email="serris@isir.upmc.fr",
    description="Modular Vectorized with Gym, Brax, and IsaacGym  ",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/olivier-serris/ModularRollouts",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={"console_scripts": ["hydra_app = hydra_app.main:main"]},
    python_requires=">=3.7",
)
