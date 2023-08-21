from setuptools import setup, find_packages

# Read the contents of the README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='DebyeCalculator',
    version='0.1',
    description='A fast vectorized implementation of the Debye equation on both CPU and GPU.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Frederik L. Johansen and Andy S. Anker',
    author_email='fljo@di.ku.dk and andy@chem.ku.dk',
    install_requires=[
        'numpy',
        'ase',
    ],
    packages=find_packages(
        exclude=[
            'generate_nanoparticles.py', 
            'produce_figure_data.py', 
            'SASCalculator.py'
        ],
    ),
    package_data={'': ['*.yaml']},
    url='https://github.com/FrederikLizakJohansen/DebyeCalculator',
    license='Apache 2.0',
    license_files=['LICENSE*'],
)