from setuptools import setup, find_packages

setup(
    name='DebyeCalculator',
    version='0.1',
    description='A fast vectorized implementation of the Debye equation on both CPU and GPU.',
    author='Frederik L. Johansen and Andy S. Anker',
    author_email='fljo@di.ku.dk and andy@chem.ku.dk',
    install_requires=[
        '',
    ],
    packages=find_packages(
        exclude=['generate_nanoparticles.py', 'produce_figure_data.py'],
    ),
    package_data={'': ['*.yaml']}
    
)