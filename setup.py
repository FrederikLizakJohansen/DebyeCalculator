from setuptools import setup, find_packages

setup(
    name='DebyeCalculator',
    version='0.1',
    description='A fast vectorized implementation of the Debye equation on both CPU and GPU.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Frederik L. Johansen and Andy S. Anker',
    author_email='frjo@di.ku.dk and andy@chem.ku.dk',
    install_requires=[
        'numpy',
        'pyyaml',
        'ase',
        'torch',
        'torchvision',
        'torchaudio',
    ],
    packages=find_packages(
        where='DebyeCalculator',
        exclude=[
            'generate_nanoparticles*', 
            'produce_figure_data*', 
            'SASCalculator*'
        ],
    ),
    include_package_data=True,
    package_data={'': ['*.yaml']},
    url='https://github.com/FrederikLizakJohansen/DebyeCalculator',
    license='Apache 2.0',
    license_files=['LICENSE*'],
    classifiers=[
        'Programming Language :: Python'
    ]
)
