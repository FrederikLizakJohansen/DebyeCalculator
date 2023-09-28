from setuptools import setup, find_packages
from pip.req import parse_requirements

# parse requirements file
install_reqs = parse_requirements('requirements.txt', session='hack')

# get list of requirements
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='DebyeCalculator',
    version='0.1',
    description='A fast vectorized implementation of the Debye equation on both CPU and GPU.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Frederik L. Johansen and Andy S. Anker',
    author_email='frjo@di.ku.dk and andy@chem.ku.dk',
    install_requires=reqs,
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
