from setuptools import setup, find_packages

setup(
    name='RosExtractor',
    version='1.0',
    packages=find_packages(),
    maintainer='Cuisset Mattéo',
    maintainer_email='matteo.cuisset@gmail.com',
    description='A flask API to generate image with AI',
    long_description=open('README.md').read(),
    install_requires=['Flask', 'python-dotenv'], # TODO : Add others dependencies
)
