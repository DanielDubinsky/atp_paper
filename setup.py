from setuptools import setup, find_packages

setup(
    name='atp',
    version='0.0.0',
    description='atp prediction',
    author='Daniel Dubinsky',
    author_email='danield95@gmail.com',
    url='https://github.com/QurisAI/atp_paper.git',
    install_requires=['lightning'],
    packages=find_packages(),
)