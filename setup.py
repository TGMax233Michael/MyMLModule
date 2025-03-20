from setuptools import setup, find_packages

setup(
    name='myMLModule',
    version='0.1.0',
    description='自己制作的机器学习库, gong',
    author='TGMax233_Michael',
    author_email='youremail@test.com',
    packages=['myMLModule'],
    install_requires=[
        'numpy',
        'pandas'
    ],
)
