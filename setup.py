from setuptools import setup

setup(
    name='pytorch_cem',
    version='0.1.0',
    packages=['pytorch_cem'],
    url='https://github.com/LemonPi/pytorch_cem',
    license='MIT',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='Cross Entropy Method (CEM) implemented in pytorch',
    install_requires=[
        'torch',
        'numpy'
    ],
    tests_requires=[
        'gym<=0.20',
        'pygame'
    ]
)
