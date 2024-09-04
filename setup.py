from setuptools import setup, find_packages

setup(
    name='casegen-mc',
    version='0.1.01',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit',
        'matplotlib',

    ],
    author='lvenneri',
    author_email='lorenzo.venneri@gmail.com',
    description='Case generator, optimzer, and summarizer for models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lveneri/casegen-mc',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

