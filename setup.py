from setuptools import setup

setup(
    name='mditre',
    version='0.1.0',    
    description='Microbial Differentibale Temporal Rule Engine',
    url='https://github.com/gerberlab/mditre',
    author='Venkata Suhas Maringanti',
    author_email='vsuhas.m@gmail.com',
    license='GPLv3',
    packages=['mditre', 'mditre.datasets'],
    install_requires=[
                      'numpy',
                      'scikit-learn',
                      'PyQt5==5.11.3',
                      'ete3',
                      'matplotlib',
                      'seaborn',
                      'pandas',
                      'scipy',
                      'dendropy',
                      'jupyterlab',                    
                      ],

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
