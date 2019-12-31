from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
PACKAGES=['scikit-learn>=0.22',
          'shap>=0.34.0',
          'dask>=2.9.1',
          'patsy',
          'safe-transformer>=0.0.5',
          'numpy>=0.16.4',
          'pandas>=0.25.3',
          'kneed>=0.5.1',
          'imbalanced-learn>=0.6.1',
          'pysnooper>=0.3.0',
          'matplotlib>=3.0.0',
          'openml==0.9.0']

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

class CustomInstallCommand(install):
    """Custom install setup to help run shell commands (outside shell) before installation"""
    def run(self):
        self.do_egg_install()

setup(name='interactiontransformer',
      version='0.1.1',
      description='Extract interactions from complex model using SHAP and add to linear model..',
      url='https://github.com/jlevy44/InteractionExtractor',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['interactiontransformer'],
      install_requires=PACKAGES)
