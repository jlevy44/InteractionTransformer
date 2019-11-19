from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
PACKAGES=['scikit-learn',
          'shap',
          'dask',
          'patsy',
          'safe-transformer',
          'numpy',
          'pandas',
          'kneed',
          'imbalanced-learn',
          'pysnooper',
          'matplotlib>=3.0.0',
          'openml==0.9.0']

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

class CustomInstallCommand(install):
    """Custom install setup to help run shell commands (outside shell) before installation"""
    def run(self):
        self.do_egg_install()

setup(name='interactiontransformer',
      version='0.1',
      description='Extract interactions from complex model using SHAP and add to linear model..',
      url='https://github.com/jlevy44/InteractionExtractor',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['interactiontransformer'],
      install_requires=PACKAGES)
