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
          'kneed']

with open('README.md','r', encoding='utf-8') as f:
      long_description = f.read()

class CustomInstallCommand(install):
    """Custom install setup to help run shell commands (outside shell) before installation"""
    def run(self):
        self.do_egg_install()

setup(name='interactionextractor',
      version='0.1',
      description='Extract interactions from complex model using SHAP and add to linear model..',
      url='https://github.com/jlevy44/InteractionExtractor',
      author='Joshua Levy',
      author_email='joshualevy44@berkeley.edu',
      license='MIT',
      scripts=['bin/update_kneed'],
      #cmdclass={'install': CustomInstallCommand},
      # entry_points={
      #       'console_scripts':[]
      # },
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['interactionextractor'],
      install_requires=PACKAGES)
