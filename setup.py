"""Installation metadata for the mjlab HIM Go2 project."""

from setuptools import find_packages, setup


setup(
  name="mjlab_himloco",
  version="0.1.0",
  description="HIM locomotion training with mjlab and Unitree Go2",
  packages=find_packages(),
  include_package_data=True,
  package_data={
    "src.assets.robots.unitree_go2": ["xmls/*.xml", "xmls/assets/*"],
  },
  python_requires=">=3.10,<3.14",
  install_requires=["mjlab==1.6.0"],
)
