from setuptools import find_packages, setup

setup(
    name="cgpos",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    version="0.1.0",
    description="POS tagging for Clasical Greek.",
    author="Tejomay Gadgil",
    author_email="tejomay.gadgil@gmail.com",
    license="MIT",
)
