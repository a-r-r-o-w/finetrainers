from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

with open("requirements.txt", "r", encoding="utf-8") as file:
    requirements = [line for line in file.read().splitlines() if len(line) > 0]

setup(
    name="finetrainers",
    version="0.2.0.dev0",
    description="Finetrainers is a work-in-progress library to support (accessible) training of diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Aryan V S",
    author_email="contact.aryanvs@gmail.com",
    url="https://github.com/a-r-r-o-w/finetrainers",
    python_requires=">=3.8.0",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"dev": ["pytest==8.3.2", "ruff==0.1.5"]},
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Steps to publish:
# 1. Update version in setup.py
# 2. python setup.py sdist bdist_wheel
# 3. Check if everything works with testpypi:
#    twine upload --repository testpypi dist/*
# 4. Upload to pypi:
#    twine upload dist/*
