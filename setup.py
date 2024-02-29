from setuptools import setup, find_packages

setup(
    name="ssl_fts",
    version="2024.0",
    description="Self-supervised learning for Time Series",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    package_data={
        "": ["data/*.csv"],
    },
)
