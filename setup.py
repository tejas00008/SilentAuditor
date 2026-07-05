from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="silentauditor",
    version="0.1.0",
    description="AI-powered accounts payable fraud detection and audit system",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "silentauditor=src.main:cli",
        ],
    },
)
