from setuptools import setup, find_packages

setup(
    name="csv2json",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'csv2json=csv2json.cli:main',
        ],
    },
    install_requires=[],
    author="User",
    author_email="user@example.com",
    description="Simple CSV to JSON converter CLI tool",
    python_requires='>=3.6',
)