from setuptools import setup, find_packages

setup(
    name="rna3d_feature_extractor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "jupyter",
        "notebook",
        "biopython",
        "forgi",
        "viennarna",
    ],
    python_requires=">=3.8",
    description="RNA 3D structure feature extraction toolkit",
    author="RNA 3D Explorer Team",
    author_email="rna3d@example.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)