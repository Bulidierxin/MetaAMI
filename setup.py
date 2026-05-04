from setuptools import setup, find_packages

setup(
    name="MetaAMI",
    version="0.1.0",
    description=(
        "A meta-learning framework based on random projection and "
        "stacking learning for in-hospital mortality prediction in "
        "acute myocardial infarction"
    ),
    url="https://github.com/wan-mlab/RPSLearner",
    author="Bulidierxin Tuerhanbayi, Shibiao Wan",
    author_email="btuerhanbayi@unmc.edu",
    license="MIT",

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "joblib>=1.2",
    ],

    extras_require={
        "full": [
            "xgboost>=1.7.0",
            "torch>=1.0",
        ]
    },

    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],

    python_requires=">=3.8",
)
