#打包文件

from setuptools import setup, find_packages

setup(
    name="T2S",
    version="0.1.0",
    packages=find_packages(where="."),  # 确保从当前目录找
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "nltk",
        "evaluate",
        "tensorboard",
    ],
    entry_points={
        "console_scripts": [
            "T2S=T2S.cli:main"
        ]
    },
    python_requires=">=3.8",
)
