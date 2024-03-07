from setuptools import setup

package_name = "eml_rl"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ryan Clifford",
    maintainer_email="",
    description="F1Tenth Reinforcement Learning for NCSU EML",
    license="MIT License",
    tests_require=["pytest"],
)
