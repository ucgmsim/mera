from setuptools import setup, find_packages

setup(
    name="mera",
    version="22.8.1",
    packages=find_packages(),
    url="https://github.com/ucgmsim/mera",
    description="Code for GM prediction mixed-effects regression analysis",
    install_requires=[
        "numpy==1.26.4",
        "statsmodels",
        "natsort",
        "pandas",
        "pymer4 @ git+https://github.com/ucgmsim/pymer4.git@get_ranef_cond_std",
    ],
)
