from setuptools import setup, find_packages

setup(
    name="hc_xps",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "hc_xps": ["spectrum_config/*.toml"]
    },
    install_requires=['numpy', 'matplotlib', 'lmfit'],
    python_requires='>=3.11',
    author='Piotr Toka',
    author_email='pnt17@ic.ac.uk',
    description='A package for XPS data analysis of hard carbon materials',
    url='https://github.com/pntoka/HC_XPS.git',
)
