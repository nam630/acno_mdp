from setuptools import setup, find_packages

setup(name='custom_mountain',
        packages=find_packages(),
        include_package_data=False,
        version='0.0.1',
        install_requires=['gym', 'numpy', 'pandas', 'joblib']
    )

