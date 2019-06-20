from setuptools import find_packages, setup

setup(
    name='prices',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas==0.24.2',
        'scikit_learn==0.21.2'
    ],
    version='0.1.0',
    description='This challenge aims to predict interactions between atoms. Imaging technologies '
                'like MRI enable us to see and understand the molecular composition of tissues. '
                'Nuclear Magnetic Resonance (NMR) is a closely related technology which uses the '
                'same principles to understand the structure and dynamics of proteins and molecules.',
    author='Milan & Frank',
    license='',
    scripts=['prices/bin/prices'],
    data_files=[('prices_data', ['logger.ini'])]
)
