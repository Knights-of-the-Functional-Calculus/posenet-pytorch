from distutils.core import setup

setup(
    name='posenet-torch',
    version='0.0.1',
    packages=['posenet','posenet.converter', 'posenet.models'],
    package_data={'': ['converter/config.yaml']},
	include_package_data=True,
)
