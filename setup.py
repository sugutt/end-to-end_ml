
from setuptools import find_packages,setup
from typing import List

e_dot = '-e .'	
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path, 'r') as file_object:
        requirements = file_object.readlines()
        requirements = [require.replace("\n", "") for require in requirements]

        if e_dot in requirements:
            requirements.remove(e_dot)

            return requirements

setup(
    name='ml_project',
    version='0.1',
    author='Sugutt',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),

)