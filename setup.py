from setuptools import find_packages, setup  # type: ignore[import-not-found]
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(path: str) -> List[str]:
    with open(path) as fs:
        requirements = fs.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="Student_Performance_Indicator",
    version="0.0.1",
    author="Sarvadnya",
    author_email="yeolesv1012@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)