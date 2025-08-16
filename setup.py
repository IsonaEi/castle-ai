from setuptools import setup, find_packages


def readme():
    try:
        with open("README.md", encoding="UTF-8") as readme_file:
            return readme_file.read()
    except TypeError:
        # Python 2.7 doesn't support encoding argument in builtin open
        import io

        with io.open("README.md", encoding="UTF-8") as readme_file:
            return readme_file.read()





configuration = {
    "name": "castle-ai",
    "version": "1.0.0",
    "description": "a training‑free foundation‑model pipeline for unsupervised, cross‑species behavioral classification",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "classifiers": [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
    ],
    "maintainer": "Raiso Liu",
    "maintainer_email": "raiso@as.edu.tw",
    "license": "Apache License",
    "packages": find_packages(),
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "pytest",
    "tests_require": ["pytest"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)