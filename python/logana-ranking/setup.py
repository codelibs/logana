#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
from typing import Any, Dict, TextIO

from setuptools import find_packages, setup

NAME = "logana-ranking"
DESCRIPTION = "Logana Ranking Library."
URL = "https://github.com/codelibs/logana"
EMAIL = "dev@codelibs.org"
AUTHOR = "CodeLibs Project"
REQUIRES_PYTHON = ">=3.6.0"
REQUIRED = ["numpy", "tensorflow", "tensorflow-serving-api", "tensorflow-ranking"]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about: Dict[str, Any] = {}
with open(os.path.join(here, "logana", "__version__.py")) as f_in:  # type: TextIO
    exec(f_in.read(), about)


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require={},
    include_package_data=True,
    license="Apache Software License",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
)
