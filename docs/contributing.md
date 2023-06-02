# Contributing to rllte

Thank you for using and contributing to rllte project!!!👋👋👋 Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it:
    - Post about your intended feature in an [issue](https://github.com/RLE-Foundation/rllte/issues), and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.

2. You want to implement a feature or bug-fix for an outstanding issue:
    - Search for your issue in the [rllte issue list](https://github.com/RLE-Foundation/rllte/issues).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, please submit a Pull Request to [https://github.com/RLE-Foundation/rllte](https://github.com/RLE-Foundation/rllte).

## Get rllte
Open up a terminal and clone the repository from [GitHub](https://github.com/RLE-Foundation/rllte) with `git`:
``` sh
git clone https://github.com/RLE-Foundation/rllte.git
cd rllte/
```
After that, run the following command to install package and dependencies:
``` sh
pip install -e .[all]
```

## Codestyle
We use [black codestyle](https://github.com/psf/black) (max line length of 127 characters) together with isort to sort the imports. For the documentation, we use the default line length of 88 characters per line.

**Please run `make format`** to reformat your code. You can check the codestyle using make `check-codestyle` and `make lint`.

Please document each function/method and [type](https://google.github.io/pytype/user_guide.html) them using the following [Google style docstring](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) template:
```
def function_with_types_in_docstring(param1: type1, param2: type2):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (type1): The first parameter.
        param2 (type2): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
```

## Pull Request (PR)
Before proposing a PR, please open an issue, where the feature will be discussed. This prevent from duplicated PR to be proposed and also ease the code review process. Each PR need to be reviewed and accepted by at least one of the maintainers (@[yuanmingqi](https://github.com/yuanmingqi), @[ShihaoLuo](https://github.com/orgs/RLE-Foundation/people/ShihaoLuo)). A PR must pass the Continuous Integration tests to be merged with the master branch.

See the [Pull Request Template](https://github.com/RLE-Foundation/rllte/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

## Tests
All new features must add tests in the `tests/` folder ensuring that everything works fine. We use [pytest](https://pytest.org/). Also, when a bug fix is proposed, tests should be added to avoid regression.

To run tests with `pytest`:

```
make pytest
```

Type checking with `pytype`:

```
make type
```

Codestyle check with `black`, `isort` and `ruff`:

```
make check-codestyle
make lint
```

To run `type`, `format` and `lint` in one command:
```
make commit-checks
```

## Acknowledgement
This contributing guide is based on the [stable-Baselines3](https://github.com/DLR-RM/stable-baselines3/blob/master/CONTRIBUTING.md) one.