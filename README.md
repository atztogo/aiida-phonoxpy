[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/atztogo/aiida-phonoxpy/develop.svg)](https://results.pre-commit.ci/latest/github/aiida-phonopy/aiida-phonopy/develop)

# AiiDA yet another phonopy plugin

This is an unofficial aiida-phonopy plugin for automating phonopy and phono3py calculation. VASP and QE are supported.

## Documentation

https://atztogo.github.io/aiida-phonoxpy/

## Test

```bash
% pip install -e ."[tests]"
% pytest
```

## Development

The development is managed on the `develop` branch.

- Github issues is the place to discuss about phonopy issues.
- Github pull request is the place to request merging source code.
- Formatting is written in `pyproject.toml`.
- Not strictly, but VSCode's `settings.json` may be written like

  ```json
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=88", "--ignore=E203,W503"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.linting.pycodestyleEnabled": false,
  "python.linting.pydocstyleEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=88"],
  "python.sortImports.args": ["--profile", "black"],
  "[python]": {
      "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
  }
  ```

- Use of pre-commit (https://pre-commit.com/) is encouraged.
  - Installed by `pip install pre-commit`, `conda install pre_commit` or see
    https://pre-commit.com/#install.
  - pre-commit hook is installed by `pre-commit install`.
  - pre-commit hook is run by `pre-commit run --all-files`.
