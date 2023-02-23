<!-- markdownlint-disable MD046 -->
<!-- markdownlint-disable-next-line MD041 -->

# Getting started with `neuralfields`

This pages explains how to install `neuralfields` via the package management system `poetry`, how the CI/CD pipelines
are set up using `poe` tasks, and how `pre-commit` hooks are configures to keep the commits clean.

## Installation

### Installation of `poetry`

This project is managed by [poetry], a Python packaging and dependency management tool.
Therefore, `poetry` must be installed _first_.

Please have a look at [the official poetry documentation][poetry-doc-install] on how to install `poetry` on
different platforms and under different conditions.

### Installation of `neuralfields`

The installation of this project is quite straightforward. Simply go to the project's directory, and run

```sh
poetry install
```

!!! tip "No project development intended?"

    If you don't need any development setup, you can pass the `--no-dev` flag to skip the development dependencies.

??? fail "Computer says no..."

    Please find a collection of known failure cases below, and feel free to report more.

    | Symptom                                         | Hint                                                            |
    | :---------------------------------------------- | :-------------------------------------------------------------- |
    | _Something is wrong with my poetry environment_ | Delete the `.venv` folder and recreate the virtual environment. |

## Dependency Management & Packaging

As mentioned in the [Installation](#installation) section, [poetry] is employed to keep the dependencies of different
projects from interfering with each other.
By running `poetry install` in the project's root folder, a separate virtual environment is created into which all
dependencies are installed automatically (typically takes a few seconds to minutes).
This is similar to running `pip install -r requirements.txt` in an isolated virtual environment.

## Poe Task Runner

This project defines so-called tasks using [poe] which are executed in commit hooks as well as the CI/CD pipeline.
These tasks are essentially a sequence of commands, and can also be executed locally in the terminal by running

```sh
poetry run poe <task_name>
```

??? info "Available tasks"

    To get a list of available tasks, execute `poetry run poe --help`

    ```sh
    --8<-- "docs/exported/poe_options.txt"
    ```

## Git Hooks

This project uses [pre-commit] hooks to automatically check for common formatting issues.
The hooks are executed before every commit, but can be disabled by adding `--no-verify` when committing, e.g.
`git commit . -m "Fixed something" --no-verify`.

!!! info "Installation of pre-commit"

    After you cloned this project and plan to develop in it, don't forget to install these hooks via

    ```sh
    poetry run pre-commit install
    ```

??? example "Available pre-commit hooks"

    The pre-commit hooks are configured in the `.pre-commit-config.yaml` file as follows

    ```yaml
    --8<-- ".pre-commit-config.yaml"
    ```

## GitHub Actions

There are basic CI, CD, and Release pipelines which are executed as [GitHub actions workflow][gh-workflows] on pushing
changes or opening pull requests.

??? example "Available workflows"

    === ".github/workflows/ci.yaml"

        ```yaml
        --8<-- ".github/workflows/ci.yaml"
        ```

    === ".github/workflows/cd.yaml"

        ```yaml
        --8<-- ".github/workflows/cd.yaml"
        ```

    === ".github/workflows/release.yaml"

        ```yaml
        --8<-- ".github/workflows/release.yaml"
        ```


<!-- URLs -->
[gh-workflows]: https://docs.github.com/en/actions/using-workflows
[poe]: https://github.com/nat-n/poethepoet/
[poetry]: https://python-poetry.org/
[poetry-doc-install]: https://python-poetry.org/docs/#installation
[pre-commit]: https://pre-commit.com/
