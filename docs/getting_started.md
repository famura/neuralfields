<!-- markdownlint-disable MD046 -->
<!-- markdownlint-disable-next-line MD041 -->

# Getting started with `neuralfields`

This pages explains how to install `neuralfields` via the package management system `uv`, how the CI/CD pipelines
are set up using Task, and how `pre-commit` hooks are configured to keep the commits clean.

## Installation

### Installation of `uv`

This project is managed by [uv], a Python packaging and dependency management tool.
Therefore, `uv` must be installed _first_.

Please have a look at [the official uv documentation][uv-doc-install] on how to install `uv` on
different platforms and under different conditions.

### Installation of Task

The project commands are defined with [Task]. Install Task using the instructions in its official documentation.

### Installation of `neuralfields`

The installation of this project is quite straightforward. Simply go to the project's directory, and run

```sh
uv sync
```

!!! tip "No project development intended?"

    If you don't need any development setup, run `uv sync --no-dev` to skip the development dependencies.

??? fail "Computer says no..."

    Please find a collection of known failure cases below, and feel free to report more.

    | Symptom                                     | Hint                                               |
    | :------------------------------------------ | :------------------------------------------------- |
    | _Something is wrong with my uv environment_ | Delete the `.venv` folder and run `uv sync` again. |

## Dependency Management & Packaging

As mentioned in the [Installation](#installation) section, [uv] keeps the dependencies of different projects from
interfering with each other. By running `uv sync` in the project's root folder, a separate `.venv` is created into
which all runtime and development dependencies are installed automatically (typically takes a few seconds to minutes).
This is similar to running `pip install -r requirements.txt` in an isolated virtual environment.

## Project Tasks

Common project tasks are exposed through Task and run their Python tools through `uv`:

```sh
task test
```

??? info "Available tasks"

    To get a list of available tasks, execute `task --list`.

    ```sh
    --8<-- "docs/exported/task_options.txt"
    ```

## Git Hooks

This project uses [pre-commit] hooks to automatically check for common formatting issues.
The hooks are executed before every commit, but can be disabled by adding `--no-verify` when committing, e.g.
`git commit . -m "Fixed something" --no-verify`.

!!! info "Installation of pre-commit"

    After you cloned this project and plan to develop in it, don't forget to install these hooks via

    ```sh
    task hooks
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
[Task]: https://taskfile.dev/
[uv]: https://docs.astral.sh/uv/
[uv-doc-install]: https://docs.astral.sh/uv/getting-started/installation/
[pre-commit]: https://pre-commit.com/
