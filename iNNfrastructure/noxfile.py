import nox


@nox.session
def testing(session):
    session.install(".")
    session.install("pytest")
    session.install("coverage")

    session.run("coverage", "run", "-m", "pytest", "-k", "not gcloud")


@nox.session
def coverage(session):
    session.install("coverage")

    session.run("coverage", "report", "--fail-under=100")


@nox.session
def linting(session):
    session.install("flake8")

    session.run("flake8", "innfrastructure", "tests")


@nox.session
def docstyle(session):
    session.install("pydocstyle")

    session.run("pydocstyle", "innfrastructure")
