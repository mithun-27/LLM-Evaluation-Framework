import typer

app = typer.Typer(
    help="LLM Evaluation Framework CLI"
)


@app.command()
def hello():
    """Test the CLI."""
    typer.echo("🚀 Welcome to the LLM Evaluation Framework!")


@app.command()
def version():
    """Show project version."""
    typer.echo("Version: 1.0.0")


if __name__ == "__main__":
    app()