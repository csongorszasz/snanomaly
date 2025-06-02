import click


@click.group()
def interpolate():
    pass

@interpolate.command()
@click.option("--sn", help="Supernova candidate name", required=True)
def one(sn: str):
    click.echo(f"Interpolating `{sn}`")

@interpolate.command()
def batch():
    pass
