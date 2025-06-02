import click

from snanomaly.dataset.exception import DataPointNotFoundError
from snanomaly.dataset.factory import OSCFactory


@click.command()
@click.argument("sn_name", required=True)
def sninfo(sn_name: str):
    try:
        sn_obj = OSCFactory.OSC2018June().load_datapoint(name=sn_name)
        click.echo(sn_obj)
    except DataPointNotFoundError:
        click.echo(message=f"Could not find supernova candidate with name `{sn_name}`", err=True)



