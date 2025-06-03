from typing import Optional

import click

from snanomaly.dataset.exception import DataPointNotFoundError
from snanomaly.dataset.factory import OSCFactory
from snanomaly.visualization.photometry import PlotPhotometry


@click.command()
@click.argument("sn_name", required=True, type=str, metavar="SN_NAME")
@click.option("-p", "--photometry", is_flag=True, default=False, help="Show a plot of the photometry data." )
def sninfo(sn_name: str, photometry: Optional[bool] = False):
    try:
        sn_obj = OSCFactory.OSC2018June().load_datapoint(name=sn_name)
        click.echo(sn_obj)
        if photometry:
            if not sn_obj.photometry:
                click.echo("Warning: Object has no photometry.")
            else:
                PlotPhotometry(photometry=sn_obj.photometry, title=sn_obj.name).show(1200, 600)
    except DataPointNotFoundError:
        click.echo(message=f"Could not find supernova candidate with name `{sn_name}`", err=True)
