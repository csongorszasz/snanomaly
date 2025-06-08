import click

from snanomaly.cli.dimreduce import dimreduce
from snanomaly.cli.interpolate import interpolate
from snanomaly.cli.sninfo import sninfo


@click.group()
def cli():
    pass

cli.add_command(interpolate)
cli.add_command(sninfo)
cli.add_command(dimreduce)
