import click

from snanomaly.cli.dimreduce import dimreduce
from snanomaly.cli.interpolate import interpolate
from snanomaly.cli.outlier import outlier
from snanomaly.cli.sninfo import sninfo
from snanomaly.cli.validate import validate


@click.group()
def cli():
    pass

cli.add_command(sninfo)
cli.add_command(validate)
cli.add_command(interpolate)
cli.add_command(dimreduce)
cli.add_command(outlier)
