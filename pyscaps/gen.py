# scaps/gen

import re

import numpy as np
import pandas as pd

from . import common 


def import_gen_data(
    file,
    remove_header_units = True,
    separator = '\t',
    encoding = 'iso-8859-1'
):
    """
    Import a generation-recombination data file from SCAPS.

    :param file: Path to the data file. Usually a .gen file.
    :param remove_header_units: Remove units from headers. [Default: True]
    :param separator: Data separator. [Default: '\t']
    :returns: DataFrame representing the data.
    """
    with open( file, encoding = encoding ) as f:
        content = f.read().strip()

    header_pattern = f'\s*x\s*\(.m\)\s*{separator}'
    lines = content.split( '\n' )
    header = None
    for line_no, line in enumerate( lines ):
        match = re.match( header_pattern, line )
        if match is not None:
            header = line
            data_start = line_no + 1
            break

    if header is None:
        raise RuntimeError( 'Could not find data.' )

    header = [ 
        h.strip() 
        for h in line.split( separator ) 
        if h.strip() != ''
    ]

    if remove_header_units:
        header = common.remove_header_units( header )

    data = [
        [
            float( d )
            for d in line.split( separator )
            if d.strip() != ''
        ]
        for line in lines[ data_start: ]
    ]

    df = pd.DataFrame( data = data, columns = header )
    df = df.set_index( header[ 0 ] )
    return df

