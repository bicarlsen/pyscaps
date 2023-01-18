# scaps/gnr

import re

import numpy as np
import pandas as pd

from . import common 


def import_gnr_data( file, separator = '\t' ):
    """
    Import a general data file from SCAPS.

    :param file: Path to the data file. Usually a .gnr file.
    :param separator: Data separator. [Default: '\t']
    :returns: DataFrame representing the data.
    """

    with open( file ) as f:
        h_content, d_content = common.split_content_header_data( f.read() )

    h_names = common.batch_settings( h_content )
    h_names = { f'bp {i}': d for i, d in h_names.items() }
    
    data = [ d.split( separator ) for d in d_content.strip().split( '\n' ) ]
    header = data[ 0 ]
    data = [ map( float, d ) for d in data[ 1: ] ]
    
    df = pd.DataFrame( data, columns = header )
    df = df.set_index( 'i' )
    df = df.set_index( list(h_names.keys() ) )
    df.index = df.index.rename( h_names.values() )
    df = df.rename( { h: h.strip() for h in df.columns }, axis = 1 )
    df = df.sort_index()
    
    return df