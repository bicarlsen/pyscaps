# scaps/eb

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from bric_analysis_libraries import standard_functions as stdfn

from . import common


def import_eb_data(
    file,
    separator = '\t',
    remove_header_units = True,
    encoding = 'iso-8859-1'
):
    """
    Import an energy band data from SCAPS.

    :param file: Path to the file, usually a .eb file.
    :param separator: Data separator. [Default: '\t']
    :param encoding: File encoding. [Default: iso-8859-1]
    :returns: DataFrame representing the data. 
    """
    with open( file, encoding = 'iso-8859-1' ) as f:
        content = f.read()
        
    section_pattern = 'bulk\n\n(.+\n)\n((?:.+\n)+)\n' 
    match = re.search( section_pattern, content )
    if match is None:
        raise RuntimeError( 'Could not find data.' )
        
    header = [ h.strip() for h in match.group( 1 ).split( separator ) ]
    if remove_header_units:
        header = common.remove_header_units( header )

    data = [ datum for datum in match.group( 2 ).split( '\n' ) if len( datum ) ]
    data = [ [ float( d ) for d in datum.split( separator ) ] for datum in data ]

    df = pd.DataFrame( data = data, columns = header )
    df = df.set_index( 'x' if remove_header_units else 'x(um)' )
    df = df.drop( 'i', axis = 1 )  # remove mesh point index
    return df


def import_interface_data( file, separator = '\t', remove_header_units = True, encoding = 'iso-8859-1' ):
    """
    Import an interface data from SCAPS.

    :param file: Path to the file, usually a .eb file.
    :param separator: Data separator. [Default: '\t']
    :param encoding: File encoding. [Default: iso-8859-1]
    :returns: DataFrame representing the interface. 
    """
    with open( file, encoding = 'iso-8859-1' ) as f:
        content = f.read()
        
    section_pattern = 'interface\n\n(.+\n)\n((?:.+\n)+)\n?'
    match = re.search( section_pattern, content )
    if match is None:
        raise RuntimeError( 'Could not find data.' )
        
    header = [ h.strip() for h in match.group( 1 ).split( separator ) ]
    if remove_header_units:
        header = common.remove_header_units( header )

    data = [ datum for datum in match.group( 2 ).split( '\n' ) if len( datum ) ]
    data = [ [ float( d ) for d in datum.split( separator ) ] for datum in data ]

    df = pd.DataFrame( data = data, columns = header )
    df = df.set_index( 'IF' )
    return df


def import_batch_eb_data( file, separator = '\t', encoding = 'iso-8859-1' ):
    """
    Import an energy band file from SCAPS.

    :param file: Path to the file, usually a .eb file.
    :param separator: Data separator. [Default: '\t']
    :param encoding: File encoding. [Default: iso-8859-1]
    :returns: DataFrame representing the data. 
    """
    with open( file, encoding = 'iso-8859-1' ) as f:
        content = f.read()
        h_content, d_content = common.split_content_header_data( content )

    m_order = metrics_order( h_content )
    h_names = common.batch_settings( h_content )

    mesh = mesh_from_content( d_content, names = h_names, separator = separator )
    mesh = stdfn.insert_index_levels(
        mesh, 'x', names = 'metrics', key_level = mesh.columns.shape[ 0 ]
    )

    df = [ mesh ]
    sections = metric_sections( d_content, names = h_names )
    for param, data in sections.items():
        tdf = data[ 2 ]
        tdf = stdfn.insert_index_levels(
            tdf, data[ 0 ], names = 'metrics', key_level = tdf.columns.shape[ 0 ]
        )

        df.append( tdf )
    
    df = pd.concat( df, axis = 1 ).sort_index( axis = 1 )
    return df
    

def metrics_order( content ):
    """
    Get order of metrics from content.

    :param content: Content to search.
    :returns: List of metrics in order.
    :raises RuntimeError: If the metrics section could not be found.
    """
    metrics_pattern = '\*{3} Overview of recorder settings \*{3}\n\n(.+?)\n\n'
    match = re.search( metrics_pattern, content, re.DOTALL )
    
    if match is None:
        raise RuntimeError( 'Metrics order could not be found.' )
        
    metrics = match.group( 1 ).split( '\n' )
    return metrics


def split_section_header_data( content ):
    """
    Splits a section into its header and data components.

    :param content: Section content.
    :returns: Tuple of ( header, data )
    """
    i_content = tuple( enumerate( content ) )
    
    header_lines = [ 
        index 
        for index, line in i_content 
        if line.startswith( 'bp' )
    ]
    
    headers = [ l for index, l in i_content if index in header_lines ]
    data = [ l for index, l in i_content if index not in header_lines ]
    
    return ( headers, data )


def headers_to_index( headers, names = None, separator = '\t' ):
    """
    Converts headers into a Pandas Index or MultiIndex.

    :param headers: List of headers.
    :param names: Level names to use for the index. [Default: None]
    :param separator: Data separator. [Default: '\t']
    :returns: Pandas Index or MultiIndex.
    """
    header_pattern = 'bp (\d+):(.+)'
    headers = [ re.match( header_pattern, h ) for h in headers ]
    headers = { 
        h.group( 1 ): map( float, h.group( 2 ).strip().split( separator ) )
        for h in headers 
        if h is not None 
    }
    
    if len( headers ) == 0:
        raise RuntimeError( 'No valid headers.' )
        
    elif len( headers ) == 1:
        for name, vals in headers.items():
            index = pd.Index( vals, name = name )

    else:
        values = []
        o_names = []
        for index, vals in headers.items():
            values.append( vals )
            o_names.append( index )
            
        values = tuple( zip( *values ) )
        index = pd.MultiIndex.from_tuples( values, names = o_names )
        
    if names is not None:
        rename = tuple( names[ h_name ] for h_name in index.names )
        index = index.rename( rename )
    
    return index


def section_to_dataframe( section, separator = '\t', names = None ):
    """
    Converts a section into a DataFrame.

    :param section: Section to convert.
    :param separator: Data separator. [Default: '\t']
    :param names: Names to use for column levels. [Default: None]
    :return: DataFrame representing the section data.
    """
    if not isinstance( section, list ):
        section = section.split( '\n' )

    headers, data = split_section_header_data( section )
    cols = headers_to_index( headers, names = names, separator = separator )
    
    data = [ map( float, d.split( separator ) ) for d in data ]
    df = pd.DataFrame( data )
    df = df.set_index( 0 )
    df.index = df.index.astype( np.int64 )
    df.columns = cols
    
    return df


def mesh_from_content( content, names = None, section_break = '\n{4}', separator = '\t' ):
    """
    Creates a DataFrame representing the simulation mesh.

    :param content: Content to parse.
    :param names: Names to use for column levels. [Default: None]
    :param section_break: RegEx representing the section break. [Default: '\n{5}']
    :returns: DataFrame representing the mesh.
    """
    mesh_pattern = '\* The mesh.+?\*(.+?)' + section_break
    match = re.search( mesh_pattern, content, re.DOTALL )
    if match is None:
        raise RuntimeError( 'Mesh could not be found.' )
        
    mesh = match.group( 1 ).strip().split( '\n' )
    mesh = section_to_dataframe( mesh, names = names, separator = separator )
    return mesh
    
    
def metric_sections( content, names = None, section_break = '\n{4}', separator = '\t' ):
    """
    Creates DataFrames representing the simulation parameters.

    :param content: Content to parse.
    :param names: Names to use for column levels. [Default: None]
    :param section_break: RegEx representing the section break. [Default: '\n{5}']
    :returns: Dictionary of tuples of the form
        { index: ( name, code, DataFrame ) } representing the sections.
    """
    section_pattern = '\* Parameter (\d+):\s+(.+?)\*\n\*(.+?)\*\n\n(.+?)' + section_break
    matches = re.findall( section_pattern, content, re.DOTALL )
    sections = {}
    for match in matches:
        param = int( match[ 0 ].strip() )
        sections[ param ] = (
            match[ 1 ].strip(),
            match[ 2 ].strip(),
            section_to_dataframe( match[ 3 ], names = names, separator = separator )
        )
    
    return sections


def plot_band_diagram( edf, model = None, ax = None, **kwargs ):
    """
    Plot the energy band diagram.

    :param edf: Energy band DataFrame.
    :param model: A scaps.Model, used to color the background of each layer.
        [Default: None]
    :param ax: matplotlib.pyplot.Axes to plot on. If None a new one will be created.
        [Default: None]
    :param **kwargs: Additional plotting arguments.
    :returns: Tuple of ( ax, fig ) on which the diagram is plotted.
    """
    # data setup
    htl, pvk, etl = model.layers
    edf = edf[[ 'Ec', 'Fn', 'Fp', 'Ev' ]]

    # original 0 energy is leftmost hole quasi-Fermi level
    # off set to left most perovskite valence band 
    edf_idx = edf.index.values
    t_htl = htl.thickness.value* 1e6
    pvk_start_x = edf_idx[ edf_idx > t_htl ].min()

    e_offset = edf.loc[ pvk_start_x, 'Ev' ].mean()

    edf -= e_offset
    edf.index *= 1e3

    # plotting
    if ax is None:
        fig, ax = plt.subplots()

    # energy bands
    edf.plot(
        ax = ax,
        color = ( 'C0', '#444', '#444', 'C3' ),
        **kwargs
    )

    if model is None:
        return ( ax.figure, ax )

    # model background
    x_scale = 1e9
    y0 = edf.min().min()
    h = edf.max().max() - y0
    bg_alpha = 0.25

    # htl
    x = 0
    w = htl.thickness.value* x_scale
    r_htl = patches.Rectangle(
        ( x, y0 ), w, h,
        facecolor = 'C3', alpha = bg_alpha,
        zorder = 0
    )
    ax.add_patch( r_htl )

    # perovskite
    x += w
    w = pvk.thickness.value* x_scale
    r_pvk = patches.Rectangle(
        ( x, y0 ), w, h,
        facecolor = '#444', alpha = bg_alpha,
        zorder = 0
    )
    ax.add_patch( r_pvk )

    # etl
    x += w
    w = etl.thickness.value* x_scale
    r_etl = patches.Rectangle(
        ( x, y0 ), w, h,
        facecolor = 'C0', alpha = bg_alpha,
        zorder = 0
    )
    ax.add_patch( r_etl )

    ax.legend( [
        'E$_\mathregular{c}$',
        'F$_\mathregular{n}$',
        'F$_\mathregular{p}$',
        'E$_\mathregular{v}$'
    ] )
    ax.set_xlabel( 'x / nm' )
    ax.set_ylabel( 'Energy / eV' )

    return ( ax.figure, ax )