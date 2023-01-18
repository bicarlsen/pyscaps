# scaps/iv

import re

import numpy as np
import pandas as pd
from bric_analysis_libraries import standard_functions as stdfn

from . import common


def content_to_data_sections( content ):
    """
    :returns: List of data section each as a list of lines.
    """
    lines = content.split( '\n' )
    
    data_break_pattern = 'SCAPS [\d+\.]+'
    data_breaks = []
    for index, line in enumerate( lines ):
        if re.match( data_break_pattern, line ):
            data_breaks.append( index )

    data_breaks.append( len( lines ) )
    
    data_sections = [ 
        lines[ data_breaks[ i ] : data_breaks[ i + 1 ] - 1 ]
        for i in range( len( data_breaks ) - 1 )
    ]
    
    return data_sections


def section_positions( section ):
    """
    :returns: Dictionary of line numbers for section components.
    """
    parameters_pattern = '**Batch parameters**'
    try:
        params_start = section.index( parameters_pattern ) + 1
    
    except ValueError as err:
        raise RuntimeError( 'Could not find batch parameters.' )
        
    params_end = section.index( '', params_start )
    
    header = params_end + 1 
    data_start = params_end + 3
    data_end = section.index( '', data_start )
    
    cp_pattern = 'solar cell parameters deduced from calculated IV-curve:'
    cp_start = section.index( cp_pattern )
    cp_end = section.index( '', cp_start )
    
    return {
        'params': ( params_start, params_end ),
        'header': header,
        'data':   ( data_start, data_end ),
        'cell parameters': ( cp_start, cp_end )
    }
    

def section_parameters( section ):
    """
    :returns: List of parameters from a data section.
    """
    pos = section_positions( section )[ 'params' ]
    params = [
        param.split( ':' )
        for param in section[ pos[ 0 ] : pos[ 1 ] ]
    ]
    
    params = [
        [ param[ 0 ].split( '>>' ), float( param[ 1 ] ) ]
        for param in params
    ]
    
    return params


def section_data( section, remove_header_units = True ):
    """
    :param remove_header_units: Whether to remove header units or not.
        [Default: True]
    :returns: Pandas DataFrame representing the section.
    """
    pos = section_positions( section )
    # get data
    header = [ h.strip() for h in section[ pos[ 'header' ] ].split( '\t' ) ]
    v_index = 'v(V)'
    if remove_header_units:
        header = common.remove_header_units( header )
        v_index = 'v'
        
    data = [ 
        [ float( v ) for v in d.split( '\t' ) ] 
        for d in section[ pos[ 'data' ][ 0 ] : pos[ 'data' ][ 1 ] ] 
    ]

    df = pd.DataFrame( data, columns = header )
    df = df.set_index( v_index )
    df.columns = df.columns.rename( 'metrics' )
    
    # get parameters
    params = section_parameters( section )
    p_names = [ tuple( p[ 0 ] ) for p in params ]
    p_vals =  [ p[ 1 ] for p in params ]
    df = stdfn.insert_index_levels( df, p_vals, names = p_names )

    return df


def section_cell_parameters( section, remove_header_units = True ):
    """
    """
    pos = section_positions( section )[ 'cell parameters' ]
    param_pattern = '(.+)=\s*(\S+)\s+(.+)'  # <name> = <value> <unit>
    params = {}
    for line in section[ pos[ 0 ]: pos[ 1 ] ]:
        m = re.match( param_pattern, line )
        if m is None:
            continue
            
        name = m.group( 1 ).strip()
        val  = float( m.group( 2 ).strip() )
        unit = m.group( 3 ).strip()
        
        if not remove_header_units:
            name = f'{name} ({unit})'
        
        params[ name ] = [ val ]
    
    return params


def import_batch_iv_data( file, encoding = 'iso-8859-1', **kwargs ):
    """
    Imports data from a batch calculation as a Pandas DataFrame.

    :param file: File of IV data.
    :param encoding: File encoding. [Default: iso-8859-1]
    :param **kwargs: Keyword arguments passed to #section_data.
    :returns: Pandas DataFrame of IV data.
    """
    with open( file, encoding = encoding ) as f:
        content = f.read()
        
    df = []
    for section in content_to_data_sections( content ):
        tdf = section_data( section, **kwargs )
        df.append( tdf )

    df = stdfn.common_reindex( df, fillna = np.nan )
    df = pd.concat( df, axis = 1 ).sort_index( axis = 1 )
    return df


def import_batch_cell_parameters( file, encoding = 'iso-8859-1', **kwargs ):
    """
    Imports cell parameters from a batch caluclation as a Pandas DataFrame.

    :param file: File of IV data.
    :param encoding: File encoding. [Default: iso-8859-1]
    :returns: Pandas DataFrame of cell parameters.
    """
    with open( file, encoding = encoding ) as f:
        content = f.read()
        
    df = []
    for section in content_to_data_sections( content ):
        cp_params = section_cell_parameters( section, **kwargs )
        
        params = section_parameters( section )
        p_names = [ tuple( p[ 0 ] ) for p in params ]
        p_vals =  tuple( p[ 1 ] for p in params )

        tdf = pd.DataFrame( cp_params )
        tdf.index = pd.MultiIndex.from_tuples(
            ( p_vals, ), 
            names = p_names 
        )
    
        df.append( tdf )

    df = pd.concat( df, axis = 0 ).sort_index( axis = 0 )
    return( df )


def import_iv_data( file, encoding = 'iso-8859-1', separator = '\t', remove_header_units = True ):
    """
    Imports data from a single shot calculation as a Pandas DataFrame.
    
    :param file: File of IV data.
    :param encoding: File encoding. [Default: iso-8859-1]
    :param separator: Data separator. [Default: '\t']
    :param remove_header_units: Remove units from parameter names.
        [Default: True]
    :returns: Pandas DataFrame of IV data.
    """
    with open( file, encoding = encoding ) as f:
        content = f.read()

    data_pattern = '\n\n +(v\(V\).+\n\n(?:.+\n)+)\n'
    match = re.search( data_pattern, content )
    
    if match is None:
        raise RuntimeError( 'Could not find IV data.' )
    
    data = match.group( 1 )
    data = data.split( '\n' )
    data = [ line for line in data if len( line ) ]  # remove empty lines

    header = [ x.strip() for x in data[ 0 ].split( separator ) ]
    if remove_header_units:
        header = common.remove_header_units( header )

    df = []
    for line in data[ 1: ]:
        df.append( [ float( x ) for x in line.split( separator ) ] )
        
    df = pd.DataFrame( data = df, columns = header )
    df = df.set_index( 'v(V)' if not remove_header_units else 'v' )
    return df


def import_cell_parameters( file, encoding = 'iso-8859-1', remove_header_units = True ):
    """
    Imports cell parameters from a single shot calculation as a Pandas Series.
    
    :param file: File of IV data.
    :param encoding: File encoding. [Default: iso-8859-1]
    :param remove_header_units: Remove units from parameter names.
        [Default: True]
    :returns: Pandas Series of cell parameters.
    """
    with open( file, encoding = encoding ) as f:
        content = f.read()
        
    # get cell parameter section
    section_pattern = 'solar cell parameters deduced from calculated IV-curve:\n((?:.+\n)+)\n'
    match = re.search( section_pattern, content )
    if match is None:
        raise RuntimeError( 'Could not find cell parameter data.' )
        
    # parse cell parameter data
    section = match.group( 1 ).split( '\n' )
    data_pattern = '(\w+)\s*=\s*([\d|\.]+)\s*(.+)'
    data = []
    index = []
    for line in section:
        match = re.search( data_pattern, line )
        if match is None:
            # match not found
            continue
        
        data.append( float( match.group( 2 ) ) )
        
        header = match.group( 1 )
        if not remove_header_units:
            header += f' ({match.group( 3 )})'

        index.append( header )
        
    df = pd.Series( data = data, index = index )
    return df


def import_batch_characterization_parameters( file, encoding = 'iso-8859-1', remove_header_units = True ):
    """
    Import characterization parameters of a batch experiment.

    :param file: File of IV data.
    :param encoding: file encoding. [Default: 'iso-8859-1']
    :param remove_header_units: Remove units from parameter names.
        [Default: True]
    :returns: Pandas DataFrame of characterization parameters indexed by batch parameters.
    """
    with open( file, encoding = encoding ) as f:
        h_data, v_data = common.split_content_header_data( f.read() )

    # create data frame
    rows = v_data.strip().split( '\n' )     
    df = [ list( map( lambda v: float( v.strip() ), row.split( '\t' ) ) ) for row in rows[ 1: ] ]
    df = pd.DataFrame( df, columns = rows[ 0 ].split( '\t' ) )
    df = df.drop( 'i', axis = 1 )

    # set index from batch parameters
    bp_pattern = 'bp (\d+)'
    h_names = common.batch_settings( h_data )
    ind = []
    for name, data in df.items():
        match = re.match( bp_pattern, name )
        if match is None:
            continue
            
        bp_name = h_names[ match.group( 1 ) ]
        df = df.rename( { name: bp_name }, axis = 1 )
        
        ind.append( bp_name )

    df = df.set_index( ind )
    
    if remove_header_units:
        df = df.rename( { name: re.sub( '\(.*\)', '', name ) for name in df.columns }, axis = 1 )

    df = df.rename( { name: name.strip() for name in df.columns }, axis = 1 )
    return df