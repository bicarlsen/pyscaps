# scaps/model

from __future__ import annotations  # required for type hints

# check python version
# required for dataclasses
import sys
if sys.version_info < ( 3, 7 ):
	raise RuntimeError( 'pyscaps.model requires Python version 3.7 or higher.' )

import re
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass, field
import typing


def get_problem_definition_from_file( file, comment = '>' ):
	"""
	:param file: Path to file containing a SCAPS problem definition.
		Usually as .scaps file.
	:param comment: Comment character. [Default: '>']
	:return: Content representing the problem definition.
	"""
	start_pattern = '>> begin of problem definition file'
	end_pattern = '>> end of problem definition file'
	
	in_def = False
	def_read = False
	content = []
	with open( file ) as f:
		for line in f:
			line = line.strip()

			if line == start_pattern:
				# check if we have enetered the problem definition
				if def_read:
					# problem definition already encountered
					raise RuntimeError( 'Multiple problem definitions encountered.' )

				in_def = True

			if not in_def:
				continue

			if line == end_pattern:
				in_def = False
				def_read = True

			if line.startswith( comment ):
				# line starts with comment, skip
				continue

			if not line:
				# skip blank lines
				continue

			content.append( line )

	return content


def split_problem_definition( content ):
	"""
	Splits a problem definition into its components.

	:param content: Problem definition content in the form of a list of strings.
	:returns: List of problem definition sections as tuples of ( type, content ).
	"""
	section_keywords = [
		'convergence',
		'layer',
		'interface properties',
		'front contact',
		'back contact'
	]

	if content[ 0 ] not in section_keywords:
		raise ValueError( 'Content does not begin with a know section type.' )

	sections = []
	section_header = None
	section_content = None

	for line in content:
		if line in section_keywords:
			# save previous section
			sections.append( ( section_header, section_content ) )

			# set new section
			section_header = line
			section_content = []

		else:
			section_content.append( line )

	# save final section
	sections.append( ( section_header, section_content ) )

	# remove initial empty section
	return sections[ 1: ]


def split_section_defects( content, defect_delimeter ):
	"""
	:param content: List of content strings.
	:defect_delimeter: String delimiting defects.
	:returns: Tuple of ( properties content, [ defects content ] ).
	"""
	defect_indices = [
		i 
		for i, line in enumerate( content ) 
		if line == defect_delimeter
	]

	properties = (
		content[ :defect_indices[ 0 ] ]
		if len( defect_indices ) else
		content
	)

	defects = []
	if len( defect_indices ):
		defect_indices.append( None )
		for i_start, i_end in zip( defect_indices[ :-1 ], defect_indices[ 1: ] ):
			defects.append( content[ i_start + 1: i_end ] )

	return ( properties, defects )


def content_to_param_dict(
	content,
	param_map,
	delimeter = ':',
	list_delimeter = '\t',
	unit_marker = ( '\\[', '\\]' ),
	comment_markers = [ ( '\\(', '\\)' ), ';' ],
	ignore_errors = True
):
	"""
	Converts content into the correct type.

	:param content: List of content lines.
	:param map: Map of content keys to tuples of ( param key, data type[, sub data type] ).
		Content keys that are not included in the map are ignored.
	:param delimeter: Key value delimeter in content. [Default: ':']
	:param list_delimeter: Delimeter for list values. [Default: '\t']
	:param unit_marker: Tuple of start and end markers which enclose units.
		If markers contain special regex characters they must be escaped.
		[Default: ( '\\[', '\\]' )]
	:param comment_marker: List of comment markers.
		Tuple can be used for start and end markers which enclose comments.
		If markers contain special regex characters they must be escaped.
		[Default: [ ( '\\(', '\\)' ), ';' ]]
	:param ignore_errors: Ignore lines that can not be parsed as params.
		[Default: True]
	:returns: Dictionary of { key: value }.
		If value is numeric value is a tuple of ( val, unit ).
	"""

	def _parse_value( val, kind ):
		"""
		:param val: Value to parse.
		:param kind: Class of desired value.
		:returns: val parsed as kind
		"""
		if kind is bool:
			# true/false values are represented as '0', '1'
			# so must first be converted to int then bool.
			return bool( int( val ) )

		elif issubclass( kind, Enum ):
			try:
				# try to cast value to int before enum
				val = int( val )

			except:
				# if not a valid int, assume it is a string enum
				pass

			return kind( val )

		else:
			return kind( val )


	unit_pattern = f'{unit_marker[ 0 ]}(.*?){unit_marker[ 1 ]}'
	
	params = {}
	for line in content:
		try:
			c_key, val = tuple( s.strip() for s in line.split( delimeter, 1 ) )

		except ValueError as err:
			if not ignore_errors:
				raise ValueError( f'Could not parse "{line}" as parameter.' )

		if c_key in param_map:
			p_map = param_map[ c_key ]
			p_key = p_map[ 0 ]
			
			if len( p_map ) == 2:
				kind = p_map[ 1 ]
				d_kind = None

			elif len( p_map ) == 3:
				kind, d_kind = p_map[ 1: ]

			else:
				raise ValueError( f'Invalid parameter map for "{c_key}".' )

			# remove comments
			for comment_marker in comment_markers:
				comment_pattern = (
					f'{comment_marker[ 0 ]}.*{comment_marker[ 1 ]}'
					if isinstance( comment_marker, tuple ) else
					f'{comment_marker}.*$'
				)
				
				comment_match = re.search( comment_pattern, val )
				if comment_match is not None:
					# remove comment
					val = val.replace( comment_match.group( 0 ), '' ).strip()

			# get unit
			unit_match = re.search( unit_pattern, val )
			if unit_match is not None:
				# remove unit
				val = val.replace( unit_match.group( 0 ), '' ).strip()
				unit = unit_match.group( 1 )

			else:
				unit = None

			# parse value
			if kind is list:
				val = [ _parse_value( v.strip(), d_kind ) for v in val.split( list_delimeter ) ]

			else:
				val = _parse_value( val, kind )

			# assign value
			if ( kind is float ) or ( d_kind is float ):
				params[ p_key ] = ValueParameter( val, unit )

			else:
				params[ p_key ] = val

	return params



class InterfaceDefectEnergyRef( Enum ):
	"""
	Energy reference for interface defects.
	"""
	above_vb = 3
	below_cb = 4
	above_ei_left  = 5
	above_ei_right = 6
	above_vb_left  = 7
	above_vb_right = 8
	above_ei_middle = 9


class BulkDefectEnergyRef( Enum ):
	"""
	Energy reference for bulk defects.
	"""
	above_vb = 0
	above_ei = 1
	below_cb = 2


class DefectType( Enum ):
	"""
	Types of defects.
	"""
	neutral = 'neutral'


class GradingProfile( Enum ):
	"""
	Grading profiles.
	"""
	homogenous = 'homogenous'
	uniform = 'uniform'


class EnergyDistribution( Enum ):
	"""
	Energy distribution profiles.
	"""
	single = 'single'
	gauss = 'gauss'


class ModelType( Enum ):
	"""
	Layer types in the model.
	"""
	layer = 'layer',
	interface = 'interface properties'


ValueParameter = namedtuple( 'ValueParameter', [ 'value', 'unit' ] )


@dataclass
class Contact():
	"""
	Represents a contact in a SCAPS model.
	"""
	work_function: ValueParameter
	flatband: bool
	recalculate: bool

	Sn: ValueParameter = ValueParameter( 0, 'm/s' )
	Sp: ValueParameter = ValueParameter( 0, 'm/s' )

	tunneling_to_contact: bool = False
	e_rel_mass: ValueParameter = ValueParameter( 1, None )
	h_rel_mass: ValueParameter = ValueParameter( 1, None )
	wavelength_independent: bool = True


	@staticmethod
	def from_content( content ) -> Contact:
		"""
		:param content: List of strings representing the contact.
		:returns: Contact representing the content.		
		"""
		param_map = {
			'flatband': ( 'flatband', bool ),
			'recalculate': ( 'recalculate', bool ),
			'Fi_m': ( 'work_function', float ),
			'Sn': ( 'Sn', float ),
			'Sp': ( 'Sp', float ),
			'Tunneling to contact': ( 'tunneling_to_contact', bool ),
			'Relative electron mass': ( 'e_rel_mass', float ),
			'Relative hole mass': ( 'h_rel_mass', float ),
			'wavelength independent': ( 'wavelength_independent', bool )
		}

		params = content_to_param_dict( content, param_map )
		return Contact( **params )
		

@dataclass
class Defect():
	"""
	Represents a generic defect.
	"""
	kind: DefectType
	density: float
	energy: float

	energy_distribution: EnergyDistribution
	energy_reference: typing.Union[ InterfaceDefectEnergyRef, BulkDefectEnergyRef ]


@dataclass
class BulkDefect( Defect ):
	"""
	Represents a bulk defect in a SCAPS model.
	"""
	e_capture_xs: float
	h_capture_xs: float

	grading_profile: GradingProfile = GradingProfile.homogenous

	e_optical_capture: bool = False
	h_optical_capture: bool = False

	e_rel_mass: float = 1
	h_rel_mass: float = 1

	e_refractive_index:float = 1
	h_refractive_index: float = 1

	e_effective_field_ratio: float = 1
	h_effective_field_ratio: float = 1

	e_cutoff_energy: float = 10
	h_cutoff_energy: float = 10

	characteristic_energy: float = 0


	@staticmethod
	def from_content( content ) -> BulkDefect:
		"""
		:param content: List of strings representing the defect.
		:returns: BulkDefect representing the content.		
		"""
		param_map = {
			'type': ( 'kind', DefectType ),
			'Nt(uniform)': ( 'density', list, float ),
			'Et': ( 'energy', float ),
			'energy distribution': ( 'energy_distribution', EnergyDistribution ),
			'Reference for defect energy': ( 'energy_reference', BulkDefectEnergyRef ),
			'sigma_n': ( 'e_capture_xs', float ),
			'sigma_p': ( 'h_capture_xs', float ),
			'profile': ( 'grading_profile', GradingProfile ),
			'Ekar': ( 'characteristic_energy', float )
		}

		params = content_to_param_dict( content, param_map )
		return BulkDefect( **params )


@dataclass
class InterfaceDefect( Defect ):
	"""
	Represents an interface defect in a SCAPS model.
	"""
	e_capture_xs_left: float
	e_capture_xs_right: float
	h_capture_xs_left: float
	h_capture_xs_right: float

	trap_tunneling: bool = False

	e_rel_mass: float = 1
	h_rel_mass: float = 1

	characteristic_energy: float = 0

	@staticmethod
	def from_content( content ) -> InterfaceDefect:
		"""
		:param content: List of strings representing the defect.
		:returns: BulkDefect representing the content.		
		"""
		param_map = {
			'type': ( 'kind', DefectType ),
			'N': ( 'density', float ), 
			'Et': ( 'energy', float ),
			'energy distribution': ( 'energy_distribution', EnergyDistribution ),
			'Reference for defect energy': ( 'energy_reference', InterfaceDefectEnergyRef ),
			'sigma_nleft': ( 'e_capture_xs_left', float ),
			'sigma_nright': ( 'e_capture_xs_right', float ),
			'sigma_pleft': ( 'h_capture_xs_left', float ),
			'sigma_pright': ( 'h_capture_xs_right', float ),
			'Tunneling to trap': ( 'trap_tunneling', bool ),
			'Relative electron mass': ( 'e_rel_mass', float ),
			'Relative hole mass': ( 'h_rel_mass', float ),
			'Ekar': ( 'characteristic_energy', float )
		}

		params = content_to_param_dict( content, param_map )
		return InterfaceDefect( **params )


@dataclass
class Layer():
	"""
	Represents a layer in a SCAPS model.
	"""
	name: str
	thickness: float
	electron_affinity: float
	band_gap: float
	relative_permittivity: float

	conduction_dos: float
	valence_dos: float
	
	e_mobility: float
	h_mobility: float
	
	e_thermal_velocity: float
	h_thermal_velocity: float
	
	donor_concentration: float
	acceptor_concentration: float

	radiative_recombination: float
	e_auger_capture: float = 0
	h_auger_capture: float = 0
	
	e_rel_mass: float = 1
	h_rel_mass: float = 1
	tunneling: bool = False

	defects: typing.List[ BulkDefect ] = field( default_factory = list )


	@property
	def work_function( self ) -> float:
		if self.electron_affinity.unit != self.band_gap.unit:
			raise ValueError( 'Electron Affinity and Band Gap have different units. Can not compute.' )

		ea = self.electron_affinity.value
		bg = self.band_gap.value

		ea_list = isinstance( ea, list )
		bg_list = isinstance( bg, list )
		if ea_list and bg_list:
			if len( ea ) != len( bg ):
				raise ValueError( 'Electron Affinity and Band Gap have different lengths. Can not compute.' )

			wf = [ ( ea[ i ] + bg[ i ] ) for i in range( len ( ea ) ) ]

		elif ea_list:
			# ea is list, bg is number
			wf = [ ( ea_val + bg ) for ea_val in ea ]

		elif bg_list:
			# bg is list, ea is number
			wf = [ ( bg_val + ea ) for bg_val in bg ]

		else:
			# bg and ea are single values
			wf = bg + ea

		return ValueParameter( value = wf, unit = self.electron_affinity.unit )


	@staticmethod
	def from_content( content ) -> Layer:
		"""
		:param content: List of strings representing the layer.
		:returns: Layer representing the content.		
		"""
		layer_content, defects_content = split_section_defects( content, 'srhrecombination' )
		
		param_map = {
			'name': ( 'name', str ),
			'd': ( 'thickness', float ),
			'Tunneling in layer': ( 'tunneling', bool ),
			'Relative electron mass': ( 'e_rel_mass', list, float ),
			'Relative hole mass': ( 'h_rel_mass', list, float ),
			'v_th_n': ( 'e_thermal_velocity', list, float ),
			'v_th_p': ( 'h_thermal_velocity', list, float ),
			'eps': ( 'relative_permittivity', list, float ),
			'chi': ( 'electron_affinity', list, float ),
			'Eg': ( 'band_gap', list, float ),
			'Nc': ( 'conduction_dos', list, float ),
			'Nv': ( 'valence_dos', list, float ),
			'mu_n': ( 'e_mobility', list, float ),
			'mu_p': ( 'h_mobility', list, float ),
			'K_rad': ( 'radiative_recombination', list, float ),
			'c_n_auger': ( 'e_auger_capture', list, float ),
			'c_p_auger': ( 'h_auger_capture', list, float ),
			'Na(uniform)': ( 'acceptor_concentration', list, float ),
			'Nd(uniform)': ( 'donor_concentration', list, float )
		}

		params = content_to_param_dict( layer_content, param_map )
		layer = Layer( **params )

		for defect_content in defects_content:
			defect = BulkDefect.from_content( defect_content )
			layer.defects.append( defect )

		return layer


@dataclass
class Interface():
	"""
	Represents an interface in a SCAPS model.
	"""
	name: str

	e_rel_mass: float = 1
	h_rel_mass: float = 1

	intraband_tunneling: bool = False

	defects: typing.List[InterfaceDefect] = field( default_factory = list )


	@staticmethod
	def from_content( content ):
		"""
		:param content: List of strings representing the interface.
		:returns: Interface representing the content.		
		"""
		interface_content, defects_content = split_section_defects( content, 'interface recombination' )

		param_map = {
			'interfacename': ( 'name', str ),
			'Relative electron mass': ( 'e_rel_mass', float ),
			'Relative hole mass': ( 'h_rel_mass', float ),
			'intraband tunneling': ( 'intraband_tunneling', bool )
		}

		interface_params = content_to_param_dict( interface_content, param_map )
		interface = Interface( **interface_params )

		for defect_content in defects_content:
			defect = InterfaceDefect.from_content( defect_content )
			interface.defects.append( defect )

		return interface


class Model():
	"""
	Represents a SCAPS model.
	"""

	def __init__( self, stack = [], front_start = True ):
		"""
		:param stack: Model stack as a list. [Default: []]
		:param front_start: True if the start of the stack is the front,
			False if the end of the stack is the front.
		"""
		self.stack = stack
		self.front_start = front_start


	@property
	def front_contact( self ):
		"""
		:returns: Front contact if it exists, or None.
		"""
		front_section = (
			self.stack[ 0 ]
			if self.front_start else
			self.stack[ -1 ]
		)

		return (
			front_section
			if isinstance( front_section, Contact ) else
			None
		)


	@property
	def back_contact( self ):
		"""
		:returns: Back contact if it exists, or None.
		"""
		back_section = (
			self.stack[ -1 ]
			if self.front_start else
			self.stack[ 0 ]
		)

		return (
			back_section
			if isinstance( back_section, Contact ) else
			None
		)


	@property
	def contacts( self ):
		"""
		:returns: List of model contacts.
		"""
		return self.sections_of_type( Contact )


	@property
	def layers( self ):
		"""
		:returns: List of model layers.
		"""
		return self.sections_of_type( Layer )


	@property
	def interfaces( self ):
		"""
		:returns: List of model interfaces.
		"""
		return self.sections_of_type( Interface )

	
	@property
	def total_thickness( self ):
		"""
		:returns: Total thickness of all layers in the model.
		"""
		units = [ layer.thickness.unit for layer in self.layers ]
		if any( [ unit != units[ 0 ] for unit in units ] ):
			raise ValueError( 'Layer thicknesses have different units. Can not compute.' )

		t = sum( [ layer.thickness.value for layer in self.layers ] )
		return ValueParameter( value = t, unit = units[ 0 ] )


	def sections_of_type( self, kind ):
		"""
		:param kind: Type of section.
		:returns: List of model sections with given kind.
		"""
		sections = []
		for sec in self.stack:
			if isinstance( sec, kind ):
				sections.append( sec )

		return sections


	def section( self, name ):
		"""
		:param name: Name of the section to return.
		:return: Section of the stack with the matching name.
		"""
		for sec in self.stack:
			if hasattr( sec, 'name' ):
				if sec.name == name:
					return sec


	def interface( self, l1, l2 ):
		"""
		:param l1: Name of layer 1.
		:param l2: Name of layer 2.
		:returns: Interface between the two layers.
		"""
		interface = self.section( f'{l1} / {l2}' )
		if interface is None:
			# swap layer order
			interface = self.section( f'{l2} / {l1}' )

		return interface
		

	@staticmethod
	def from_file( file ) -> Model:
		"""
		Load a model in form a SCAPS file (usually .scaps).

		:param file: Path to the file to load.
		:returns: Model.
		"""
		model_keywords = {
			'back contact': Contact,
			'front contact': Contact,
			'layer': Layer,
			'interface properties': Interface
		}

		content = get_problem_definition_from_file( file )
		sections = split_problem_definition( content )
		stack = []
		for section_type, section_content in sections:
			if section_type in model_keywords.keys():
				model_kind = model_keywords[ section_type ]
				section_model = model_kind.from_content( section_content )

				stack.append( section_model )

		model = Model( stack )

		# determine if front is start or end
		stack_arch = [ sec[ 0 ] for sec in sections ]
		if ( 'back contact' in stack_arch ) and ( 'front contact' in stack_arch ):
			model.front_start =  ( stack_arch.index( 'front contact' ) < stack_arch.index( 'back contact' ) )
		
		return model
