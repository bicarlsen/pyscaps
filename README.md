# PySCAPS

> python -m pip install pyscaps 

> Note: This library is part of the [`bric_analysis_libraries`](https://pypi.org/project/bric-analysis-libraries/) Python package. It is provided here as an isolated package for convenience.

Python library to analyze [SCAPS-1D](https://scaps.elis.ugent.be/) models and results. 

### Modules

#### Model
> Module name: `model`

Loads model data into an object for analysis.

#### General
> Module name: `gnr`

Formats general data into Pandas DataFrames.

#### JV
> Module name: `iv`

Formats IV data into Pandas DataFrames.

#### Energy Band
> Module name: `eb`

Formats energy band data into Pandas DataFrames.

#### Generation and Recombination
> Module name: `gen`

Formats generation and recombination and data into Pandas DataFrames.

#### Common
> Module name: `common`

Common functions with low level functionality.

## Examples

```python
import pyscaps

# import model into object
model = pyscaps.model.Model.from_file( 'path/to/my_model.scaps' )

# import single shot results into Pandas DataFrame
eb = pyscaps.eb.import_eb_data( 'path/to/my_eb_file.eb' )

# import batch record results into Pandas DataFrame
iv = pyscaps.iv.import_batch_iv_data( 'path/to/my_batch_iv_file.iv' )
cp = pyscaps.iv.import_batch_cell_parameters( 'path/to/my_batch_iv_file.iv' )
```
