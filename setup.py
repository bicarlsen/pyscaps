import setuptools


# get __version__
exec( open( 'pyscaps/_version.py' ).read() )

with open("README.md", "r") as fh:
    long_description = fh.read()


project_urls = {
    'Source Code': 'https://github.com/bicarlsen/pyscaps',
    'Bug Tracker': 'https://github.com/bicarlsen/pyscaps/issues'
}


setuptools.setup(
    name = "pyscaps",
    version = __version__,
    author = "Brian Carlsen",
    author_email = "carlsen.bri@gmail.com",
    description = "Library for analyzing SCAPS-1D models and results.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = [ 'scaps', 'scaps-1d' ],
    url = "",
    project_urls = project_urls,
    packages = setuptools.find_packages(),
    python_requires = '>=3.7',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        "Development Status :: 3 - Alpha"
    ],
    install_requires = [
        'bric_analysis_libraries >= 0.1.1',
        'numpy >= 1.22',
        'pandas >= 1.4'
    ],
    package_data = {}
)
