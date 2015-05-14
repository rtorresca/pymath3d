from distutils.core import setup
from distutils.command.install_data import install_data

setup (
    name = 'math3d', 
    version = '3.0.2',
    description = '3D Special Euclidean mathematics package for Python.',
    author = 'Morten Lind',
    author_email = 'morten@lind.dyndns.dk',
    url = 'http://launchpad.net/pymath3d',
    packages = ['math3d', 'math3d.interpolation', 'math3d.reference_system', 'math3d.dynamics', 'math3d.geometry'],
    provides = ['math3d'],
    license = 'GNU General Public License v3'
)
