from distutils.core import setup
from distutils.command.install_data import install_data

setup (
    name = 'math3d', 
    version = '3.0.3',
    description = '3D Special Euclidean mathematics package for Python.',
    author = 'Morten Lind',
    author_email = 'morten@lind.dyndns.dk',
    url = 'http://git.automatics.dyndns.dk/?p=pymath3d.git',
    packages = ['math3d', 'math3d.interpolation', 'math3d.reference_system', 'math3d.dynamics', 'math3d.geometry'],
    provides = ['math3d'],
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
    ],
    license = 'GNU General Public License v3'
)
