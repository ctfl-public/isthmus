import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

# Define the extension module
extensions = [
    Extension(
        'Marching_Cubes._marching_cubes_lewiner_cy',  # Adjusted name
        sources=['Marching_Cubes/_marching_cubes_lewiner_cy.pyx'],
        include_dirs=[numpy.get_include()],
        # Add other necessary parameters like libraries, library_dirs, etc.
    )
]

setup(
    name='MyProject',
    packages=['Marching_Cubes'],  # Make sure to declare your package
    ext_modules=cythonize(extensions)
)
