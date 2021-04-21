# /*
# * Atomic cluster expansion
# *
# * Copyright 2021  (c) Yury Lysogorskiy, Sarath Menon, Ralf Drautz
# *
# * Ruhr-University Bochum, Bochum, Germany
# *
# * See the LICENSE file.
# * This FILENAME is free software: you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation, either version 3 of the License, or
# * (at your option) any later version.
#
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
#     * You should have received a copy of the GNU General Public License
# * along with this program.  If not, see <http://www.gnu.org/licenses/>.
# */


import os
import re
import sys
import sysconfig
import platform
import subprocess
import shutil

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, target=None, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.target = target


# checks for the cmake version
class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        # build_args = ['-DCMAKE_BUILD_TYPE='+cfg]
        build_args = []

        # Pile all .so in one place and use $ORIGIN as RPATH
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]
        cmake_args += ["-DBUILD_SHARED_LIBS=ON"]
        cmake_args += ["-DYAML_BUILD_SHARED_LIBS=ON"]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2 ** 32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j']

        if ext.target is not None:
            build_args += [ext.target]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        print("Building arguments: ", " ".join(build_args))
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


with open('README.md') as readme_file:
    readme = readme_file.read()

if not os.path.exists('bin'):
    os.makedirs('bin')
try:
    shutil.copyfile('bin/pacemaker.py', 'bin/pacemaker')
    shutil.copyfile('bin/pace_yaml2ace.py', 'bin/pace_yaml2ace')
except FileNotFoundError as e:
    print("File not found (skipping):", e)

setup(
    name='pyace-lite',
    version='0.0.1.5',
    author='Yury Lysogorskiy, Anton Bochkarev, Sarath Menon, Ralf Drautz',
    author_email='yury.lysogorskiy@rub.de',
    description='Python bindings, utilities for PACE and fitting code "pacemaker"',
    long_description=readme,
    long_description_content_type='text/markdown',
    # tell setuptools to look for any packages under 'src'
    packages=find_packages('src'),
    # tell setuptools that all packages will be under the 'src' directory
    # and nowhere else
    package_dir={'': 'src'},
    # add an extension module named 'python_cpp_example' to the package
    ext_modules=[CMakeExtension('pyace/sharmonics', target='sharmonics'),
                 CMakeExtension('pyace/coupling', target='coupling'),
                 CMakeExtension('pyace/basis', target='basis'),
                 CMakeExtension('pyace/evaluator', target='evaluator'),
                 CMakeExtension('pyace/catomicenvironment', target='catomicenvironment'),
                 CMakeExtension('pyace/calculator', target='calculator'),
                 ],
    # add custom build_ext command
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    url='https://git.noc.ruhr-uni-bochum.de/atomicclusterexpansion/pyace',
    install_requires=['numpy', 'ase', 'pandas', 'ruamel.yaml'],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    package_data={"pyace.data": ["pyace_selected_bbasis_funcspec.pckl.gzip"]},
    scripts=["bin/pacemaker", "bin/pace_yaml2ace"]
)
