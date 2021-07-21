# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *


class Meep(AutotoolsPackage):
    """Meep (or MEEP) is a free finite-difference time-domain (FDTD) simulation
    software package developed at MIT to model electromagnetic systems."""

    homepage = "http://ab-initio.mit.edu/wiki/index.php/Meep"
    url      = "https://github.com/NanoComp/meep/releases/download/v1.19.0/meep-1.19.0.tar.gz" 
             # "http://ab-initio.mit.edu/meep/meep-1.3.tar.gz"
    list_url = "http://ab-initio.mit.edu/meep/old"

    version('1.19.0', sha256='5a73e719b274017015e9e7994b7a7a11139d4eb6')
    version('1.3',   sha256='564c1ff1b413a3487cf81048a45deabfdac4243a1a37ce743f4fcf0c055fd438')
    version('1.2.1', sha256='f1f0683e5688d231f7dd1863939677148fc27a6744c03510e030c85d6c518ea5')
    version('1.1.1', sha256='7a97b5555da1f9ea2ec6eed5c45bd97bcd6ddbd54bdfc181f46c696dffc169f2')

    variant('harminv', default=True, description='Enable Harminv support')
    variant('guile',   default=True, description='Enable Guile support')
    variant('libgdsii', default=False, description='Enable libGDSII support')

    # depends_on('python')
    depends_on('libctl@4.0:')
    depends_on('mpi')
    depends_on('hdf5+mpi')
    depends_on('blas',        when='+harminv')
    depends_on('lapack',      when='+harminv')
    depends_on('harminv',     when='+harminv')
    depends_on('guile',       when='+guile')
    depends_on('libgdsii',    when='+libgdsii')

    def configure_args(self):
        spec = self.spec

        config_args = [
            '--enable-shared',
            '--with-libctl={0}'.format(join_path(spec['libctl'].prefix.share, 
                                                 'libctl')),
            '--with-mpi',
            '--with-hdf5'
        ]
        
        if '+harminv' in spec:
            config_args.append('--with-blas={0}'.format(
                spec['blas'].prefix.lib))
            config_args.append('--with-lapack={0}'.format(
                spec['lapack'].prefix.lib))
        else:
            config_args.append('--without-blas')
            config_args.append('--without-lapack')

        return config_args

    def check(self):
        spec = self.spec

        # aniso_disp test fails unless installed with harminv
        # near2far test fails unless installed with gsl
        if '+harminv' in spec and '+gsl' in spec:
            # Most tests fail when run in parallel
            # 2D_convergence tests still fails to converge for unknown reasons
            make('check', parallel=False)
