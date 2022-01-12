# PyMeepPlasmonics

Free and open-source code package designed to perform PyMEEP FDTD simulations applied to Plasmonics.

## About

`PyMeepPlasmonics` is a freely accesible FDTD-MEEP implementation that contains code modules and routine scripts that allow for serial and parallel Plasmonics simulations to be run both in personal computers and clusters. 

`PyMeepPlasmonics` allows to perform **Finite-Differences Time-Domain (FDTD) simulations**, where a certain domain is discretized both in space and time in order to simulate the electromagnetic response in time by advancing step-by-step the electromagnetic field evaluated in each and every position or node. It uses a free and open-source FDTD implementation called **MIT Electromagnetic Propagation (MEEP)**. [MEEP](https://meep.readthedocs.io/en/latest/) has been cited in many scientific publications and it holds a Python library that offers great flexibility and versatibility while allowing users to create customized scripts and routines.

`PyMeepPlasmonics` aims to apply PyMEEP FDTD simulations to the nanoscale and it is particularly focused in studying the optical properties of metallic nanoparticles. It has four fully equiped routines that configure and perform simulations to extract, analyse and save data both in time-domain (direct electromagnetic field calculations) and in frequency domain (indirect scattering cross section calculations).

In order to build these routines, several code modules are used, each and every one of them highly documented. By calling functions from these modules and combining them with other famous scientific packages such as `numpy` or `matplotlib`, `PyMeepPlasmonics` should be able to present you with a whole spectrum of possibilities that will help you build your own Plasmonics or Nanophysics simulations.

## Installation

### Ubuntu

A series of simple instructions can be followed to install the required packages in Ubuntu 20.04 LTS (Focal Fossa) or a newer Ubuntu OS.

*Section under construction*

### Windows

As required by MEEP, an Ubuntu OS will be necessary, so if you wish to work in Windows you will need a virtual machine or similar. This master thesis simulations were executed in Ubuntu, so saddly you'll need to find out how to install it yourself.

## Licence

*Free open-source code developed as part of a Master Thesis Project in Physics*

***FDTD Applications To The Nanoscale: Photonics & Plasmonics Through MEEP***

Development by Valeria R. Pais, [DF, FCEyN, UBA](https://sitio.df.uba.ar/es/)

Direction by Prof. Dr. Fernando D. Stefani, [Applied Nanophysics Group, CIBION, CONICET](https://stefani-lab.ar/)

