#.rst:
# FindDLib
# --------
#
# Find the native DLib includes and libraries.
#
# DLib is a collection of C++ classes to solve common tasks in C++ programs, as well as to 
# offer additional functionality to use OpenCV data and to solve computer vision problems.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  DLIB_FOUND          - True if DLib found on the local system
#  DLIB_INCLUDE_DIRS   - Location of DLib header files.
#  DLIB_LIBRARIES      - The DLib libraries.
#
# Hints
# ^^^^^
#
# Set ``DLIB_ROOT_DIR`` to a directory that contains a DLib installation.
#
# This script expects to find libraries at ``$DLIB_ROOT_DIR/`` and the DLib
# headers at ``$DLIB_ROOT_DIR``.  

# =============================================================================
# Copyright (c) 2015, Simone Gasparini <simone.gasparini@gmail.com> All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================================================================

FIND_PATH(DLIB_INCLUDE_DIRS
	NAMES ./DUtils/DUtils.h
	HINTS
    ${DLIB_ROOT_DIR}/include
	PATHS
    /usr/include
    /usr/local/include
    /sw/include
    /opt/local/include
    DOC "The directory where DLib headers reside")

 message(STATUS  "DLIB_INCLUDE_DIRS = ${DLIB_INCLUDE_DIRS}")


FIND_LIBRARY(DLIB_LIBRARIES 
	NAMES DLib
	PATHS ${DLIB_ROOT_DIR}/lib
	DOC "The DLib library")

message(STATUS  "DLIB_LIBRARIES = ${DLIB_LIBRARIES}")


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBDLIB_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args( libDLIB  DEFAULT_MSG
                                  DLIB_LIBRARIES DLIB_INCLUDE_DIRS )

