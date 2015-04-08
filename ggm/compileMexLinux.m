
blaslib = [matlabroot, '/bin/glnxa64/libmwblas.so'];
lapacklib = [matlabroot, '/bin/glnxa64/libmwlapack.so'];

mex('-v', '-largeArrayDims', 'CXXFLAGS="$CXXFLAGS -std=c++0x"', 'gwishrnd_mex.cpp', blaslib, lapacklib);
mex('-v', '-largeArrayDims', 'CXXFLAGS="$CXXFLAGS -std=c++0x"', 'ggm_cbf_mex.cpp', blaslib, lapacklib);
