
blaslib = [matlabroot, '/bin/glnxa64/libmwblas.so'];
lapacklib = [matlabroot, '/bin/glnxa64/libmwlapack.so'];

mex('-v', '-largeArrayDims', 'CXXFLAGS="$CXXFLAGS -std=c++0x"', 'struct_conn_density_prior_mex.cpp', blaslib, lapacklib);
