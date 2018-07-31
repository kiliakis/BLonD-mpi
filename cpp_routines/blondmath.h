/**
Copyright 2016 CERN. This software is distributed under the
terms of the GNU General Public Licence version 3 (GPL Version 3),
copied verbatim in the file LICENCE.md.
In applying this licence, CERN does not waive the privileges and immunities
granted to it by virtue of its status as an Intergovernmental Organization or
submit itself to any jurisdiction.
Project website: http://blond.web.cern.ch/


C++ Math library
@Author: Konstantinos Iliakis
@Date: 20.10.2017
*/



extern "C" {

    void convolution(const double * __restrict__ signal,
                     const int SignalLen,
                     const double * __restrict__ kernel,
                     const int KernelLen,
                     double * __restrict__ res);

    double mean(const double * __restrict__ data, const int n);
    double stdev(const double * __restrict__ data,
                 const int n);
    double fast_sin(double x);
    double fast_cos(double x);
    double fast_exp(double x);
    void fast_sinv(const double * __restrict__ in,
                   const int size,
                   double * __restrict__ out);
    void fast_cosv(const double * __restrict__ in,
                   const int size,
                   double * __restrict__ out);
    void fast_expv(const double * __restrict__ in,
                   const int size,
                   double * __restrict__ out);
    void add_int_vector(const int *__restrict__ a,
                        const int *__restrict__ b,
                        const int size,
                        int *__restrict__ result);
    void add_longint_vector(const long *__restrict__ a,
                            const long *__restrict__ b,
                            const int size,
                            long *__restrict__ result);
    void add_double_vector(const double *__restrict__ a,
                           const double *__restrict__ b,
                           const int size,
                           double *__restrict__ result);

    /**
    Parameters are like python's np.interp

    @x: x-coordinates of the interpolated values
    @N: The x array size
    @xp: The x-coords of the data points, !!must be sorted!!
    @M: The xp array size
    @yp: the y-coords of the data points
    @left: value to return for x < xp[0]
    @right: value to return for x > xp[last]
    @y: the interpolated values, same shape as x
    */
    void interp(const double * __restrict__ x,
                const int N,
                const double * __restrict__ xp,
                const int M,
                const double * __restrict__ yp,
                const double left,
                const double right,
                double * __restrict__ y);

    // Function to implement integration of f(x); over the interval
    // [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_wo_initial(const double * __restrict__ f,
                             const double deltaX,
                             const int nsub,
                             double * __restrict__ psum);

    // Function to implement integration of f(x); over the interval
    // [a,b] using the trapezoid rule with nsub subdivisions.
    void cumtrapz_w_initial(const double * __restrict__ f,
                            const double deltaX,
                            const double initial,
                            const int nsub,
                            double * __restrict__ psum);

    double trapz_var_delta(const double * __restrict__ f,
                           const double * __restrict__ deltaX,
                           const int nsub);

    double trapz_const_delta(const double * __restrict__ f,
                             const double deltaX,
                             const int nsub);

    int min_idx(const double * __restrict__ a, int size);
    int max_idx(const double * __restrict__ a, int size);
    void linspace(const double start, const double end, const int n,
                  double *__restrict__ out);

    void arange_double(const double start, const double stop,
                       const double step,
                       double * __restrict__ out);

    void arange_int(const int start, const int stop,
                    const int step,
                    int * __restrict__ out);

    double sum(const double * __restrict__ data, const int n);
    void sort_double(double * __restrict__ in, const int n, bool reverse);
    void sort_int(int * __restrict__ in, const int n, bool reverse);
    void sort_longint(long int * __restrict__ in, const int n, bool reverse);
}
