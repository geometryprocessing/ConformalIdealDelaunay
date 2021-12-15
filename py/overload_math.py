import math
from multipledispatch import dispatch 
from mpmath import fp,mp,iv
import numpy as np

# support for (more or less transparent) use of float, multiprecision and interval
# interchangeably
# not trusting mpmath dispatch 

def mpi(T):
    if T == iv.mpf:
        return iv.pi
    elif T == mp.mpf:
        return mp.pi
    else:
        return math.pi

@dispatch(float)
def mexp(x):
    return math.exp(x)

@dispatch(mp.mpf)
def mexp(x):
    return mp.exp(x)

@dispatch(iv.mpf)
def mexp(x):
    return iv.exp(x)

@dispatch(float)
def mlog(x):
    return math.log(x)

@dispatch(mp.mpf)
def mlog(x):
    return mp.log(x)

@dispatch(iv.mpf)
def mlog(x):
    return iv.log(x)



@dispatch(float)
def msqrt(x):
    return math.sqrt(x)

@dispatch(mp.mpf)
def msqrt(x):
    return mp.sqrt(x)

@dispatch(iv.mpf)
def msqrt(x):
    return iv.sqrt(x)


@dispatch(float)
def macos(x):
    return math.acos(x)

@dispatch(mp.mpf)
def macos(x):
    return mp.acos(x)

@dispatch(iv.mpf)
def macos(x):
    return iv.acos(x)

@dispatch(float,float)
def matan2(y,x):
    return math.atan2(y,x)

@dispatch(mp.mpf,mp.mpf)
def matan2(y,x):
    return mp.atan2(y,x)

@dispatch(iv.mpf,iv.mpf)
def matan2(y,x):
    return iv.atan2(y,x)

# Clausen function -- needed for conformal energy, not in standard math 
# but thankfully supported by mpmath 

@dispatch(int,float)
def mclsin(n,x):
    return fp.clsin(n,x)

@dispatch(int,mp.mpf)
def mclsin(n,x):
    return mp.clsin(n,x)

@dispatch(int,iv.mpf)
def mclsin(n,x):
    return iv.clsin(n,x)   

def nparray_to_float64(a):
    assert type(a) == np.ndarray,'input not an numpy array'
    return a.astype(float)

def nparray_from_float64(a,ftype):
    assert type(a) == np.ndarray,'input not an numpy array'
    return np.array([ftype(d) for d in a])
