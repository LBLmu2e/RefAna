import numpy as np
import h5py
import math
import hist

def new_hist(name,label,bins=100,range=[],title="",xlabel=""):
    h = hist.Hist(hist.axis.Regular(bins,range[0],range[1],label=xlabel),name=name,label=label)
    h.title = title
    return h

def load_hist(name,label,file,verbose=False):
    grp = "/"+name+"/"+label
    with h5py.File(file, 'r') as hdf5file: # closes on exit
        data = hdf5file[grp+"/data"][:]
        edges = hdf5file[grp+"/edges"][:]
        title = hdf5file.get(grp+"/title").asstr()[0]
        xlabel = hdf5file.get(grp+"/xlabel").asstr()[0]
    h = hist.Hist(hist.axis.Regular(len(data),edges[0],edges[-1],label=xlabel),name=name,label=label)
    h.title = title
    h.view()[:] = data
    if verbose:
        print("Read ",end=' ')
        print_hist(h)
    return h

def save_hist(h,hdf5file,verbose=False):
    grp = hdf5file.create_group("/"+h.name+"/"+h.label)
    grp.create_dataset("data",data=h.view())
    grp.create_dataset("edges",data=h.axes[0].edges)
    dst = grp.create_dataset("title", shape=1, dtype=h5py.string_dtype())
    dst[:] = h.title
    dsx = grp.create_dataset("xlabel", shape=1, dtype=h5py.string_dtype())
    dsx[:] = h.axes[0].label
    if verbose:
        print("Saved","/"+h.name+"/"+h.label,"to",hdf5file.filename)

def print_hist(h):
    print("hist","/"+h.name+"/"+h.label,"with",len(h.view()),"bins and",integral(h),"entries")

def bin_centers(h):
    return h.axes[0].centers

def bin_width(h):
    return h.axes[0].widths[0]

def integral(h):
    return h.sum()

def max_val(h):
    return np.max(h.view())

def max_x(h):
    data = h.view()
    edges = h.axes[0].edges
    maxindex = np.argmax(data)
    return 0.5*(edges[maxindex]+edges[maxindex+1])

# bin range for bins with values above the given value
def bin_range_above_value(h,minval):
    data = h.view()
    istart = 0
    while (istart < len(data)) and (data[istart] < minval):
        istart += 1
    iend = len(data)-1
    while (iend > 0) and (data[iend] < minval):
        iend -= 1
    iend += 1
    return [istart,iend]

def range_above_value(h,minval):
    edges = h.axes[0].edges
    brange = bin_range_above_value(h,minval)
    if brange[1] >= brange[0]:
        return np.array([edges[brange[0]],edges[brange[1]+1]])
    else:
        return np.array([edges[0],edges[-1]])

def bin_range(h,xrange):
    data = h.view()
    edges = h.axes[0].edges
    if xrange[1] < xrange[0]:
        return [0,len(data)-1]
    istart = 0
    while (istart < len(data)) and (edges[istart] < xrange[0]):
        istart += 1
    iend = len(data)-1
    while (iend > 0) and (edges[iend] > xrange[1]):
        iend -= 1
    iend += 1
    return [istart,iend]

def fit_arrays(h,xrange=[0,-1]):
    br = bin_range(h,xrange)
    binmid = bin_centers(h)[br[0]:br[1]]
    binval = h.view()[br[0]:br[1]]
    binerr = bin_errors(h)[br[0]:br[1]]
    return binmid, binval, binerr

def bin_errors(h):
    # assume unweighted bins, Poisson statis
    data = h.view()
    errors = np.sqrt(data)
    ones = np.ones(len(data))
    errors = np.maximum(ones,errors)
    return errors

def mean(h):
    return np.average(bin_centers(h),weights=h.view())

def variance(h):
    m = mean(h)
    return np.average(np.square(bin_centers(h)-m),weights=h.view())

def RMS(h):
    return math.sqrt(variance(h))

def FWHM(h):
    halfmax = 0.5*max_val(h)
    [ilow,ihigh] = bin_range_above_value(h,halfmax)
    edges = h.axes[0].edges
    # this is intentionally one higher than ilow, as bin edges are at the lower edge
    return edges[ihigh]-edges[ilow]

def plot(h,axis):
    plt = axis.stairs(h.view(),h.axes[0].edges,label=h.label)
    title = getattr(h,'title',"")
    if title:
        axis.set_title(title)
    xlabel = h.axes[0].label
    if xlabel:
        axis.set_xlabel(xlabel)
    return plt

def plot_errors(h,axis):
    plt = plot(h,axis)
    errors = bin_errors(h)
    centers = bin_centers(h)
    axis.errorbar(x=centers,y=h.view(),yerr=errors)
    return plt

def fit(h,fitobj,xrange=[0,-1],verbose=False,subplot=None):
    params,pcov,binmid = fitobj.fit(h,verbose=verbose,xrange=xrange)
    if subplot != None:
        fval = fitobj.fxn(binmid, *params)
        subplot.plot(binmid, fval, 'r-',label="Fit")
        ymax = 0.8
        edges = h.axes[0].edges
        subplot.text(edges[-1], ymax, f"{fitobj.name()}",transform=subplot.get_xaxis_transform(),ha="right" )
        for ipar in range(len(params)):
            yval = ymax - 0.05*(1+ipar)
            subplot.text(edges[-1], yval, f"{fitobj.pname[ipar]} = {params[ipar]:.3f}",transform=subplot.get_xaxis_transform(),ha="right" )
    return params, pcov
