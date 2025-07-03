#
# fit reflected momentum different response to a convolved function.
#
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special
import SurfaceIds as SID
import MyHist
import h5py
from scipy.stats import crystalball

def fxn_ExpGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

# Linearly interpolate/extrapolate a function sampled on an evenly-spaced set of values for a give value
def LinInterp(x,xmin,xstep,nstep,yvals):
    xmax = xmin + nstep*xstep
    abovemin = (x > xmin)
    belowmax = (x < xmax)
    inrange = abovemin & belowmax
    if abovemin & belowmax:
        # interpolate
        ibin = np.floor((x-xmin)/xstep).astype(np.int64)
        xbin = xmin+ibin*xstep
        return yvals[ibin] + (yvals[ibin+1]-yvals[ibin])*(x-xbin)/xstep
    elif belowmax:
        # negative extrapolation
        return yvals[0] + (yvals[1]-yvals[0])*(x-xmin)/xstep
    else:
        # positive extrapolation
        return yvals[-1] + (yvals[-1]-yvals[-2])*(x-xmax)/xstep


def fxn_CrystalBall(x, amp, beta, m, loc, scale):
    pars = np.array([beta, m, loc, scale])
    return amp*crystalball.pdf(x,*pars)


#def fxn_ConvCrystalBall(x, beta, m, loc, scale):





def fxn_ConvExpGauss(x, amp, mu, sigma, lamb):
    momstep =0.010 # 10 KeV step
    lowmom = -10.0
    himom = 5.0
    xvals = np.arange(lowmom,himom,momstep)
    pars = np.array([amp, mu, sigma, lamb]) # initial parameters
    yvals = list(map(lambda x: fxn_ExpGauss(x,*pars),xvals))
    conv = np.convolve(yvals,yvals,mode="same")
#    return LinInterp(x,lowmom,momstep,len(xvals),conv)
    ibin = np.floor((x-lowmom)/momstep).astype(np.int64)
    xbin = lowmom+ibin*momstep
    return conv[ibin] + (conv[ibin+1]-conv[ibin])*(x-xbin)/momstep

class FitReflections(object):
    def __init__(self,savefile):
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Mat",file=savefile)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="Target",file=savefile)

    def TestConvExpGauss(self,lowmom=-10.0, himom = 5.0,step=0.1):
        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        amp_0 = dmomsum*step # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        xvals = np.arange(lowmom,himom,step)
        edges = np.arange(lowmom-0.4999*step,himom+0.4999*step,step)
        noconvvals = list(map(lambda x: fxn_ExpGauss(x,*p0),xvals))
        convvals = list(map(lambda x: fxn_ConvExpGauss(x,*p0),xvals))
        fig, (noconv,conv) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        noconvplt = noconv.stairs(noconvvals,edges,label="noconv")
        convplt = conv.stairs(convvals,edges,label="convolved")

    def TestCrystalBall(self,beta_0=1.5,m_0=3.0,loc_0=0.0,scale_0=1.0):
        fig, (noconv,conv) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        binsize = self.HDeltaNoMatMom.edges[1]- self.HDeltaNoMatMom.edges[0]
        # initialize the fit parameters
#        loc_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
#        beta_0 = 1.5
#        m_0 = 3.0
#        scale_0 = 1.0/dmomsum # initial amplitud
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0,beta_0, m_0, loc_0, scale_0]) # initial parameters
        noconv.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        noconv.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\beta$ = {p0[1]:.3f}")
        fig.text(0.1, 0.4, f"m = {p0[2]:.3f}")
        fig.text(0.1, 0.3,  f"loc = {p0[3]:.3f}")

    def FitCrystalBall(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        binsize = self.HDeltaNoMatMom.edges[1]- self.HDeltaNoMatMom.edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaNoMatMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        self.HDeltaNoMatMom.plotErrors(delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\beta$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"m = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"loc = {popt[3]:.3f}")
        fig.text(0.1, 0.2,  f"scale = {popt[4]:.3f}")
        fig.text(0.1, 0.1,  f"amp = {popt[0]:.3f}")

        dmomerr = self.HDeltaTgtMom.binErrors()
        dmommid = self.HDeltaTgtMom.binCenters()
        dmomsum = self.HDeltaTgtMom.integral()
        binsize = self.HDeltaTgtMom.edges[1]- self.HDeltaTgtMom.edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMom.data/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.5
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0,beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        self.HDeltaTgtMom.plotErrors(delselmom)
        delselmom.plot(dmommid, fxn_CrystalBall(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\beta$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"m = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"loc = {popt[3]:.3f}")
        fig.text(0.6, 0.2,  f"scale = {popt[4]:.3f}")
        fig.text(0.6, 0.1,  f"amp = {popt[0]:.3f}")

    def FitExpGauss(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaNoMatMom.edges[1]- self.HDeltaNoMatMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ExpGauss, dmommid, self.HDeltaNoMatMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        self.HDeltaNoMatMom.plotErrors(delmom)
        delmom.plot(dmommid, fxn_ExpGauss(dmommid, *popt), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

        dmomerr = self.HDeltaTgtMom.binErrors()
        dmommid = self.HDeltaTgtMom.binCenters()
        dmomsum = self.HDeltaTgtMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaTgtMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaTgtMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaTgtMom.edges[1]- self.HDeltaTgtMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ExpGauss, dmommid, self.HDeltaTgtMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        self.HDeltaTgtMom.plotErrors(delselmom)
        delselmom.plot(dmommid, fxn_ExpGauss(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

    def FitConvExpGauss(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaNoMatMom.edges[1]- self.HDeltaNoMatMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ConvExpGauss, dmommid, self.HDeltaNoMatMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        self.HDeltaNoMatMom.plotErrors(delmom)
        delmom.plot(dmommid, fxn_ConvExpGauss(dmommid, *popt), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

        dmomerr = self.HDeltaTgtMom.binErrors()
        dmommid = self.HDeltaTgtMom.binCenters()
        dmomsum = self.HDeltaTgtMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaTgtMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaTgtMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaTgtMom.edges[1]- self.HDeltaTgtMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ConvExpGauss, dmommid, self.HDeltaTgtMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        self.HDeltaTgtMom.plotErrors(delselmom)
        delselmom.plot(dmommid, fxn_ConvExpGauss(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")


