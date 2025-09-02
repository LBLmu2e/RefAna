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
from scipy.signal import convolve as convo
import copy

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

def fxn_ConvCrystalBall(x, amp, beta, m, loc, scale):
    momstep =0.01 # 10 KeV step
    lowmom = -10.0
    himom = 10.0
    xvals = np.arange(lowmom,himom,momstep)
    pars = np.array([beta, m, loc, scale])
    yvals = list(map(lambda x: crystalball.pdf(x,*pars),xvals))
#    conv = scipy.convolve(yvals,yvals,mode="same",method="direct")
    conv = convo(yvals,yvals,mode="same",method="direct")
    ibin = np.floor((x-lowmom)/momstep).astype(np.int64)
    xbin = lowmom+ibin*momstep
    return amp*(conv[ibin] + (conv[ibin+1]-conv[ibin])*(x-xbin)/momstep)

def fxn_ExpGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

def fxn_ConvExpGauss(x, amp, mu, sigma, lamb):
    momstep =0.010 # 10 KeV step
    lowmom = -10.0
    himom = 10.0
    xvals = np.arange(lowmom,himom,momstep)
    pars = np.array([momstep, mu, sigma, lamb]) # initial parameters
    yvals = list(map(lambda x: fxn_ExpGauss(x,*pars),xvals))
    conv = np.convolve(yvals,yvals,mode="same")
    ibin = np.floor((x-lowmom)/momstep).astype(np.int64)
    xbin = lowmom+ibin*momstep
    return amp*(conv[ibin] + (conv[ibin+1]-conv[ibin])*(x-xbin)/momstep)

class FitReflections(object):
    def __init__(self,reffile,cefile=None):
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Mat",file=reffile)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="Target",file=reffile)
        self.HDeltaTgtMomB12 = MyHist.MyHist(name="DeltaMom",label="B12",file=reffile)
        self.HDeltaTgtMomB34 = MyHist.MyHist(name="DeltaMom",label="B34",file=reffile)
        self.HDeltaTgtMomB56 = MyHist.MyHist(name="DeltaMom",label="B56",file=reffile)
        self.HDeltaTgtMomB78 = MyHist.MyHist(name="DeltaMom",label="B78",file=reffile)
        self.HDeltaTgtMomB9p = MyHist.MyHist(name="DeltaMom",label="B9p",file=reffile)
        self.HDeltaNoMatMom.title = "Reflected " + self.HDeltaNoMatMom.title
        self.HDeltaTgtMom.title = "Reflected " + self.HDeltaTgtMom.title
        self.HDeltaTgtMomB12.title = "Reflected " + self.HDeltaTgtMomB12.title
        self.HDeltaTgtMomB34.title = "Reflected " + self.HDeltaTgtMomB34.title
        self.HDeltaTgtMomB56.title = "Reflected " + self.HDeltaTgtMomB56.title
        self.HDeltaTgtMomB78.title = "Reflected " + self.HDeltaTgtMomB78.title
        self.HDeltaTgtMomB9p.title = "Reflected " + self.HDeltaTgtMomB9p.title
        self.hasCe = ( cefile != None)
        if self.hasCe:
            self.HCeRefResp = MyHist.MyHist(name="TT_FrontResponse",label="Reflectable",file=cefile)
            self.HCeAllResp = MyHist.MyHist(name="TT_FrontResponse",label="All",file=cefile)
            self.HCeRefResp.title="Ce "+ self.HCeRefResp.title
            self.HCeAllResp.title="Ce "+ self.HCeAllResp.title

            ### BINNED ###
            self.HCeTgtMomB12 = MyHist.MyHist(name="DMom",label="B12", file=cefile)
            self.HCeTgtMomB34 = MyHist.MyHist(name="DMom",label="B34", file=cefile)
            self.HCeTgtMomB56 = MyHist.MyHist(name="DMom",label="B56", file=cefile)
            self.HCeTgtMomB78 = MyHist.MyHist(name="DMom",label="B78", file=cefile)
            self.HCeTgtMomB9p = MyHist.MyHist(name="DMom",label="B9p", file=cefile)

            self.HCeTgtMomB12.title = "Ce " + self.HDeltaTgtMomB12.title
            self.HCeTgtMomB34.title = "Ce " + self.HDeltaTgtMomB34.title
            self.HCeTgtMomB56.title = "Ce " + self.HDeltaTgtMomB56.title
            self.HCeTgtMomB78.title = "Ce " + self.HDeltaTgtMomB78.title
            self.HCeTgtMomB9p.title = "Ce " + self.HDeltaTgtMomB9p.title

    def TestExpGauss(self):
        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        binsize = self.HDeltaNoMatMom.edges[1]- self.HDeltaNoMatMom.edges[0]
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        fig, (noconv,conv) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        noconv.plot(dmommid, fxn_ExpGauss(dmommid, *p0), 'r-',label="Direct")
        conv.plot(dmommid, fxn_ConvExpGauss(dmommid, *p0), 'r-',label="Convolved")
        conv.legend(loc="upper right")
        noconv.legend(loc="upper right")

    def TestCrystalBall(self,beta_0=1.0,m_0=3.0,loc_0=-0.5,scale_0=0.3):
        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        binsize = self.HDeltaNoMatMom.edges[1]- self.HDeltaNoMatMom.edges[0]
        # initialize the fit parameters
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0,beta_0, m_0, loc_0, scale_0]) # initial parameters
        fig, (anoconv,aconv,afit) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        anoconv.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="CB Function")
        aconv.plot(dmommid, fxn_ConvCrystalBall(dmommid, *p0), 'r-',label="Convolved CB Function")
        aconv.legend(loc="upper right")
        anoconv.legend(loc="upper right")
        fig.text(0.1, 0.8, f"$\\beta$ = {p0[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {p0[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {p0[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {p0[4]:.3f}")
        # test fitting: first generate a convolved distribution
        r1 = crystalball.rvs(beta=beta_0,m=m_0, loc=loc_0, scale=scale_0, random_state=0, size=10000)
        r2 = crystalball.rvs(beta=beta_0,m=m_0, loc=loc_0, scale=scale_0, random_state=1, size=10000)
        rconv = r1+r2
        convhist = MyHist.MyHist(name="convhist",label="Manually Convolved CB",bins=200,range=[-10.0,5],xlabel="$\\Delta$ Momentum")
        convhist.fill(rconv)
        convhist.plot(afit)
        # then fit
        dmomerr = convhist.binErrors()
        dmommid = convhist.binCenters()
        dmomsum = convhist.integral()
        binsize = convhist.edges[1]- convhist.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        refpartest, refcovtest = curve_fit(fxn_ConvCrystalBall, dmommid, convhist.data, p0, sigma=dmomerr)
        refperrtest = np.sqrt(np.diagonal(refcovtest))
#        convhist.plotErrors(afit)
        maxval = np.amax(convhist.data)
        afit.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refpartest), 'r-',label="Convolved CB Fit")
        afit.legend(loc="upper right")
        afit.text(-8, 0.8*maxval, f"$\\beta$ = {refpartest[1]:.3f} $\\pm$ {refperrtest[1]:.3f}")
        afit.text(-8, 0.7*maxval, f"m = {refpartest[2]:.3f} $\\pm$ {refperrtest[2]:.3f}")
        afit.text(-8, 0.6*maxval,  f"loc = {refpartest[3]:.3f} $\\pm$ {refperrtest[3]:.3f}")
        afit.text(-8, 0.5*maxval,  f"scale = {refpartest[4]:.3f} $\\pm$ {refperrtest[4]:.3f}")

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
        perr = np.sqrt(np.diagonal(pcov))

        self.HDeltaNoMatMom.plotErrors(delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        
        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

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
        perr = np.sqrt(np.diagonal(pcov))

        self.HDeltaTgtMom.plotErrors(delselmom)
        delselmom.plot(dmommid, fxn_CrystalBall(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")

        fig.text(0.6, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.6, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.6, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.6, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        """ BINS """

        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        # B12

        dmomerr = self.HDeltaTgtMomB12.binErrors()
        dmommid = self.HDeltaTgtMomB12.binCenters()
        dmomsum = self.HDeltaTgtMomB12.integral()
        binsize = self.HDeltaTgtMomB12.edges[1]- self.HDeltaTgtMomB12.edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB12.data/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB12.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        self.HDeltaTgtMomB12.plotErrors(delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        
        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B34

        dmomerr = self.HDeltaTgtMomB34.binErrors()
        dmommid = self.HDeltaTgtMomB34.binCenters()
        dmomsum = self.HDeltaTgtMomB34.integral()
        binsize = self.HDeltaTgtMomB34.edges[1]- self.HDeltaTgtMomB34.edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB34.data/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB34.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        self.HDeltaTgtMomB34.plotErrors(delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        
        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B56

        dmomerr = self.HDeltaTgtMomB56.binErrors()
        dmommid = self.HDeltaTgtMomB56.binCenters()
        dmomsum = self.HDeltaTgtMomB56.integral()
        binsize = self.HDeltaTgtMomB56.edges[1]- self.HDeltaTgtMomB56.edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB56.data/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB56.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        self.HDeltaTgtMomB56.plotErrors(delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        
        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B78

        dmomerr = self.HDeltaTgtMomB78.binErrors()
        dmommid = self.HDeltaTgtMomB78.binCenters()
        dmomsum = self.HDeltaTgtMomB78.integral()
        binsize = self.HDeltaTgtMomB78.edges[1]- self.HDeltaTgtMomB78.edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB78.data/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB78.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        self.HDeltaTgtMomB78.plotErrors(delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        
        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B9p

        dmomerr = self.HDeltaTgtMomB9p.binErrors()
        dmommid = self.HDeltaTgtMomB9p.binCenters()
        dmomsum = self.HDeltaTgtMomB9p.integral()
        binsize = self.HDeltaTgtMomB9p.edges[1]- self.HDeltaTgtMomB9p.edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB9p.data/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB9p.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        self.HDeltaTgtMomB9p.plotErrors(delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        
        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")


    def FitConvCrystalBall(self): # steal normalization from fit
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(15,5))

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
        refparnomat, refcovnomat = curve_fit(fxn_ConvCrystalBall, dmommid, self.HDeltaNoMatMom.data, p0, sigma=dmomerr)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat = np.sqrt(np.diagonal(refcovnomat))

        self.HDeltaNoMatMom.plotErrors(delmom)
        maxval = np.amax(self.HDeltaNoMatMom.data)
        delmom.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refparnomat), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        delmom.text(-8, 0.8*maxval, f"$\\beta$ = {refparnomat[1]:.3f} $\\pm$ {refperrnomat[1]:.3f}")
        delmom.text(-8, 0.7*maxval, f"m = {refparnomat[2]:.3f} $\\pm$ {refperrnomat[2]:.3f}")
        delmom.text(-8, 0.6*maxval,  f"loc = {refparnomat[3]:.3f} $\\pm$ {refperrnomat[3]:.3f}")
        delmom.text(-8, 0.5*maxval,  f"scale = {refparnomat[4]:.3f} $\\pm$ {refperrnomat[4]:.3f}")

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
        refpartgt, refcovtgt = curve_fit(fxn_ConvCrystalBall, dmommid, self.HDeltaTgtMom.data, p0, sigma=dmomerr)
#        print("Target inter fit parameters",refpartgt)
#        print("Target inter fit covariance",refcovtgt)
        refperrtgt = np.sqrt(np.diagonal(refcovtgt))

        self.HDeltaTgtMom.plotErrors(delselmom)
        maxval = np.amax(self.HDeltaTgtMom.data)
        delselmom.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refpartgt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        delselmom.text(-8, 0.8*maxval, f"$\\beta$ = {refpartgt[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
        delselmom.text(-8, 0.7*maxval, f"m = {refpartgt[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
        delselmom.text(-8, 0.6*maxval,  f"loc = {refpartgt[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
        delselmom.text(-8, 0.5*maxval,  f"scale = {refpartgt[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

        """ Binned """

        fig, (delmomB12,delmomB34,delmomB56,delmomB78,delmomB9p) = plt.subplots(1,5,layout='constrained', figsize=(15,5))

        # B12

        dmomerr12 = self.HDeltaTgtMomB12.binErrors()
        dmommid12 = self.HDeltaTgtMomB12.binCenters()
        dmomsum12 = self.HDeltaTgtMomB12.integral()
        binsize12 = self.HDeltaTgtMomB12.edges[1]- self.HDeltaTgtMomB12.edges[0]
        # initialize the fit parameters
        loc_012 = np.mean(dmommid12*self.HDeltaTgtMomB12.data/dmomsum12) # initial mean
        beta_012 = 1.0
        m_012 = 3.0
        scale_012 = 0.20
        amp_012 = dmomsum12*binsize12 # initial amplitude
        p012 = np.array([amp_012, beta_012, m_012, loc_012, scale_012]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat12, refcovnomat12 = curve_fit(fxn_ConvCrystalBall, dmommid12, self.HDeltaTgtMomB12.data, p012, sigma=dmomerr12)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat12 = np.sqrt(np.diagonal(refcovnomat12))

        self.HDeltaTgtMomB12.plotErrors(delmomB12)
        maxval12 = np.amax(self.HDeltaTgtMomB12.data)
        delmomB12.plot(dmommid12, fxn_ConvCrystalBall(dmommid12, *refparnomat12), 'r-',label="Fit")
        delmomB12.legend(loc="upper right")
        delmomB12.text(-8, 0.8*maxval12, f"$\\beta$ = {refparnomat12[1]:.3f} $\\pm$ {refperrnomat12[1]:.3f}")
        delmomB12.text(-8, 0.7*maxval12, f"m = {refparnomat12[2]:.3f} $\\pm$ {refperrnomat12[2]:.3f}")
        delmomB12.text(-8, 0.6*maxval12,  f"loc = {refparnomat12[3]:.3f} $\\pm$ {refperrnomat12[3]:.3f}")
        delmomB12.text(-8, 0.5*maxval12,  f"scale = {refparnomat12[4]:.3f} $\\pm$ {refperrnomat12[4]:.3f}")

        # B34

        dmomerr34 = self.HDeltaTgtMomB34.binErrors()
        dmommid34 = self.HDeltaTgtMomB34.binCenters()
        dmomsum34 = self.HDeltaTgtMomB34.integral()
        binsize34 = self.HDeltaTgtMomB34.edges[1]- self.HDeltaTgtMomB34.edges[0]
        # initialize the fit parameters
        loc_034 = np.mean(dmommid34*self.HDeltaTgtMomB34.data/dmomsum34) # initial mean
        beta_034 = 1.0
        m_034 = 3.0
        scale_034 = 0.20
        amp_034 = dmomsum34*binsize34 # initial amplitude
        p034 = np.array([amp_034, beta_034, m_034, loc_034, scale_034]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat34, refcovnomat34 = curve_fit(fxn_ConvCrystalBall, dmommid34, self.HDeltaTgtMomB34.data, p034, sigma=dmomerr34)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat34 = np.sqrt(np.diagonal(refcovnomat34))

        self.HDeltaTgtMomB34.plotErrors(delmomB34)
        maxval34 = np.amax(self.HDeltaTgtMomB34.data)
        delmomB34.plot(dmommid34, fxn_ConvCrystalBall(dmommid34, *refparnomat34), 'r-',label="Fit")
        delmomB34.legend(loc="upper right")
        delmomB34.text(-8, 0.8*maxval34, f"$\\beta$ = {refparnomat34[1]:.3f} $\\pm$ {refperrnomat34[1]:.3f}")
        delmomB34.text(-8, 0.7*maxval34, f"m = {refparnomat34[2]:.3f} $\\pm$ {refperrnomat34[2]:.3f}")
        delmomB34.text(-8, 0.6*maxval34,  f"loc = {refparnomat34[3]:.3f} $\\pm$ {refperrnomat34[3]:.3f}")
        delmomB34.text(-8, 0.5*maxval34,  f"scale = {refparnomat34[4]:.3f} $\\pm$ {refperrnomat34[4]:.3f}")

        # B56
        
        dmomerr56 = self.HDeltaTgtMomB56.binErrors()
        dmommid56 = self.HDeltaTgtMomB56.binCenters()
        dmomsum56 = self.HDeltaTgtMomB56.integral()
        binsize56 = self.HDeltaTgtMomB56.edges[1]- self.HDeltaTgtMomB56.edges[0]
        # initialize the fit parameters
        loc_056 = np.mean(dmommid56*self.HDeltaTgtMomB56.data/dmomsum56) # initial mean
        beta_056 = 1.0
        m_056 = 3.0
        scale_056 = 0.20
        amp_056 = dmomsum56*binsize56 # initial amplitude
        p056 = np.array([amp_056, beta_056, m_056, loc_056, scale_056]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat56, refcovnomat56 = curve_fit(fxn_ConvCrystalBall, dmommid56, self.HDeltaTgtMomB56.data, p056, sigma=dmomerr56)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat56 = np.sqrt(np.diagonal(refcovnomat56))

        self.HDeltaTgtMomB56.plotErrors(delmomB56)
        maxval56 = np.amax(self.HDeltaTgtMomB56.data)
        delmomB56.plot(dmommid56, fxn_ConvCrystalBall(dmommid56, *refparnomat56), 'r-',label="Fit")
        delmomB56.legend(loc="upper right")
        delmomB56.text(-8, 0.8*maxval56, f"$\\beta$ = {refparnomat56[1]:.3f} $\\pm$ {refperrnomat56[1]:.3f}")
        delmomB56.text(-8, 0.7*maxval56, f"m = {refparnomat56[2]:.3f} $\\pm$ {refperrnomat56[2]:.3f}")
        delmomB56.text(-8, 0.6*maxval56,  f"loc = {refparnomat56[3]:.3f} $\\pm$ {refperrnomat56[3]:.3f}")
        delmomB56.text(-8, 0.5*maxval56,  f"scale = {refparnomat56[4]:.3f} $\\pm$ {refperrnomat56[4]:.3f}") 

        # B78
        
        dmomerr78 = self.HDeltaTgtMomB78.binErrors()
        dmommid78 = self.HDeltaTgtMomB78.binCenters()
        dmomsum78 = self.HDeltaTgtMomB78.integral()
        binsize78 = self.HDeltaTgtMomB78.edges[1]- self.HDeltaTgtMomB78.edges[0]
        # initialize the fit parameters
        loc_078 = np.mean(dmommid78*self.HDeltaTgtMomB78.data/dmomsum78) # initial mean
        beta_078 = 1.0
        m_078 = 3.0
        scale_078 = 0.20
        amp_078 = dmomsum78*binsize78 # initial amplitude
        p078 = np.array([amp_078, beta_078, m_078, loc_078, scale_078]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat78, refcovnomat78 = curve_fit(fxn_ConvCrystalBall, dmommid78, self.HDeltaTgtMomB78.data, p078, sigma=dmomerr78)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat78 = np.sqrt(np.diagonal(refcovnomat78))

        self.HDeltaTgtMomB78.plotErrors(delmomB78)
        maxval78 = np.amax(self.HDeltaTgtMomB78.data)
        delmomB78.plot(dmommid78, fxn_ConvCrystalBall(dmommid78, *refparnomat78), 'r-',label="Fit")
        delmomB78.legend(loc="upper right")
        delmomB78.text(-8, 0.8*maxval78, f"$\\beta$ = {refparnomat78[1]:.3f} $\\pm$ {refperrnomat78[1]:.3f}")
        delmomB78.text(-8, 0.7*maxval78, f"m = {refparnomat78[2]:.3f} $\\pm$ {refperrnomat78[2]:.3f}")
        delmomB78.text(-8, 0.6*maxval78,  f"loc = {refparnomat78[3]:.3f} $\\pm$ {refperrnomat78[3]:.3f}")
        delmomB78.text(-8, 0.5*maxval78,  f"scale = {refparnomat78[4]:.3f} $\\pm$ {refperrnomat78[4]:.3f}")       

        # B9p
        
        dmomerr9p = self.HDeltaTgtMomB9p.binErrors()
        dmommid9p = self.HDeltaTgtMomB9p.binCenters()
        dmomsum9p = self.HDeltaTgtMomB9p.integral()
        binsize9p = self.HDeltaTgtMomB9p.edges[1]- self.HDeltaTgtMomB9p.edges[0]
        # initialize the fit parameters
        loc_09p = np.mean(dmommid9p*self.HDeltaTgtMomB9p.data/dmomsum9p) # initial mean
        beta_09p = 1.0
        m_09p = 3.0
        scale_09p = 0.20
        amp_09p = dmomsum9p*binsize9p # initial amplitude
        p09p = np.array([amp_09p, beta_09p, m_09p, loc_09p, scale_09p]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat9p, refcovnomat9p = curve_fit(fxn_ConvCrystalBall, dmommid9p, self.HDeltaTgtMomB9p.data, p09p, sigma=dmomerr9p)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat9p = np.sqrt(np.diagonal(refcovnomat9p))

        self.HDeltaTgtMomB9p.plotErrors(delmomB9p)
        maxval9p = np.amax(self.HDeltaTgtMomB9p.data)
        delmomB9p.plot(dmommid9p, fxn_ConvCrystalBall(dmommid9p, *refparnomat9p), 'r-',label="Fit")
        delmomB9p.legend(loc="upper right")
        delmomB9p.text(-8, 0.8*maxval9p, f"$\\beta$ = {refparnomat9p[1]:.3f} $\\pm$ {refperrnomat9p[1]:.3f}")
        delmomB9p.text(-8, 0.7*maxval9p, f"m = {refparnomat9p[2]:.3f} $\\pm$ {refperrnomat9p[2]:.3f}")
        delmomB9p.text(-8, 0.6*maxval9p,  f"loc = {refparnomat9p[3]:.3f} $\\pm$ {refperrnomat9p[3]:.3f}")
        delmomB9p.text(-8, 0.5*maxval9p,  f"scale = {refparnomat9p[4]:.3f} $\\pm$ {refperrnomat9p[4]:.3f}")               

        if self.hasCe:
            fig, (cefitrefresp,cecomprefresp) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = self.HCeRefResp.binCenters()
            ceint = self.HCeRefResp.integral()
            cebinsize = self.HCeRefResp.edges[1]- self.HCeRefResp.edges[0]
            loc_0 = np.mean(cebins*self.HCeRefResp.data/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeRefResp.data, p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            self.HCeRefResp.plotErrors(cefitrefresp)
            maxval = np.amax(self.HCeRefResp.data)
            cefitrefresp.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            cefitrefresp.legend(loc="upper right")
            cefitrefresp.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            cefitrefresp.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            cefitrefresp.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            cefitrefresp.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            self.HCeRefResp.plotErrors(cecomprefresp)
            comppars = copy.deepcopy(refpartgt)
            comppars[0] = cepars[0] # steal normalization from fit
            cecomprefresp.plot(cebins, fxn_CrystalBall(cebins, *comppars), 'r-',label="Comparison")
            cecomprefresp.legend(loc="upper right")
            cecomprefresp.text(-8, 0.8*maxval, f"$\\beta$ = {comppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            cecomprefresp.text(-8, 0.7*maxval, f"m = {comppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            cecomprefresp.text(-8, 0.6*maxval,  f"loc = {comppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            cecomprefresp.text(-8, 0.5*maxval,  f"scale = {comppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

            fig, (cefitallresp,cecompallresp) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = self.HCeAllResp.binCenters()
            ceint = self.HCeAllResp.integral()
            cebinsize = self.HCeAllResp.edges[1]- self.HCeAllResp.edges[0]
            loc_0 = np.mean(cebins*self.HCeAllResp.data/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeAllResp.data, p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            self.HCeAllResp.plotErrors(cefitallresp)
            maxval = np.amax(self.HCeAllResp.data)
            cefitallresp.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            cefitallresp.legend(loc="upper right")
            cefitallresp.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            cefitallresp.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            cefitallresp.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            cefitallresp.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            self.HCeAllResp.plotErrors(cecompallresp)
            comppars = copy.deepcopy(refpartgt)
            comppars[0] = cepars[0] # steal normalization from fit
            cecompallresp.plot(cebins, fxn_CrystalBall(cebins, *comppars), 'r-',label="Comparison")
            cecompallresp.legend(loc="upper right")
            cecompallresp.text(-8, 0.8*maxval, f"$\\beta$ = {comppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            cecompallresp.text(-8, 0.7*maxval, f"m = {comppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            cecompallresp.text(-8, 0.6*maxval,  f"loc = {comppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            cecompallresp.text(-8, 0.5*maxval,  f"scale = {comppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

            ### BINNED ### 

            fig, (aB12, aB34, aB56, aB78, aB9p) = plt.subplots(1,5,layout='constrained', figsize=(15,5))

            # B12
        
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = self.HCeTgtMomB12.binCenters()
            ceint = self.HCeTgtMomB12.integral()
            cebinsize = self.HCeTgtMomB12.edges[1]- self.HCeTgtMomB12.edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB12.data/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB12.data, p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            self.HCeTgtMomB12.plotErrors(cefitallresp)
            maxval = np.amax(self.HCeTgtMomB12.data)
            aB12.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB12.legend(loc="upper right")
            aB12.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB12.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB12.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB12.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            self.HCeTgtMomB12.plotErrors(cecompallresp)
            comppars = copy.deepcopy(refpartgt)
            comppars[0] = cepars[0] # steal normalization from fit
            aB12.plot(cebins, fxn_CrystalBall(cebins, *comppars), 'r-',label="Comparison")
            aB12.legend(loc="upper right")
            aB12.text(-8, 0.8*maxval, f"$\\beta$ = {comppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            aB12.text(-8, 0.7*maxval, f"m = {comppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            aB12.text(-8, 0.6*maxval,  f"loc = {comppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            aB12.text(-8, 0.5*maxval,  f"scale = {comppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

            # B34
        
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = self.HCeTgtMomB34.binCenters()
            ceint = self.HCeTgtMomB34.integral()
            cebinsize = self.HCeTgtMomB34.edges[1]- self.HCeTgtMomB34.edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB34.data/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB34.data, p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            self.HCeTgtMomB34.plotErrors(cefitallresp)
            maxval = np.amax(self.HCeTgtMomB34.data)
            aB34.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB34.legend(loc="upper right")
            aB34.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB34.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB34.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB34.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            self.HCeTgtMomB34.plotErrors(cecompallresp)
            comppars = copy.deepcopy(refpartgt)
            comppars[0] = cepars[0] # steal normalization from fit
            aB34.plot(cebins, fxn_CrystalBall(cebins, *comppars), 'r-',label="Comparison")
            aB34.legend(loc="upper right")
            aB34.text(-8, 0.8*maxval, f"$\\beta$ = {comppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            aB34.text(-8, 0.7*maxval, f"m = {comppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            aB34.text(-8, 0.6*maxval,  f"loc = {comppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            aB34.text(-8, 0.5*maxval,  f"scale = {comppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

            # B56
        
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = self.HCeTgtMomB56.binCenters()
            ceint = self.HCeTgtMomB56.integral()
            cebinsize = self.HCeTgtMomB56.edges[1]- self.HCeTgtMomB56.edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB56.data/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB56.data, p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            self.HCeTgtMomB56.plotErrors(cefitallresp)
            maxval = np.amax(self.HCeTgtMomB56.data)
            aB56.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB56.legend(loc="upper right")
            aB56.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB56.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB56.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB56.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            self.HCeTgtMomB56.plotErrors(cecompallresp)
            comppars = copy.deepcopy(refpartgt)
            comppars[0] = cepars[0] # steal normalization from fit
            aB56.plot(cebins, fxn_CrystalBall(cebins, *comppars), 'r-',label="Comparison")
            aB56.legend(loc="upper right")
            aB56.text(-8, 0.8*maxval, f"$\\beta$ = {comppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            aB56.text(-8, 0.7*maxval, f"m = {comppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            aB56.text(-8, 0.6*maxval,  f"loc = {comppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            aB56.text(-8, 0.5*maxval,  f"scale = {comppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

            # B78
        
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = self.HCeTgtMomB78.binCenters()
            ceint = self.HCeTgtMomB78.integral()
            cebinsize = self.HCeTgtMomB78.edges[1]- self.HCeTgtMomB78.edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB78.data/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB78.data, p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            self.HCeTgtMomB78.plotErrors(cefitallresp)
            maxval = np.amax(self.HCeTgtMomB78.data)
            aB78.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB78.legend(loc="upper right")
            aB78.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB78.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB78.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB78.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            self.HCeTgtMomB78.plotErrors(cecompallresp)
            comppars = copy.deepcopy(refpartgt)
            comppars[0] = cepars[0] # steal normalization from fit
            aB78.plot(cebins, fxn_CrystalBall(cebins, *comppars), 'r-',label="Comparison")
            aB78.legend(loc="upper right")
            aB78.text(-8, 0.8*maxval, f"$\\beta$ = {comppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            aB78.text(-8, 0.7*maxval, f"m = {comppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            aB78.text(-8, 0.6*maxval,  f"loc = {comppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            aB78.text(-8, 0.5*maxval,  f"scale = {comppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

            # B9p
        
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = self.HCeTgtMomB9p.binCenters()
            ceint = self.HCeTgtMomB9p.integral()
            cebinsize = self.HCeTgtMomB9p.edges[1]- self.HCeTgtMomB9p.edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB9p.data/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB9p.data, p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            self.HCeTgtMomB9p.plotErrors(cefitallresp)
            maxval = np.amax(self.HCeTgtMomB9p.data)
            aB9p.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB9p.legend(loc="upper right")
            aB9p.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB9p.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB9p.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB9p.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            self.HCeTgtMomB9p.plotErrors(cecompallresp)
            comppars = copy.deepcopy(refpartgt)
            comppars[0] = cepars[0] # steal normalization from fit
            aB9p.plot(cebins, fxn_CrystalBall(cebins, *comppars), 'r-',label="Comparison")
            aB9p.legend(loc="upper right")
            aB9p.text(-8, 0.8*maxval, f"$\\beta$ = {comppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            aB9p.text(-8, 0.7*maxval, f"m = {comppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            aB9p.text(-8, 0.6*maxval,  f"loc = {comppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            aB9p.text(-8, 0.5*maxval,  f"scale = {comppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

        fig, (delmomlog,delselmomlog) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
        delmomlog.set_yscale("log")
        delmomlog.set_ylim(1e-2,2*maxval)
        delmomlog.semilogy(dmommid, fxn_ConvCrystalBall(dmommid, *refparnomat), 'r-',label="Fit")
        self.HDeltaNoMatMom.plotErrors(delmomlog)
        delselmomlog.set_yscale("log")
        delselmomlog.set_ylim(1e-2,2*maxval)
        delselmomlog.semilogy(dmommid, fxn_ConvCrystalBall(dmommid, *refpartgt), 'r-',label="Fit")
        self.HDeltaTgtMom.plotErrors(delselmomlog)

    def FitExpGauss(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = self.HCeRefResp.binErrors()
        dmommid = self.HCeRefResp.binCenters()
        dmomsum = self.HCeRefResp.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HCeRefResp.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HCeRefResp.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HCeRefResp.edges[1]- self.HCeRefResp.edges[0]
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
