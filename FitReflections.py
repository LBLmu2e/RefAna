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
import HistUtil
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

        self.HDeltaTgtMomB12 = HistUtil.load_hist(name="DeltaMom",label="B12",file=reffile)
        self.HDeltaTgtMomB34 = HistUtil.load_hist(name="DeltaMom",label="B34",file=reffile)
        self.HDeltaTgtMomB56 = HistUtil.load_hist(name="DeltaMom",label="B56",file=reffile)
        self.HDeltaTgtMomB78 = HistUtil.load_hist(name="DeltaMom",label="B78",file=reffile)
        self.HDeltaTgtMomB9p = HistUtil.load_hist(name="DeltaMom",label="B9p",file=reffile)
        self.HDeltaNoMatMom = HistUtil.load_hist(name="DeltaMom",label="No Material",file=reffile)
        self.HDeltaTgtMom = HistUtil.load_hist(name="DeltaMom",label="$N_{ST}$>0",file=reffile)
        self.HDeltaNoMatMom.title = "Reflected " + self.HDeltaNoMatMom.title
        self.HDeltaTgtMom.title = "Reflected " + self.HDeltaTgtMom.title
        self.HDeltaTgtMomB12.title = "Reflected " + self.HDeltaTgtMomB12.title
        self.HDeltaTgtMomB34.title = "Reflected " + self.HDeltaTgtMomB34.title
        self.HDeltaTgtMomB56.title = "Reflected " + self.HDeltaTgtMomB56.title
        self.HDeltaTgtMomB78.title = "Reflected " + self.HDeltaTgtMomB78.title
        self.HDeltaTgtMomB9p.title = "Reflected " + self.HDeltaTgtMomB9p.title
        self.hasCe = ( cefile != None)
        if self.hasCe:
            loc = SID.SurfaceName(SID.TT_Front())
            self.HTrkRefRespMom = HistUtil.load_hist(name=loc+"Response",label="Reflectable",file=cefile)
            self.HTrkRefRespMom.title = "Ce " + self.HTrkRefRespMom.title
            self.HTrkRefRespMom.label = "$N_{TSDA}$==0"
            self.HTrkResoMom = HistUtil.load_hist(name=loc+"Resolution",label="",file=cefile)
            self.HTrkResoMom.title = "Ce " + self.HTrkResoMom.title
            self.HTrkResoMom.label = "$N_{TSDA}$==0"

            ### BINNED ###
            self.HCeTgtMomB12 = HistUtil.load_hist(name="DMom",label="B12", file=cefile)
            self.HCeTgtMomB34 = HistUtil.load_hist(name="DMom",label="B34", file=cefile)
            self.HCeTgtMomB56 = HistUtil.load_hist(name="DMom",label="B56", file=cefile)
            self.HCeTgtMomB78 = HistUtil.load_hist(name="DMom",label="B78", file=cefile)
            self.HCeTgtMomB9p = HistUtil.load_hist(name="DMom",label="B9p", file=cefile)

            self.HCeTgtMomB12.title = "Ce " + self.HDeltaTgtMomB12.title
            self.HCeTgtMomB34.title = "Ce " + self.HDeltaTgtMomB34.title
            self.HCeTgtMomB56.title = "Ce " + self.HDeltaTgtMomB56.title
            self.HCeTgtMomB78.title = "Ce " + self.HDeltaTgtMomB78.title
            self.HCeTgtMomB9p.title = "Ce " + self.HDeltaTgtMomB9p.title

    def TestExpGauss(self):
        dmomerr = HistUtil.bin_errors(self.HDeltaNoMatMom)
        dmommid = HistUtil.bin_centers(self.HDeltaNoMatMom)
        dmomsum = HistUtil.integral(self.HDeltaNoMatMom)
        binsize = self.HDeltaNoMatMom.axes[0].edges[1]- self.HDeltaNoMatMom.axes[0].edges[0]
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.view()/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.view())/dmomsum) - mu_0**2
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
        dmomerr = HistUtil.bin_errors(self.HDeltaNoMatMom)
        dmommid = HistUtil.bin_centers(self.HDeltaNoMatMom)
        dmomsum = HistUtil.integral(self.HDeltaNoMatMom)
        binsize = self.HDeltaNoMatMom.axes[0].edges[1]- self.HDeltaNoMatMom.axes[0].edges[0]
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
        convhist = HistUtil.new_hist(name="convhist",label="Manually Convolved CB",bins=200,range=[-10.0,5],xlabel="$\\Delta$ Momentum")
        convhist.fill(rconv)
        HistUtil.plot(convhist,afit)
        # then fit
        dmomerr = HistUtil.bin_errors(convhist)
        dmommid = HistUtil.bin_centers(convhist)
        dmomsum = HistUtil.integral(convhist)
        binsize = convhist.axes[0].edges[1]- convhist.axes[0].edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        refpartest, refcovtest = curve_fit(fxn_ConvCrystalBall, dmommid, convhist.view(), p0, sigma=dmomerr)
        refperrtest = np.sqrt(np.diagonal(refcovtest))
#        HistUtil.plot_errors(convhist,afit)
        maxval = np.amax(convhist.view())
        afit.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refpartest), 'r-',label="Convolved CB Fit")
        afit.legend(loc="upper right")
        afit.text(-8, 0.8*maxval, f"$\\beta$ = {refpartest[1]:.3f} $\\pm$ {refperrtest[1]:.3f}")
        afit.text(-8, 0.7*maxval, f"m = {refpartest[2]:.3f} $\\pm$ {refperrtest[2]:.3f}")
        afit.text(-8, 0.6*maxval,  f"loc = {refpartest[3]:.3f} $\\pm$ {refperrtest[3]:.3f}")
        afit.text(-8, 0.5*maxval,  f"scale = {refpartest[4]:.3f} $\\pm$ {refperrtest[4]:.3f}")

    def FitCrystalBall(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = HistUtil.bin_errors(self.HDeltaNoMatMom)
        dmommid = HistUtil.bin_centers(self.HDeltaNoMatMom)
        dmomsum = HistUtil.integral(self.HDeltaNoMatMom)
        binsize = self.HDeltaNoMatMom.binWidth()
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaNoMatMom.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaNoMatMom.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        HistUtil.plot_errors(self.HDeltaNoMatMom,delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")

        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMom)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMom)
        dmomsum = HistUtil.integral(self.HDeltaTgtMom)
        binsize = self.HDeltaTgtMom.axes[0].edges[1]- self.HDeltaTgtMom.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMom.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.5
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0,beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMom.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        HistUtil.plot_errors(self.HDeltaTgtMom,delselmom)
        delselmom.plot(dmommid, fxn_CrystalBall(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")

        fig.text(0.6, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.6, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.6, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.6, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        """ BINS """

        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        # B12

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMomB12)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMomB12)
        dmomsum = HistUtil.integral(self.HDeltaTgtMomB12)
        binsize = self.HDeltaTgtMomB12.axes[0].edges[1]- self.HDeltaTgtMomB12.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB12.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB12.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        HistUtil.plot_errors(self.HDeltaTgtMomB12,delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")

        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B34

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMomB34)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMomB34)
        dmomsum = HistUtil.integral(self.HDeltaTgtMomB34)
        binsize = self.HDeltaTgtMomB34.axes[0].edges[1]- self.HDeltaTgtMomB34.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB34.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB34.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        HistUtil.plot_errors(self.HDeltaTgtMomB34,delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")

        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B56

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMomB56)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMomB56)
        dmomsum = HistUtil.integral(self.HDeltaTgtMomB56)
        binsize = self.HDeltaTgtMomB56.axes[0].edges[1]- self.HDeltaTgtMomB56.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB56.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB56.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        HistUtil.plot_errors(self.HDeltaTgtMomB56,delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")

        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B78

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMomB78)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMomB78)
        dmomsum = HistUtil.integral(self.HDeltaTgtMomB78)
        binsize = self.HDeltaTgtMomB78.axes[0].edges[1]- self.HDeltaTgtMomB78.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB78.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB78.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        HistUtil.plot_errors(self.HDeltaTgtMomB78,delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")

        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")

        # B9p

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMomB9p)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMomB9p)
        dmomsum = HistUtil.integral(self.HDeltaTgtMomB9p)
        binsize = self.HDeltaTgtMomB9p.axes[0].edges[1]- self.HDeltaTgtMomB9p.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMomB9p.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaTgtMomB9p.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        perr = np.sqrt(np.diagonal(pcov))

        HistUtil.plot_errors(self.HDeltaTgtMomB9p,delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")

        fig.text(0.1, 0.8, f"$\\beta$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f}")
        fig.text(0.1, 0.7, f"m = {popt[2]:.3f} $\\pm$ {perr[2]:.3f}")
        fig.text(0.1, 0.6,  f"loc = {popt[3]:.3f} $\\pm$ {perr[3]:.3f}")
        fig.text(0.1, 0.5,  f"scale = {popt[4]:.3f} $\\pm$ {perr[4]:.3f}")


    def FitConvCrystalBall(self): # steal normalization from fit
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(15,5))

        dmomerr = HistUtil.bin_errors(self.HDeltaNoMatMom)
        dmommid = HistUtil.bin_centers(self.HDeltaNoMatMom)
        dmomsum = HistUtil.integral(self.HDeltaNoMatMom)
        binsize = self.HDeltaNoMatMom.axes[0].edges[1]- self.HDeltaNoMatMom.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaNoMatMom.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.20
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat, refcovnomat = curve_fit(fxn_ConvCrystalBall, dmommid, self.HDeltaNoMatMom.view(), p0, sigma=dmomerr)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat = np.sqrt(np.diagonal(refcovnomat))

        HistUtil.plot_errors(self.HDeltaNoMatMom,delmom)
        maxval = np.amax(self.HDeltaNoMatMom.view())
        delmom.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refparnomat), 'r-',label="Conv. CB Fit")
        delmom.legend(loc="upper right")
        delmom.text(-8, 0.8*maxval, f"$\\beta$ = {refparnomat[1]:.3f} $\\pm$ {refperrnomat[1]:.3f}")
        delmom.text(-8, 0.7*maxval, f"m = {refparnomat[2]:.3f} $\\pm$ {refperrnomat[2]:.3f}")
        delmom.text(-8, 0.6*maxval,  f"loc = {refparnomat[3]:.3f} $\\pm$ {refperrnomat[3]:.3f}")
        delmom.text(-8, 0.5*maxval,  f"scale = {refparnomat[4]:.3f} $\\pm$ {refperrnomat[4]:.3f}")

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMom)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMom)
        dmomsum = HistUtil.integral(self.HDeltaTgtMom)
        binsize = self.HDeltaTgtMom.axes[0].edges[1]- self.HDeltaTgtMom.axes[0].edges[0]
        # initialize the fit parameters
        loc_0 = np.mean(dmommid*self.HDeltaTgtMom.view()/dmomsum) # initial mean
        beta_0 = 1.0
        m_0 = 3.0
        scale_0 = 0.5
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0,beta_0, m_0, loc_0, scale_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        refpartgt, refcovtgt = curve_fit(fxn_ConvCrystalBall, dmommid, self.HDeltaTgtMom.view(), p0, sigma=dmomerr)
#        print("Target inter fit parameters",refpartgt)
#        print("Target inter fit covariance",refcovtgt)
        refperrtgt = np.sqrt(np.diagonal(refcovtgt))

        HistUtil.plot_errors(self.HDeltaTgtMom,delselmom)
        maxval = np.amax(self.HDeltaTgtMom.view())
        delselmom.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refpartgt), 'r-',label="Conv. CB Fit")
        delselmom.legend(loc="upper right")
        delselmom.text(-8, 0.8*maxval, f"$\\beta$ = {refpartgt[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
        delselmom.text(-8, 0.7*maxval, f"m = {refpartgt[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
        delselmom.text(-8, 0.6*maxval,  f"loc = {refpartgt[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
        delselmom.text(-8, 0.5*maxval,  f"scale = {refpartgt[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

        """ Binned """

        fig, (delmomB12,delmomB34,delmomB56,delmomB78,delmomB9p) = plt.subplots(1,5,layout='constrained', figsize=(15,5))

        # B12

        dmomerr12 = HistUtil.bin_errors(self.HDeltaTgtMomB12)
        dmommid12 = HistUtil.bin_centers(self.HDeltaTgtMomB12)
        dmomsum12 = HistUtil.integral(self.HDeltaTgtMomB12)
        binsize12 = self.HDeltaTgtMomB12.axes[0].edges[1]- self.HDeltaTgtMomB12.axes[0].edges[0]
        # initialize the fit parameters
        loc_012 = np.mean(dmommid12*self.HDeltaTgtMomB12.view()/dmomsum12) # initial mean
        beta_012 = 1.0
        m_012 = 3.0
        scale_012 = 0.20
        amp_012 = dmomsum12*binsize12 # initial amplitude
        p012 = np.array([amp_012, beta_012, m_012, loc_012, scale_012]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat12, refcovnomat12 = curve_fit(fxn_ConvCrystalBall, dmommid12, self.HDeltaTgtMomB12.view(), p012, sigma=dmomerr12)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat12 = np.sqrt(np.diagonal(refcovnomat12))

        HistUtil.plot_errors(self.HDeltaTgtMomB12,delmomB12)
        maxval12 = np.amax(self.HDeltaTgtMomB12.view())
        delmomB12.plot(dmommid12, fxn_ConvCrystalBall(dmommid12, *refparnomat12), 'r-',label="Fit")
        delmomB12.legend(loc="upper right")
        delmomB12.text(-8, 0.8*maxval12, f"$\\beta$ = {refparnomat12[1]:.3f} $\\pm$ {refperrnomat12[1]:.3f}")
        delmomB12.text(-8, 0.7*maxval12, f"m = {refparnomat12[2]:.3f} $\\pm$ {refperrnomat12[2]:.3f}")
        delmomB12.text(-8, 0.6*maxval12,  f"loc = {refparnomat12[3]:.3f} $\\pm$ {refperrnomat12[3]:.3f}")
        delmomB12.text(-8, 0.5*maxval12,  f"scale = {refparnomat12[4]:.3f} $\\pm$ {refperrnomat12[4]:.3f}")

        # B34

        dmomerr34 = HistUtil.bin_errors(self.HDeltaTgtMomB34)
        dmommid34 = HistUtil.bin_centers(self.HDeltaTgtMomB34)
        dmomsum34 = HistUtil.integral(self.HDeltaTgtMomB34)
        binsize34 = self.HDeltaTgtMomB34.axes[0].edges[1]- self.HDeltaTgtMomB34.axes[0].edges[0]
        # initialize the fit parameters
        loc_034 = np.mean(dmommid34*self.HDeltaTgtMomB34.view()/dmomsum34) # initial mean
        beta_034 = 1.0
        m_034 = 3.0
        scale_034 = 0.20
        amp_034 = dmomsum34*binsize34 # initial amplitude
        p034 = np.array([amp_034, beta_034, m_034, loc_034, scale_034]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat34, refcovnomat34 = curve_fit(fxn_ConvCrystalBall, dmommid34, self.HDeltaTgtMomB34.view(), p034, sigma=dmomerr34)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat34 = np.sqrt(np.diagonal(refcovnomat34))

        HistUtil.plot_errors(self.HDeltaTgtMomB34,delmomB34)
        maxval34 = np.amax(self.HDeltaTgtMomB34.view())
        delmomB34.plot(dmommid34, fxn_ConvCrystalBall(dmommid34, *refparnomat34), 'r-',label="Fit")
        delmomB34.legend(loc="upper right")
        delmomB34.text(-8, 0.8*maxval34, f"$\\beta$ = {refparnomat34[1]:.3f} $\\pm$ {refperrnomat34[1]:.3f}")
        delmomB34.text(-8, 0.7*maxval34, f"m = {refparnomat34[2]:.3f} $\\pm$ {refperrnomat34[2]:.3f}")
        delmomB34.text(-8, 0.6*maxval34,  f"loc = {refparnomat34[3]:.3f} $\\pm$ {refperrnomat34[3]:.3f}")
        delmomB34.text(-8, 0.5*maxval34,  f"scale = {refparnomat34[4]:.3f} $\\pm$ {refperrnomat34[4]:.3f}")

        # B56

        dmomerr56 = HistUtil.bin_errors(self.HDeltaTgtMomB56)
        dmommid56 = HistUtil.bin_centers(self.HDeltaTgtMomB56)
        dmomsum56 = HistUtil.integral(self.HDeltaTgtMomB56)
        binsize56 = self.HDeltaTgtMomB56.axes[0].edges[1]- self.HDeltaTgtMomB56.axes[0].edges[0]
        # initialize the fit parameters
        loc_056 = np.mean(dmommid56*self.HDeltaTgtMomB56.view()/dmomsum56) # initial mean
        beta_056 = 1.0
        m_056 = 3.0
        scale_056 = 0.20
        amp_056 = dmomsum56*binsize56 # initial amplitude
        p056 = np.array([amp_056, beta_056, m_056, loc_056, scale_056]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat56, refcovnomat56 = curve_fit(fxn_ConvCrystalBall, dmommid56, self.HDeltaTgtMomB56.view(), p056, sigma=dmomerr56)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat56 = np.sqrt(np.diagonal(refcovnomat56))

        HistUtil.plot_errors(self.HDeltaTgtMomB56,delmomB56)
        maxval56 = np.amax(self.HDeltaTgtMomB56.view())
        delmomB56.plot(dmommid56, fxn_ConvCrystalBall(dmommid56, *refparnomat56), 'r-',label="Fit")
        delmomB56.legend(loc="upper right")
        delmomB56.text(-8, 0.8*maxval56, f"$\\beta$ = {refparnomat56[1]:.3f} $\\pm$ {refperrnomat56[1]:.3f}")
        delmomB56.text(-8, 0.7*maxval56, f"m = {refparnomat56[2]:.3f} $\\pm$ {refperrnomat56[2]:.3f}")
        delmomB56.text(-8, 0.6*maxval56,  f"loc = {refparnomat56[3]:.3f} $\\pm$ {refperrnomat56[3]:.3f}")
        delmomB56.text(-8, 0.5*maxval56,  f"scale = {refparnomat56[4]:.3f} $\\pm$ {refperrnomat56[4]:.3f}")

        # B78

        dmomerr78 = HistUtil.bin_errors(self.HDeltaTgtMomB78)
        dmommid78 = HistUtil.bin_centers(self.HDeltaTgtMomB78)
        dmomsum78 = HistUtil.integral(self.HDeltaTgtMomB78)
        binsize78 = self.HDeltaTgtMomB78.axes[0].edges[1]- self.HDeltaTgtMomB78.axes[0].edges[0]
        # initialize the fit parameters
        loc_078 = np.mean(dmommid78*self.HDeltaTgtMomB78.view()/dmomsum78) # initial mean
        beta_078 = 1.0
        m_078 = 3.0
        scale_078 = 0.20
        amp_078 = dmomsum78*binsize78 # initial amplitude
        p078 = np.array([amp_078, beta_078, m_078, loc_078, scale_078]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat78, refcovnomat78 = curve_fit(fxn_ConvCrystalBall, dmommid78, self.HDeltaTgtMomB78.view(), p078, sigma=dmomerr78)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat78 = np.sqrt(np.diagonal(refcovnomat78))

        HistUtil.plot_errors(self.HDeltaTgtMomB78,delmomB78)
        maxval78 = np.amax(self.HDeltaTgtMomB78.view())
        delmomB78.plot(dmommid78, fxn_ConvCrystalBall(dmommid78, *refparnomat78), 'r-',label="Fit")
        delmomB78.legend(loc="upper right")
        delmomB78.text(-8, 0.8*maxval78, f"$\\beta$ = {refparnomat78[1]:.3f} $\\pm$ {refperrnomat78[1]:.3f}")
        delmomB78.text(-8, 0.7*maxval78, f"m = {refparnomat78[2]:.3f} $\\pm$ {refperrnomat78[2]:.3f}")
        delmomB78.text(-8, 0.6*maxval78,  f"loc = {refparnomat78[3]:.3f} $\\pm$ {refperrnomat78[3]:.3f}")
        delmomB78.text(-8, 0.5*maxval78,  f"scale = {refparnomat78[4]:.3f} $\\pm$ {refperrnomat78[4]:.3f}")

        # B9p

        dmomerr9p = HistUtil.bin_errors(self.HDeltaTgtMomB9p)
        dmommid9p = HistUtil.bin_centers(self.HDeltaTgtMomB9p)
        dmomsum9p = HistUtil.integral(self.HDeltaTgtMomB9p)
        binsize9p = self.HDeltaTgtMomB9p.axes[0].edges[1]- self.HDeltaTgtMomB9p.axes[0].edges[0]
        # initialize the fit parameters
        loc_09p = np.mean(dmommid9p*self.HDeltaTgtMomB9p.view()/dmomsum9p) # initial mean
        beta_09p = 1.0
        m_09p = 3.0
        scale_09p = 0.20
        amp_09p = dmomsum9p*binsize9p # initial amplitude
        p09p = np.array([amp_09p, beta_09p, m_09p, loc_09p, scale_09p]) # initial parameters
        # fit, returing optimum parameters and covariance
        refparnomat9p, refcovnomat9p = curve_fit(fxn_ConvCrystalBall, dmommid9p, self.HDeltaTgtMomB9p.view(), p09p, sigma=dmomerr9p)
#        print("No Material fit parameters",refparnomat)
#        print("No Material fit covariance",refcovnomat)
        refperrnomat9p = np.sqrt(np.diagonal(refcovnomat9p))

        HistUtil.plot_errors(self.HDeltaTgtMomB9p,delmomB9p)
        maxval9p = np.amax(self.HDeltaTgtMomB9p.view())
        delmomB9p.plot(dmommid9p, fxn_ConvCrystalBall(dmommid9p, *refparnomat9p), 'r-',label="Fit")
        delmomB9p.legend(loc="upper right")
        delmomB9p.text(-8, 0.8*maxval9p, f"$\\beta$ = {refparnomat9p[1]:.3f} $\\pm$ {refperrnomat9p[1]:.3f}")
        delmomB9p.text(-8, 0.7*maxval9p, f"m = {refparnomat9p[2]:.3f} $\\pm$ {refperrnomat9p[2]:.3f}")
        delmomB9p.text(-8, 0.6*maxval9p,  f"loc = {refparnomat9p[3]:.3f} $\\pm$ {refperrnomat9p[3]:.3f}")
        delmomB9p.text(-8, 0.5*maxval9p,  f"scale = {refparnomat9p[4]:.3f} $\\pm$ {refperrnomat9p[4]:.3f}")

        if self.hasCe:
            fig, (cefitreso,cefitresp) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            resobins = HistUtil.bin_centers(self.HTrkResoMom)
            resoint = HistUtil.integral(self.HTrkResoMom)
            resobinsize = self.HTrkResoMom.axes[0].edges[1]- self.HTrkResoMom.axes[0].edges[0]
            loc_0 = np.mean(resobins*self.HTrkResoMom.view()/resoint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = resoint*resobinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit Ce resolution
            resofitpars, resofitcov = curve_fit(fxn_CrystalBall, resobins, self.HTrkResoMom.view(), p0, sigma=dmomerr)
            resoperr = np.sqrt(np.diagonal(resofitcov))
            HistUtil.plot_errors(self.HTrkResoMom,cefitreso)
            maxval = np.amax(self.HTrkResoMom.view())
            cefitreso.plot(resobins, fxn_CrystalBall(resobins, *resofitpars), 'r-',label="CB Fit")
            cefitreso.text(-2, 0.8*maxval, f"$\\beta$ = {resofitpars[1]:.3f} $\\pm$ {resoperr[1]:.3f}")
            cefitreso.text(-2, 0.7*maxval, f"m = {resofitpars[2]:.3f} $\\pm$ {resoperr[2]:.3f}")
            cefitreso.text(-2, 0.6*maxval,  f"loc = {resofitpars[3]:.3f} $\\pm$ {resoperr[3]:.3f}")
            cefitreso.text(-2, 0.5*maxval,  f"scale = {resofitpars[4]:.3f} $\\pm$ {resoperr[4]:.3f}")
            cefitreso.legend(loc="upper right")
            # fit Ce response
            respbins = HistUtil.bin_centers(self.HTrkRefRespMom)
            respint = HistUtil.integral(self.HTrkRefRespMom)
            respbinsize = self.HTrkRefRespMom.axes[0].edges[1]- self.HTrkRefRespMom.axes[0].edges[0]
            amp_0 = respint*respbinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0])
            respfitpars, respfitcov = curve_fit(fxn_CrystalBall, respbins, self.HTrkRefRespMom.view(), p0, sigma=dmomerr)
            respperr = np.sqrt(np.diagonal(respfitcov))
            HistUtil.plot_errors(self.HTrkRefRespMom,cefitresp)
            maxval = np.amax(self.HTrkRefRespMom.view())
            cefitresp.plot(respbins, fxn_CrystalBall(respbins, *respfitpars), 'r-',label="CB Fit")
            cefitresp.text(-8, 0.8*maxval, f"$\\beta$ = {respfitpars[1]:.3f} $\\pm$ {respperr[1]:.3f}")
            cefitresp.text(-8, 0.7*maxval, f"m = {respfitpars[2]:.3f} $\\pm$ {respperr[2]:.3f}")
            cefitresp.text(-8, 0.6*maxval,  f"loc = {respfitpars[3]:.3f} $\\pm$ {respperr[3]:.3f}")
            cefitresp.text(-8, 0.5*maxval,  f"scale = {respfitpars[4]:.3f} $\\pm$ {respperr[4]:.3f}")
            cefitresp.legend(loc="upper right")

            # Plot overlay with the reflection fit results. Adjust the amplitude
            fig, (cecompreso,cecompresp) = plt.subplots(1,2,layout='constrained', figsize=(15,5))

            HistUtil.plot_errors(self.HTrkResoMom,cecompreso)
            resocomppars = copy.deepcopy(refparnomat)
            resocomppars[0] = resofitpars[0] # steal normalization from fit
            maxval = np.amax(self.HTrkResoMom.view())
            cecompreso.plot(resobins, fxn_CrystalBall(resobins, *resocomppars), 'r-',label="Deconv. Ref. CB")
            cecompreso.text(-2, 0.8*maxval, f"$\\beta$ = {resocomppars[1]:.3f} $\\pm$ {refperrnomat[1]:.3f}")
            cecompreso.text(-2, 0.7*maxval, f"m = {resocomppars[2]:.3f} $\\pm$ {refperrnomat[2]:.3f}")
            cecompreso.text(-2, 0.6*maxval,  f"loc = {resocomppars[3]:.3f} $\\pm$ {refperrnomat[3]:.3f}")
            cecompreso.text(-2, 0.5*maxval,  f"scale = {resocomppars[4]:.3f} $\\pm$ {refperrnomat[4]:.3f}")
            cecompreso.legend(loc="upper right")

            HistUtil.plot_errors(self.HTrkRefRespMom,cecompresp)
            respcomppars = copy.deepcopy(refpartgt)
            respcomppars[0] = respfitpars[0] # steal normalization from fit
            cecompresp.plot(respbins, fxn_CrystalBall(respbins, *respcomppars), 'r-',label="Deconv. Ref. CB")
            maxval = np.amax(self.HTrkRefRespMom.view())
            cecompresp.text(-8, 0.8*maxval, f"$\\beta$ = {respcomppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            cecompresp.text(-8, 0.7*maxval, f"m = {respcomppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            cecompresp.text(-8, 0.6*maxval,  f"loc = {respcomppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            cecompresp.text(-8, 0.5*maxval,  f"scale = {respcomppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")
            cecompresp.legend(loc="upper right")

            ### BINNED ###

            fig, (aB12, aB34, aB56, aB78, aB9p) = plt.subplots(1,5,layout='constrained', figsize=(15,5))

            # B12

            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            cebins = HistUtil.bin_centers(self.HCeTgtMomB12)
            ceint = HistUtil.integral(self.HCeTgtMomB12)
            cebinsize = self.HCeTgtMomB12.axes[0].edges[1]- self.HCeTgtMomB12.axes[0].edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB12.view()/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB12.view(), p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            HistUtil.plot_errors(self.HCeTgtMomB12,cefitallresp)
            maxval = np.amax(self.HCeTgtMomB12.view())
            aB12.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB12.legend(loc="upper right")
            aB12.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB12.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB12.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB12.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            HistUtil.plot_errors(self.HCeTgtMomB12,cecompallresp)
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
            cebins = HistUtil.bin_centers(self.HCeTgtMomB34)
            ceint = HistUtil.integral(self.HCeTgtMomB34)
            cebinsize = self.HCeTgtMomB34.axes[0].edges[1]- self.HCeTgtMomB34.axes[0].edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB34.view()/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB34.view(), p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            HistUtil.plot_errors(self.HCeTgtMomB34,cefitallresp)
            maxval = np.amax(self.HCeTgtMomB34.view())
            aB34.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB34.legend(loc="upper right")
            aB34.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB34.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB34.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB34.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            HistUtil.plot_errors(self.HCeTgtMomB34,cecompallresp)
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
            cebins = HistUtil.bin_centers(self.HCeTgtMomB56)
            ceint = HistUtil.integral(self.HCeTgtMomB56)
            cebinsize = self.HCeTgtMomB56.axes[0].edges[1]- self.HCeTgtMomB56.axes[0].edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB56.view()/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB56.view(), p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            HistUtil.plot_errors(self.HCeTgtMomB56,cefitallresp)
            maxval = np.amax(self.HCeTgtMomB56.view())
            aB56.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB56.legend(loc="upper right")
            aB56.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB56.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB56.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB56.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            HistUtil.plot_errors(self.HCeTgtMomB56,cecompallresp)
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
            cebins = HistUtil.bin_centers(self.HCeTgtMomB78)
            ceint = HistUtil.integral(self.HCeTgtMomB78)
            cebinsize = self.HCeTgtMomB78.axes[0].edges[1]- self.HCeTgtMomB78.axes[0].edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB78.view()/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB78.view(), p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            HistUtil.plot_errors(self.HCeTgtMomB78,cefitallresp)
            maxval = np.amax(self.HCeTgtMomB78.view())
            aB78.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB78.legend(loc="upper right")
            aB78.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB78.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB78.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB78.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            HistUtil.plot_errors(self.HCeTgtMomB78,cecompallresp)
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
            cebins = HistUtil.bin_centers(self.HCeTgtMomB9p)
            ceint = HistUtil.integral(self.HCeTgtMomB9p)
            cebinsize = self.HCeTgtMomB9p.axes[0].edges[1]- self.HCeTgtMomB9p.axes[0].edges[0]
            loc_0 = np.mean(cebins*self.HCeTgtMomB9p.view()/ceint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = ceint*cebinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit, returing optimum parameters and covariance
            cepars, cecov = curve_fit(fxn_CrystalBall, cebins, self.HCeTgtMomB9p.view(), p0, sigma=dmomerr)
            print("All fit parameters",cepars)
            print("All fit covariance",cecov)
            ceperr = np.sqrt(np.diagonal(cecov))
            HistUtil.plot_errors(self.HCeTgtMomB9p,cefitallresp)
            maxval = np.amax(self.HCeTgtMomB9p.view())
            aB9p.plot(cebins, fxn_CrystalBall(cebins, *cepars), 'r-',label="Fit")
            aB9p.legend(loc="upper right")
            aB9p.text(-8, 0.8*maxval, f"$\\beta$ = {cepars[1]:.3f} $\\pm$ {ceperr[1]:.3f}")
            aB9p.text(-8, 0.7*maxval, f"m = {cepars[2]:.3f} $\\pm$ {ceperr[2]:.3f}")
            aB9p.text(-8, 0.6*maxval,  f"loc = {cepars[3]:.3f} $\\pm$ {ceperr[3]:.3f}")
            aB9p.text(-8, 0.5*maxval,  f"scale = {cepars[4]:.3f} $\\pm$ {ceperr[4]:.3f}")
            # Plot overlay with the reflection fit results. Adjust the amplitude
            HistUtil.plot_errors(self.HCeTgtMomB9p,cecompallresp)
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
        HistUtil.plot_errors(self.HDeltaNoMatMom,delmomlog)
        delselmomlog.set_yscale("log")
        delselmomlog.set_ylim(1e-2,2*maxval)
        delselmomlog.semilogy(dmommid, fxn_ConvCrystalBall(dmommid, *refpartgt), 'r-',label="Fit")
        HistUtil.plot_errors(self.HDeltaTgtMom,delselmomlog)

    def FitExpGauss(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = HistUtil.bin_errors(self.HCeRefResp)
        dmommid = HistUtil.bin_centers(self.HCeRefResp)
        dmomsum = HistUtil.integral(self.HCeRefResp)
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HCeRefResp.view()/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HCeRefResp.view())/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HCeRefResp.axes[0].edges[1]- self.HCeRefResp.axes[0].edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ExpGauss, dmommid, self.HDeltaNoMatMom.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        HistUtil.plot_errors(self.HDeltaNoMatMom,delmom)
        delmom.plot(dmommid, fxn_ExpGauss(dmommid, *popt), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMom)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMom)
        dmomsum = HistUtil.integral(self.HDeltaTgtMom)
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaTgtMom.view()/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaTgtMom.view())/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaTgtMom.axes[0].edges[1]- self.HDeltaTgtMom.axes[0].edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ExpGauss, dmommid, self.HDeltaTgtMom.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        HistUtil.plot_errors(self.HDeltaTgtMom,delselmom)
        delselmom.plot(dmommid, fxn_ExpGauss(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

    def FitConvExpGauss(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = HistUtil.bin_errors(self.HDeltaNoMatMom)
        dmommid = HistUtil.bin_centers(self.HDeltaNoMatMom)
        dmomsum = HistUtil.integral(self.HDeltaNoMatMom)
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.view()/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.view())/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaNoMatMom.axes[0].edges[1]- self.HDeltaNoMatMom.axes[0].edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ConvExpGauss, dmommid, self.HDeltaNoMatMom.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        HistUtil.plot_errors(self.HDeltaNoMatMom,delmom)
        delmom.plot(dmommid, fxn_ConvExpGauss(dmommid, *popt), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

        dmomerr = HistUtil.bin_errors(self.HDeltaTgtMom)
        dmommid = HistUtil.bin_centers(self.HDeltaTgtMom)
        dmomsum = HistUtil.integral(self.HDeltaTgtMom)
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaTgtMom.view()/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaTgtMom.view())/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaTgtMom.axes[0].edges[1]- self.HDeltaTgtMom.axes[0].edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_ConvExpGauss, dmommid, self.HDeltaTgtMom.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        HistUtil.plot_errors(self.HDeltaTgtMom,delselmom)
        delselmom.plot(dmommid, fxn_ConvExpGauss(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")
