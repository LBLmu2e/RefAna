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
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Material",file=reffile)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="$N_{ST}$>0",file=reffile)
        self.HDeltaNoMatMom.title = "Reflected " + self.HDeltaNoMatMom.title
        self.HDeltaTgtMom.title = "Reflected " + self.HDeltaTgtMom.title
        self.hasCe = ( cefile != None)
        if self.hasCe:
            loc = SID.SurfaceName(SID.TT_Front())
            self.HTrkRefRespMom = MyHist.MyHist(name=loc+"Response",label="Reflectable",file=cefile)
            self.HTrkRefRespMom.title = "Ce " + self.HTrkRefRespMom.title
            self.HTrkRefRespMom.label = "$N_{TSDA}$==0"
            self.HTrkResoMom = MyHist.MyHist(name=loc+"Resolution",label="",file=cefile)
            self.HTrkResoMom.title = "Ce " + self.HTrkResoMom.title
            self.HTrkResoMom.label = "$N_{TSDA}$==0"

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
        delmom.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refparnomat), 'r-',label="Conv. CB Fit")
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
        delselmom.plot(dmommid, fxn_ConvCrystalBall(dmommid, *refpartgt), 'r-',label="Conv. CB Fit")
        delselmom.legend(loc="upper right")
        delselmom.text(-8, 0.8*maxval, f"$\\beta$ = {refpartgt[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
        delselmom.text(-8, 0.7*maxval, f"m = {refpartgt[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
        delselmom.text(-8, 0.6*maxval,  f"loc = {refpartgt[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
        delselmom.text(-8, 0.5*maxval,  f"scale = {refpartgt[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")

        if self.hasCe:
            fig, (cefitreso,cefitresp) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
            # fit to un-convolved Crystal Ball
            # initialize the fit parameters
            resobins = self.HTrkResoMom.binCenters()
            resoint = self.HTrkResoMom.integral()
            resobinsize = self.HTrkResoMom.edges[1]- self.HTrkResoMom.edges[0]
            loc_0 = np.mean(resobins*self.HTrkResoMom.data/resoint) # initial mean
            beta_0 = 1.0
            m_0 = 3.0
            scale_0 = 0.20
            amp_0 = resoint*resobinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
            # fit Ce resolution
            resofitpars, resofitcov = curve_fit(fxn_CrystalBall, resobins, self.HTrkResoMom.data, p0, sigma=dmomerr)
            resoperr = np.sqrt(np.diagonal(resofitcov))
            self.HTrkResoMom.plotErrors(cefitreso)
            maxval = np.amax(self.HTrkResoMom.data)
            cefitreso.plot(resobins, fxn_CrystalBall(resobins, *resofitpars), 'r-',label="CB Fit")
            cefitreso.text(-2, 0.8*maxval, f"$\\beta$ = {resofitpars[1]:.3f} $\\pm$ {resoperr[1]:.3f}")
            cefitreso.text(-2, 0.7*maxval, f"m = {resofitpars[2]:.3f} $\\pm$ {resoperr[2]:.3f}")
            cefitreso.text(-2, 0.6*maxval,  f"loc = {resofitpars[3]:.3f} $\\pm$ {resoperr[3]:.3f}")
            cefitreso.text(-2, 0.5*maxval,  f"scale = {resofitpars[4]:.3f} $\\pm$ {resoperr[4]:.3f}")
            cefitreso.legend(loc="upper right")
            # fit Ce response
            respbins = self.HTrkRefRespMom.binCenters()
            respint = self.HTrkRefRespMom.integral()
            respbinsize = self.HTrkRefRespMom.edges[1]- self.HTrkRefRespMom.edges[0]
            amp_0 = respint*respbinsize # initial amplitude
            p0 = np.array([amp_0, beta_0, m_0, loc_0, scale_0])
            respfitpars, respfitcov = curve_fit(fxn_CrystalBall, respbins, self.HTrkRefRespMom.data, p0, sigma=dmomerr)
            respperr = np.sqrt(np.diagonal(respfitcov))
            self.HTrkRefRespMom.plotErrors(cefitresp)
            maxval = np.amax(self.HTrkRefRespMom.data)
            cefitresp.plot(respbins, fxn_CrystalBall(respbins, *respfitpars), 'r-',label="CB Fit")
            cefitresp.text(-8, 0.8*maxval, f"$\\beta$ = {respfitpars[1]:.3f} $\\pm$ {respperr[1]:.3f}")
            cefitresp.text(-8, 0.7*maxval, f"m = {respfitpars[2]:.3f} $\\pm$ {respperr[2]:.3f}")
            cefitresp.text(-8, 0.6*maxval,  f"loc = {respfitpars[3]:.3f} $\\pm$ {respperr[3]:.3f}")
            cefitresp.text(-8, 0.5*maxval,  f"scale = {respfitpars[4]:.3f} $\\pm$ {respperr[4]:.3f}")
            cefitresp.legend(loc="upper right")

            # Plot overlay with the reflection fit results. Adjust the amplitude
            fig, (cecompreso,cecompresp) = plt.subplots(1,2,layout='constrained', figsize=(15,5))

            self.HTrkResoMom.plotErrors(cecompreso)
            resocomppars = copy.deepcopy(refparnomat)
            resocomppars[0] = resofitpars[0] # steal normalization from fit
            maxval = np.amax(self.HTrkResoMom.data)
            cecompreso.plot(resobins, fxn_CrystalBall(resobins, *resocomppars), 'r-',label="Deconv. Ref. CB")
            cecompreso.text(-2, 0.8*maxval, f"$\\beta$ = {resocomppars[1]:.3f} $\\pm$ {refperrnomat[1]:.3f}")
            cecompreso.text(-2, 0.7*maxval, f"m = {resocomppars[2]:.3f} $\\pm$ {refperrnomat[2]:.3f}")
            cecompreso.text(-2, 0.6*maxval,  f"loc = {resocomppars[3]:.3f} $\\pm$ {refperrnomat[3]:.3f}")
            cecompreso.text(-2, 0.5*maxval,  f"scale = {resocomppars[4]:.3f} $\\pm$ {refperrnomat[4]:.3f}")
            cecompreso.legend(loc="upper right")

            self.HTrkRefRespMom.plotErrors(cecompresp)
            respcomppars = copy.deepcopy(refpartgt)
            respcomppars[0] = respfitpars[0] # steal normalization from fit
            cecompresp.plot(respbins, fxn_CrystalBall(respbins, *respcomppars), 'r-',label="Deconv. Ref. CB")
            maxval = np.amax(self.HTrkRefRespMom.data)
            cecompresp.text(-8, 0.8*maxval, f"$\\beta$ = {respcomppars[1]:.3f} $\\pm$ {refperrtgt[1]:.3f}")
            cecompresp.text(-8, 0.7*maxval, f"m = {respcomppars[2]:.3f} $\\pm$ {refperrtgt[2]:.3f}")
            cecompresp.text(-8, 0.6*maxval,  f"loc = {respcomppars[3]:.3f} $\\pm$ {refperrtgt[3]:.3f}")
            cecompresp.text(-8, 0.5*maxval,  f"scale = {respcomppars[4]:.3f} $\\pm$ {refperrtgt[4]:.3f}")
            cecompresp.legend(loc="upper right")

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
