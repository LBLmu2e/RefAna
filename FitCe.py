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
import h5py
from scipy.stats import crystalball


def fxn_CrystalBall(x, amp, beta, m, loc, scale):
    pars = np.array([beta, m, loc, scale])
    return amp*crystalball.pdf(x,*pars)

class FitCe(object):
    def __init__(self,savefile):
        self.HTgtRho = HistUtil.new_hist(name="HTgtRho",bins=100,range=rhorange,label="Fit",title="Target Rho",xlabel="Rho (mm)")
        self.HTgtRhoMC = HistUtil.new_hist(name="HTgtRho",bins=100,range=rhorange,label="MC",title="Target Rho",xlabel="Rho (mm)")
        self.HOriginRho = HistUtil.new_hist(name="HTgtRho",bins=100,range=rhorange,label="MC Origin",title="Target Rho",xlabel="Rho (mm)")
        foilrange = [-0.5,36.5]
        self.HTgtFoil = HistUtil.new_hist(name="HTgtFoil",bins=100,range=foilrange,label="Fit",title="Target Foil",xlabel="Foil (mm)")
        self.HTgtFoilMC = HistUtil.new_hist(name="HTgtFoil",bins=100,range=foilrange,label="MC",title="Target Foil",xlabel="Foil (mm)")
        self.HOriginFoil = HistUtil.new_hist(name="HTgtFoil",bins=100,range=foilrange,label="MC Origin",title="Target Foil",xlabel="Foil (mm)")
        costrange = [20,80]
        self.HTgtCost = HistUtil.new_hist(name="HTgtCost",bins=100,range=costrange,label="Fit",title="Target Momentum Cos($\\Theta$)",xlabel="Cos($\Theta$)")
        self.HTgtCostMC = HistUtil.new_hist(name="HTgtCost",bins=100,range=costrange,label="MC",title="Target Momentum Cos($\\Theta$)",xlabel="Cos($\\Theta$)")
        self.HOriginCost = HistUtil.new_hist(name="HTgtCost",bins=100,range=costrange,label="MC Origin",title="Target Momentum Cos($\\Theta$)",xlabel="Cos($\\Theta$)")


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

    def PlotTarget(self):
        fig, (tgtrho,tgtfoil,tgtcost) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        tgtfoil.hist(TgtFoil,label="Fit", bins=37, range=[-0.5,36.5], histtype='step')
        tgtfoil.hist(TgtFoilMC,label="MC", bins=37, range=[-0.5,36.5], histtype='step')
        tgtfoil.hist(OFoil,label="Origin", bins=37, range=[-0.5,36.5], histtype='step')
        tgtfoil.set_xlabel("Foil Number")
        tgtfoil.set_title("Target Foil")
        tgtcost.hist(TgtCost,label="Fit", bins=100, range=[-1,1], histtype='step')
        tgtcost.hist(TgtCostMC,label="MC", bins=100, range=[-1,1], histtype='step')
        tgtcost.hist(OCost,label="Origin", bins=100, range=[-1,1], histtype='step')
        tgtcost.set_xlabel("Cos($\\Theta$)")
        tgtcost.set_title("Cos($\\Theta$)")
        tgtcost.legend(loc="upper left")


    def FitCrystalBall(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

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
        popt, pcov = curve_fit(fxn_CrystalBall, dmommid, self.HDeltaNoMatMom.view(), p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        HistUtil.plot_errors(self.HDeltaNoMatMom,delmom)
        delmom.plot(dmommid, fxn_CrystalBall(dmommid, *p0), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\beta$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"m = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"loc = {popt[3]:.3f}")
        fig.text(0.1, 0.2,  f"scale = {popt[4]:.3f}")
        fig.text(0.1, 0.1,  f"amp = {popt[0]:.3f}")

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

        HistUtil.plot_errors(self.HDeltaTgtMom,delselmom)
        delselmom.plot(dmommid, fxn_CrystalBall(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\beta$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"m = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"loc = {popt[3]:.3f}")
        fig.text(0.6, 0.2,  f"scale = {popt[4]:.3f}")
        fig.text(0.6, 0.1,  f"amp = {popt[0]:.3f}")

