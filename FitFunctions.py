import numpy as np
from scipy.special import erf
from scipy import special
from scipy.optimize import curve_fit
from scipy.stats import crystalball
import math
import HistUtil

def fxn_DSCB(x, mu, sigma, alphal, nl, alphar, nr):
    """ Evaluates the Double-Sided Crystal Ball Probability Density Function.  """
    # Standardize the variable
    z = (x - mu) / sigma
    # Absolute values for thresholds
    abs_alphal = np.abs(alphal)
    abs_alphar = np.abs(alphar)
    # Core mathematical helper constants
    Al = (nl / abs_alphal)**nl * np.exp(-abs_alphal**2 / 2)
    Bl = nl / abs_alphal - abs_alphal
    Ar = (nr / abs_alphar)**nr * np.exp(-abs_alphar**2 / 2)
    Br = nr / abs_alphar - abs_alphar
    # Define the piecewise conditions
    cond_left = z < -abs_alphal
    cond_right = z > abs_alphar
    cond_core = (~cond_left) & (~cond_right)
    # Calculate the unnormalized PDF values
    pdf = np.zeros_like(z)
    pdf[cond_left] = Al * (Bl - z[cond_left])**(-nl)
    pdf[cond_right] = Ar * (Br + z[cond_right])**(-nr)
    pdf[cond_core] = np.exp(-z[cond_core]**2 / 2)
    # Analytical normalization factor (Integral = 1)
    # 1. Core Gaussian integral
    int_core = sigma * np.sqrt(np.pi / 2) * (erf(abs_alphar / np.sqrt(2)) - erf(-abs_alphal / np.sqrt(2)))
    # 2. Left power-law tail integral
    int_left = sigma * (nl / abs_alphal) * np.exp(-abs_alphal**2 / 2) / (nl - 1)
    # 3. Right power-law tail integral
    int_right = sigma * (nr / abs_alphar) * np.exp(-abs_alphar**2 / 2) / (nr - 1)
    total_norm = int_core + int_left + int_right
    return pdf / total_norm

def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

class CrystalBall:
    def __init__(self,reverse=False):
        self.pname = ["Amplitude","Beta","Power","Mean","Sigma"]
        self.rev=reverse
    def name(self):
        if self.rev:
            return "ReverseCrystalBall"
        return "CrystalBall"
    def fxn(self,x, amp, beta, m, loc, scale):
        pars = np.array([beta, m, loc, scale])
        if self.rev:
            x = -x
        return amp*crystalball.pdf(x,*pars)
    def init(self,hist):
        loc_0 = HistUtil.mean(hist)
        if self.rev:
            loc_0 = -1*loc_0
        beta_0 = 3*math.sqrt(HistUtil.variance(hist))
        m_0 = 3.0
        scale_0 = HistUtil.FWHM(hist)/2.355
        amp_0 = HistUtil.integral(hist)*HistUtil.bin_width(hist)
        params = np.array([amp_0, beta_0, m_0, loc_0, scale_0]) # initial parameters
        return params
    def fit(self,hist,xrange=[0,-1],verbose=False) :
        iparams = self.init(hist)
        if verbose:
            print("Initial parameters",iparams)
        binmid,binval,binerr = HistUtil.fit_arrays(hist,xrange)
        oparams, cov = curve_fit(f=self.fxn,xdata=binmid,ydata=binval,sigma=binerr,absolute_sigma=True,p0=iparams)
        if verbose:
            print("Optimized parameters",oparams)
        return oparams,cov,binmid

def fxn_NegCrystalBall(x, amp, beta, m, loc, scale):
    pars = np.array([beta, m, loc, scale])
    return amp*crystalball.pdf(-x,*pars)

