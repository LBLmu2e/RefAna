# Compare upstream and downstream resolution at tracker mid for e- reflections
#
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special

def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

# setup general constants
minNHits = 20
minFitCon = 1.0e-5
ibatch=0
elPDG = 11
# momentum range around a conversion electron
cemom = 104
dmom = 20
minMom = cemom - dmom
maxMom = cemom + dmom
# Surface Ids
TrkSID = [None]*3
TrkSID[0] = 0 # entrance
TrkSID[1] = 1 # mid
TrkSID[2] = 2 # exit
TrkLoc = [None]*3
TrkLoc[0] = "Tracker Entrance"
TrkLoc[1] = "Tracker Middle"
TrkLoc[2] = "Tracker Exit"
FigX =[None]*3
FigX[0] = 0.05
FigX[1] = FigX[0] + 1.0/3.0
FigX[2] = FigX[1] + 1.0/3.0
# Momentum and response
Mom = []
MCMom = []
MomReso = []
MomResp = []
originMomMC = []
elPDG = 11
TSDASID = 96 # Exit of DS

for isid in range(len(TrkSID)) :
    Mom.append([])
    MomReso.append([])
    MomResp.append([])
    originMomMC.append([])

for batch,rep in uproot.iterate(f03files,filter_name="/trk|trksegs|trkmcsim|trksegsmc/i",report=True):
    print("Processing batch ",ibatch)
    ibatch = ibatch+1
    segs = batch['trksegs'] # track fit samples
    nhits = batch['trk.nactive']  # track N hits
    fitcon = batch['trk.fitcon']  # track fit consistency
    trkMC = batch['trkmcsim']  # MC genealogy of particles
    segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
    Segs = segs[:,0] # assume the 1st track is the Ce
    SegsMC = segsMC[:,0] # upstream track MC truth
    TrkMC = trkMC[:,0] # upstream fit associated true particles
    FitCon = fitcon[:,0]
    Nhits = nhits[:,0]
    goodFit = (Nhits >= minNHits) & (FitCon > minFitCon)
    goodTrkMC = (TrkMC.pdg == elPDG) & (TrkMC.trkrel._rel == 0)
    TSDASeg = Segs[Segs.sid == TSDASID]
    noTSDA = ak.num(TSDASeg)==0
#    print(noTSDA)
    originMomMC = TrkMC[goodTrkMC].mom.magnitude()
    # basic consistency test
    assert((len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(Nhits) == len(Segs)) & (len(originMomMC) == len(Segs)))
    originMomMC = originMomMC[(originMomMC>minMom) & (originMomMC < maxMom)]
    omomMC = ak.flatten(originMomMC,axis=1)
    MCMom.extend(omomMC)
    hasOriginMomMC = ak.count_nonzero(originMomMC,axis=1,keepdims=True)==1
    hasOriginMomMC = ak.flatten(hasOriginMomMC,axis=1)
#    print(hasOriginMomMC[0:10])
#    print(originMomMC[0:10])
    # sample the fits at 3 tracker locations
    for isid in range(len(TrkSID)) :
        sid = TrkSID[isid]
        segs = Segs[Segs.sid == sid]
        mom = segs.mom.magnitude()
        mom = mom[(mom > minMom) & (mom < maxMom)]
        hasmom= ak.count_nonzero(mom,axis=1,keepdims=True)==1
        hasmom =ak.flatten(hasmom,axis=1)
        segsMC = SegsMC[(SegsMC.sid == sid) & (SegsMC.mom.z() > 0.0) ]
        momMC = segsMC.mom.magnitude()
        hasMC = ak.count_nonzero(momMC,axis=1,keepdims=True)==1
        hasMC = ak.flatten(hasMC,axis=1)
        select = goodFit & hasmom & hasMC & hasOriginMomMC & noTSDA
        mom = mom[select]
        momMC = momMC[select]
        mom = ak.flatten(mom,axis=1)
        momMC = ak.flatten(momMC,axis=1)
        assert(len(mom) == len(momMC) )
        Mom[isid].extend(mom)
        momreso = mom - momMC
        MomReso[isid].extend(momreso)
        omomMC = originMomMC[select]
        omomMC = ak.flatten(omomMC,axis=1)
        momresp = mom - omomMC
        MomResp[isid].extend(momresp)
    # plot Momentum
nDeltaMomBins = 200
nMomBins = 100
momrange=(minMom,107)
momresorange=(-2.5,2.5)
momresprange=(-10,5)
#print(len(Mom[0]),Mom[0:10][0])
#print(len(MomReso[0]),MomReso[0:10][0])
momVal = [None]*3
momReso = [None]*3
momResp = [None]*3
momRespFit = [None]*3
MomRespHist = [None]*3

fig, (momVal[0], momVal[1], momVal[2], mcMom) = plt.subplots(1,4,layout='constrained', figsize=(10,5))
mcMom.hist(MCMom,label="MC Origin", bins=nMomBins, range=momrange, histtype='step')
mcMom.set_xlabel("MC Momentum (MeV)")
mcMom.set_title("MC Origin Momentum")
for isid in range(len(TrkSID)) :
    momVal[isid].hist(Mom[isid],label=TrkLoc[isid], bins=nMomBins, range=momrange, histtype='step')
    momVal[isid].set_xlabel("Fit Momentum (MeV)")
    momVal[isid].set_title(TrkLoc[isid])

fig, (momResp[0], momResp[1], momResp[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
for isid in range(len(TrkSID)) :
    MomRespHist[isid] =  momResp[isid].hist(MomResp[isid],label=TrkLoc[isid], bins=nMomBins, range=momresprange, histtype='step')
    momResp[isid].set_xlabel("Fit - MC Origin Momentum (MeV)")
    momResp[isid].set_title(TrkLoc[isid])

fig, (momReso[0], momReso[1], momReso[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
for isid in range(len(TrkSID)) :
    momReso[isid].hist(MomReso[isid],label=TrkLoc[isid], bins=nMomBins, range=momresorange, histtype='step')
    momReso[isid].set_xlabel("Fit - MC Momentum (MeV)")
    momReso[isid].set_title(TrkLoc[isid])
#
# response function fit
#
fig, (momRespFit[0],momRespFit[1],momRespFit[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
for isid in range(len(TrkSID)) :
    momRespHist = MomRespHist[isid]
    momRespErrors = np.zeros(len(momRespHist[1])-1)
    momRespBinMid = np.zeros(len(momRespHist[1])-1)
    for ibin in range(len(momRespErrors)):
        momRespBinMid[ibin] = 0.5*(momRespHist[1][ibin] + momRespHist[1][ibin+1])
        momRespErrors[ibin] = max(1.0,math.sqrt(momRespHist[0][ibin]))
    momRespIntegral = np.sum(momRespHist[0])
# initialize the fit parameters
    mu_0 = np.mean(momRespBinMid*momRespHist[0]/momRespIntegral) # initial mean
    var = np.sum(((momRespBinMid**2)*momRespHist[0])/momRespIntegral) - mu_0**2
    sigma_0 = np.sqrt(var) # initial sigma
    lamb_0 = sigma_0 # initial exponential (guess)
    binsize = momRespHist[1][1]-momRespHist[1][0]
    amp_0 = momRespIntegral*binsize # initial amplitude
    p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
# fit, returing optimum parameters and covariance
    popt, pcov = curve_fit(fxn_expGauss, momRespBinMid, momRespHist[0], p0, sigma=momRespErrors)
    print("For SID=",TrkSID[isid],"Trk fit parameters",popt)
    print("Trk fit covariance",pcov)
    momRespFit[isid].stairs(edges=momRespHist[1],values=momRespHist[0],label="$\\Delta$ P")
    momRespFit[isid].plot(momRespBinMid, fxn_expGauss(momRespBinMid, *popt), 'r-',label="EMG Fit")
    momRespFit[isid].legend()
    momRespFit[isid].set_title("EMG fit at "+TrkLoc[isid])
    momRespFit[isid].set_xlabel("Fit-MC Origin Momentum (MeV)")
    fig.text(FigX[isid], 0.5, f"$\\mu$ = {popt[1]:.3f}")
    fig.text(FigX[isid], 0.4, f"$\\sigma$ = {popt[2]:.3f}")
    fig.text(FigX[isid], 0.3, f"$\\lambda$ = {popt[3]:.3f}")
plt.show()
