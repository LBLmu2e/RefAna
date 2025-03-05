# Compare upstream and downstream resolution at tracker mid for e- reflections
#
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np

#file = '/home/online1/ejc/public/brownd/dts.mu2e.CeEndpoint.MDC2020r.001210_00000000.art.digi.art.ntuple.root'
#file = '/data/HD5/users/brownd/ntp.brownd.Reflections.v4.root'
file = [ "/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple" ]
files = [
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000010.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000024.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000053.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000085.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00004963.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00005432.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00010057.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00010872.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00015026.root:TAReM/ntuple" ]

#with uproot.open(file) as f:

DeltaEntTime = []
DeltaEntTimeElMC =[]
DeltaEntTimeMuMC =[]
DeltaEntTimeDeMC =[]
UpMidMom = []
UpGoodMidMom = []
DeltaMidMom = []
DeltaMidMomMC = []
UpMomRes = []
DnMomRes = []
# setup general constants
minNHits = 20
minFitCon = 1.0e-5
maxDeltaT = 5.0 # nsec
ibatch=0
elPDG = 11
muPDG = 13
# momentum range around a conversion electron
cemom = 104
dmom = 20
minMom = cemom - dmom
maxMom = cemom + dmom
# Surface Ids
trkEntSID = 0
trkMidSID = 1
# counts for purity and efficiency
Ngood = 0
NgoodEl = 0
NEl = 0
for batch,rep in uproot.iterate(files,filter_name="/trk|trksegs|trkmcsim|gtrksegsmc/i",report=True):
    print("Processing batch ",ibatch)
    ibatch = ibatch+1
    segs = batch['trksegs'] # track fit samples
    nhits = batch['trk.nactive']  # track N hits
    fitcon = batch['trk.fitcon']  # track fit consistency
    trkMC = batch['trkmcsim']  # MC genealogy of particles
    segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
#    ak.type(segs).show()
#    print("segs axis 0: ",ak.num(segs,axis=0))
#    print("segs axis 1: ",ak.num(segs,axis=1))
#    print("segs axis 2: ",ak.num(segs,axis=2))
    upSegs = segs[:,0] # upstream track fits
    dnSegs = segs[:,1] # downstream track fits
    upSegsMC = segsMC[:,0] # upstream track MC truth
    dnSegsMC = segsMC[:,1] # downstream track MC truth
    upTrkMC = trkMC[:,0] # upstream fit associated true particles
    dnTrkMC = trkMC[:,1] # downstream fit associated true particles
    # basic consistency test
    assert((len(upSegs) == len(dnSegs)) & (len(upSegsMC) == len(dnSegsMC)) & (len(upSegs) == len(upSegsMC))& (len(upTrkMC) == len(dnTrkMC)) & (len(upSegs) == len(upTrkMC)))
#    print(upSegsMC)
#    print(len(fitcon), fitcon)
    upFitCon = fitcon[:,0]
    dnFitCon = fitcon[:,1]
    upNhits = nhits[:,0]
    dnNhits = nhits[:,1]
#    print(len(upFitCon),upFitCon)
#    print(len(dnFitCon),dnFitCon)
#    print(len(upNhits),upNhits)
#    print(len(dnNhits),dnNhits)

# select based on time difference at tracker entrance
    upEntTime = upSegs[(upSegs.sid==trkEntSID) & (upSegs.mom.z() > 0.0) ].time
    dnEntTime = dnSegs[(dnSegs.sid==trkEntSID) & (dnSegs.mom.z() > 0.0) ].time
    deltaEntTime = upEntTime-dnEntTime
    DeltaEntTime.extend(ak.flatten(deltaEntTime))
# select by MC truth
    upTrkMC = upTrkMC[upTrkMC.trkrel._rel == 0] # select the true particle most associated with the track
    dnTrkMC = dnTrkMC[dnTrkMC.trkrel._rel == 0]
    upTrkMC = ak.flatten(upTrkMC,axis=1) # project out the struct
    dnTrkMC = ak.flatten(dnTrkMC,axis=1)

#    print( len(upTrkMC))

    upElMC = upTrkMC.pdg == elPDG
    dnElMC = dnTrkMC.pdg == elPDG
    upMuMC = upTrkMC.pdg == muPDG
    dnMuMC = dnTrkMC.pdg == muPDG
    elMC = upElMC & dnElMC
    muMC = upMuMC & dnMuMC
    deMC = upMuMC & dnElMC
    deltaEntTimeElMC = deltaEntTime[elMC]
    deltaEntTimeMuMC = deltaEntTime[muMC]
    deltaEntTimeDeMC = deltaEntTime[deMC]
    DeltaEntTimeElMC.extend(ak.flatten(deltaEntTimeElMC))
    DeltaEntTimeMuMC.extend(ak.flatten(deltaEntTimeMuMC))
    DeltaEntTimeDeMC.extend(ak.flatten(deltaEntTimeDeMC))

#    print(deltaEntTimeElMC)
#    print(deltaEntTimeMuMC)

# select good electron fits based on time difference at tracker entrance

    goodDeltaT = abs(deltaEntTime) < maxDeltaT
    goodDeltaT = ak.flatten(goodDeltaT)


#    print(goodDeltaT,len(goodDeltaT))
# select based on fit quality
    upGoodFit = (upNhits >= minNHits) & (upFitCon > minFitCon)
    dnGoodFit = (dnNhits >= minNHits) & (dnFitCon > minFitCon)
    goodFit = upGoodFit & dnGoodFit
#    print(goodFit,len(goodFit))


    # sample the fits at middle of traacker
    upMidSegs = upSegs[upSegs.sid== trkMidSID]
    dnMidSegs = dnSegs[dnSegs.sid== trkMidSID]
    upMidSegsMC = upSegsMC[upSegsMC.sid== trkMidSID]
    dnMidSegsMC = dnSegsMC[dnSegsMC.sid== trkMidSID]
#    print("Mid seg counts ",len(upMidSegs),len(dnMidSegs),len(upMidSegsMC),len(dnMidSegsMC),len(elMC),len(goodFit),len(goodDeltaT))
#    print(upMidSegs[0:10])
#    print(upMidSegsMC[0:10])

    # total momentum at tracker mid
    upMidMom = upMidSegs.mom.magnitude()
    dnMidMom = dnMidSegs.mom.magnitude()
    upMidMomMC = upMidSegsMC.mom.magnitude()
    dnMidMomMC = dnMidSegsMC.mom.magnitude()
    # select correct direction
    upMidMomMC = upMidMomMC[upMidSegsMC.mom.z()<0]
    dnMidMomMC = dnMidMomMC[dnMidSegsMC.mom.z()>0]
#    print("midMomMC",len(upMidMomMC),len(dnMidMomMC))
#    print("before flatten",len(upMidMom),len(dnMidMom),len(upMidMomMC),len(dnMidMomMC))
#    print(upMidMomMC[0:10])
#    print(dnMidMomMC[0:10])
    # flatten
    upMidMom = ak.flatten(upMidMom,axis=1)
    dnMidMom = ak.flatten(dnMidMom,axis=1)
#    print("midmom ",len(upMidMom),len(dnMidMom))
    # select 'signal-like' electrons. For now, just the momentum, later maybe add
    # consistency with the target and pitch cuts
    signalLike = (dnMidMom > minMom) & (dnMidMom < maxMom)
#    print(len(signalLike))

    hasUpMidMomMC = ak.count_nonzero(upMidMomMC,axis=1,keepdims=True)==1
    hasDnMidMomMC = ak.count_nonzero(dnMidMomMC,axis=1,keepdims=True)==1
    hasUpMidMomMC = ak.flatten(hasUpMidMomMC)
    hasDnMidMomMC = ak.flatten(hasDnMidMomMC)
#    print("hasMC",len(hasUpMidMomMC),len(hasDnMidMomMC))
#    print(hasUpMidMomMC)
    goodReco = goodFit & goodDeltaT & signalLike
    goodMC = elMC & hasUpMidMomMC & hasDnMidMomMC
    goodRes = goodReco & goodMC  # resolution plot requires a good MC match
#    print("goodres",goodRes)
    upResMom = upMidMom[goodRes]
    dnResMom = dnMidMom[goodRes]
    upResMomMC = upMidMomMC[goodRes]
    dnResMomMC = dnMidMomMC[goodRes]
#    print("resmom before flatten",len(upResMom),len(dnResMom),len(upResMomMC),len(dnResMomMC))
    dnResMom = ak.ravel(dnResMom)
    upResMom = ak.ravel(upResMom)
    dnResMomMC = ak.ravel(dnResMomMC)
    upResMomMC = ak.ravel(upResMomMC)
#    print("resmom after flatten",len(upResMom),len(dnResMom),len(upResMomMC),len(dnResMomMC))
    upMomRes = upResMom-upResMomMC
    UpMomRes.extend(upMomRes)
    dnMomRes = dnResMom-dnResMomMC
    DnMomRes.extend(dnMomRes)

    upMidMomMC = ak.ravel(upMidMomMC)
    dnMidMomMC = ak.ravel(dnMidMomMC)
 #   print(upMidMom[0:10],upMidMomMC[0:10])

    # select based on reco
    upGoodMidMom = upMidMom[goodReco]
    dnGoodMidMom = dnMidMom[goodReco]
    UpMidMom.extend(upMidMom)
    UpGoodMidMom.extend(upGoodMidMom)
    # reflection momentum difference
    deltaMidMom = upGoodMidMom - dnGoodMidMom
    DeltaMidMom.extend(deltaMidMom)
    deltaMidMomMC = upResMomMC-dnResMomMC
    DeltaMidMomMC.extend(deltaMidMomMC)

# compute purity sums
    Ngood = Ngood + len(deltaEntTime[goodDeltaT])
    NEl = NEl + len(deltaEntTime[elMC])
    NgoodEl = NgoodEl + len(deltaEntTime[elMC & goodDeltaT])

print("From ", len(UpMidMom)," total, selected ",len(DeltaMidMom)," good quality signal-like reflections and ",len(UpMomRes)," reflections for resolution")

# compute Delta-T PID performance metrics
eff = NgoodEl/NEl
pur = NgoodEl/Ngood
print("For |Delta T| < ", maxDeltaT , " efficiency = ",eff," purity = ",pur)
# plot DeltaT
fig, deltat = plt.subplots(1,1,layout='constrained', figsize=(5,5))
nbins = 100
trange=(-20,20)
dt =     deltat.hist(DeltaEntTime,label="All", bins=nbins, range=trange, histtype='bar', stacked=True)
dtElMC = deltat.hist(DeltaEntTimeElMC,label="True Electron", bins=nbins, range=trange, histtype='bar', stacked=True)
dtMuMC = deltat.hist(DeltaEntTimeMuMC,label="True Muon", bins=nbins, range=trange, histtype='bar', stacked=True)
dtDeMC = deltat.hist(DeltaEntTimeDeMC,label="Muon Decays", bins=nbins, range=trange, histtype='bar', stacked=True)
deltat.set_title("$\\Delta$ Fit Time at Tracker Entrance")
deltat.set_xlabel("Upstream time - Downstreamtime (nSec)")
deltat.legend()
plt.show()
# plot Momentum
fig, (upMom, deltaMom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
upMom.hist(UpMidMom,label="All Upstream", bins=100, range=(70.0,150.0), histtype='step')
upMom.hist(UpGoodMidMom,label="Selected Upstream", bins=100, range=(70.0,150.0), histtype='step')
upMom.set_title("Upstream Momentum at Tracker Mid")
upMom.set_xlabel("Fit Momentum (MeV)")
upMom.legend()
deltaMom.hist(DeltaMidMom,label="Fit $\\Delta$ P", bins=100, range=(-5,15), histtype='step')
deltaMom.hist(DeltaMidMomMC,label="MC $\\Delta$ P", bins=100, range=(-5,15), histtype='step')
deltaMom.set_xlabel("$\\Delta$ Fit Momentum (MeV)")
deltaMom.set_title("Upstream -Downstream Tracker Mid Momentum")
deltaMom.legend()
plt.show()
# plot momentum resolution
fig, (upMomRes, dnMomRes)= plt.subplots(1,2,layout='constrained', figsize=(10,5))
upMomRes.hist(UpMomRes,label="Upstream",bins=100, range=(-3.0,3.0), histtype='bar')
upMomRes.set_title("Upstream Momentum  Resolution at Tracker Mid")
upMomRes.set_xlabel("Reco - True Momentum (MeV)")
dnMomRes.hist(DnMomRes,label="Downstream",bins=100, range=(-3.0,3.0), histtype='bar')
dnMomRes.set_title("Downstream Momentum  Resolution at Tracker Mid")
dnMomRes.set_xlabel("Reco - True Momentum (MeV)")
plt.show()

# plot dnstream and downstream momentum  Resolution
