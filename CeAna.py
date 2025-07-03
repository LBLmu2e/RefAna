#
# class to analyze high-energy electrons
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

def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func


def TargetFoil(tgtz):
    tgtz0 = -4300. # target center in detector coordinates
    tgtdz = 22.222222 # target spacing
    ntgt = 37 # number of target foils
    tgt0z = tgtz0 - 0.5*(ntgt-1)*tgtdz
    tgtnum = (tgtz-tgt0z)/tgtdz
    itgt = int(round(tgtnum))
    return itgt


class CeAna(object):
    def __init__(self,momrange,minNHits,minFitCon,sids):
        self.MomRange = momrange
        self.minNHits = minNHits
        self.minFitCon = minFitCon
        self.SIDs = sids
        self.TrkLoc = [None]*3
        self.TrkLoc[0] = "Tracker Entrance"
        self.TrkLoc[1] = "Tracker Middle"
        self.TrkLoc[2] = "Tracker Exit"
        self.FigX =[None]*3
        self.FigX[0] = 0.05
        self.FigX[1] = self.FigX[0] + 1.0/3.0
        self.FigX[2] = self.FigX[1] + 1.0/3.0
    def Loop(self,files):
       # Momentum and response
        Mom = []
        MCMom = []
        MomReso = []
        MomResp = []
        # MC origin information
#        ORho = []*2
        ORho = []
#        ORho[0] = []
#        ORho[1] = []
        OCost = []
        OFoil = []
        OPhi = []
        # MC true foil intersection
        TgtRhoMC = []
        TgtCostMC = []
        TgtPhiMC = []
        TgtFoilMC = []
        # Reco foil intersection
        TgtRho = []
        TgtCost = []
        TgtPhi = []
        TgtFoil = []
        elPDG = 11
        for isid in range(len(self.SIDs)+1) : # add 1 for target
            Mom.append([])
            MomReso.append([])
            MomResp.append([])
        ibatch=0
        itarget = len(Mom)-1;
#        Mom[itarget] = np.array()
        Mom[itarget] = []
        MomResp[itarget] = []

        rfile = uproot.open(files[0]+":EventNtuple")
        print(rfile.keys())

        np.set_printoptions(precision=5,floatmode='fixed')
        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk|trksegs|trkmcsim|trksegsmc/i",report=True):
            print(batch,type(batch))
            print("Processing batch ",ibatch)
            ibatch = ibatch+1
            runnum = batch['run']
            subrun = batch['subrun']
            event = batch['event']
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            trkMC = batch['trkmcsim']  # MC genealogy of particles
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            # should be 1 track/event
            assert(ak.sum(ak.count_nonzero(nhits,axis=1)!=1) == 0)
            Segs = segs[:,0]
            FitCon = fitcon[:,0]
            Nhits = nhits[:,0]
            goodFit = (Nhits >= self.minNHits) & (FitCon > self.minFitCon)
            TSDASeg = Segs[Segs.sid == SID.TSDA() ]
            noTSDA = ak.num(TSDASeg)==0
            # now MC
            SegsMC = segsMC[:,0] # segments (of 1st MC match) of 1st track
            TrkMC = trkMC[:,0,0] # primary MC match of 1st track
            # basic consistency test
            assert((len(runnum) == len( Segs)) & (len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(Nhits) == len(Segs)))
            goodMC = (TrkMC.pdg == elPDG) & (TrkMC.trkrel._rel == 0)
            OMom = TrkMC[goodMC].mom.magnitude()
            goodMC = goodMC & (OMom>self.MomRange[0]) & (OMom < self.MomRange[1])
            OMom = OMom[goodMC]
            ORho.extend(TrkMC[goodMC].pos.rho())
            OPhi.extend(TrkMC[goodMC].pos.phi())
            OCost.extend(TrkMC[goodMC].mom.cosTheta())
#            print(np.array(TrkMC.pos.z()))
            OFoil.extend(list(map(TargetFoil,TrkMC[goodMC].pos.z())))
            SegsMC = SegsMC[goodMC]
            Segs = Segs[goodMC]
            MCMom.extend(OMom)
            # sample the fits at the specified
            for isid in range(len(self.SIDs)) :
                sid = self.SIDs[isid]
                segs = Segs[(Segs.sid == sid) & (Segs.mom.z() > 0.0) ]
                mom = segs.mom.magnitude()
                mom = mom[(mom > self.MomRange[0]) & (mom < self.MomRange[1])]
                hasmom= ak.count_nonzero(mom,axis=1)==1
                segsMC = SegsMC[(SegsMC.sid == sid) & (SegsMC.mom.z() > 0.0) ]
                momMC = segsMC.mom.magnitude()
                hasMC = ak.count_nonzero(momMC,axis=1)==1
                select = hasMC & goodFit & hasmom & noTSDA
                mom = mom[select]
                momMC = momMC[select]
                mom = ak.flatten(mom,axis=1)
                momMC = ak.flatten(momMC,axis=1)
                assert(len(mom) == len(momMC) )
                Mom[isid].extend(mom)
                momreso = mom - momMC
                MomReso[isid].extend(momreso)
                selMomMC = OMom[select]
                momresp = mom - selMomMC
                MomResp[isid].extend(momresp)
            #foil response
            tgtsegs = Segs[(Segs.sid == SID.ST_Foils())]
            tgtmom = tgtsegs.mom.magnitude()
            ntgts = ak.count(tgtmom,axis=1)
            goodtgt = (ntgts > 0)
            ntgts = ntgts[goodtgt]
            avgmom = ak.sum(tgtmom,axis=1)
            avgmom = avgmom[goodtgt]
            avgmom = avgmom/ntgts
            Mom[itarget].extend(avgmom)
            omomtgt = OMom[goodtgt]
            MomResp[itarget].extend((avgmom-omomtgt))
            TgtRho.extend(ak.flatten(tgtsegs.pos.rho()))
            TgtCost.extend(ak.flatten(tgtsegs.mom.cosTheta()))
            TgtPhi.extend(ak.flatten(tgtsegs.pos.phi()))
            TgtFoil.extend(map(TargetFoil,ak.flatten(tgtsegs.pos.z())))

            tgtsegsmc = SegsMC[(SegsMC.sid == SID.ST_Foils())]
            TgtRhoMC.extend(ak.flatten(tgtsegsmc.pos.rho()))
            TgtCostMC.extend(ak.flatten(tgtsegsmc.mom.cosTheta()))
            TgtPhiMC.extend(ak.flatten(tgtsegsmc.pos.phi()))
            TgtFoilMC.extend(map(TargetFoil,ak.flatten(tgtsegsmc.pos.z())))

            # test for missing intersections
            hasent = (Segs.sid == 0) & (Segs.mom.z() > 0.0)
            hasmid = (Segs.sid == 1) & (Segs.mom.z() > 0.0)
            hasxit = (Segs.sid == 2) & (Segs.mom.z() > 0.0)
            hasent = ak.any(hasent,axis=1)
            hasmid = ak.any(hasmid,axis=1)
            hasxit = ak.any(hasxit,axis=1)
            hasall = hasent & hasmid & hasxit
            missing = ak.count(hasall,0) - ak.count_nonzero(hasall)
            if(missing > 0):
                print("Found",missing,"Instances of missing intersections in",ak.count(hasall,0),"tracks")
                for itrk in range(len(hasall)):
                    if (not hasall[itrk]):
                        print("Missing intersection: ",hasent[itrk],hasmid[itrk],hasxit[itrk]," eid ",runnum[itrk],":",subrun[itrk],":",event[itrk],sep="")

        # plot positions
#        print(TgtRho)
#        tz = TgtPos.z()
        fig, (tgtrho,tgtfoil,tgtcost,tgtphi) = plt.subplots(1,4,layout='constrained', figsize=(15,5))
        tgtrho.hist(TgtRho,label="Fit", bins=100, range=[20,80], histtype='step')
        tgtrho.hist(TgtRhoMC,label="MC", bins=100, range=[20,80], histtype='step')
        tgtrho.hist(ORho,label="Origin", bins=100, range=[20,80], histtype='step')
        tgtrho.set_xlabel("Rho (mm)")
        tgtrho.set_title("Target Rho")
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
        tgtphi.hist(TgtPhi,label="Fit", bins=100, range=[-math.pi,math.pi], histtype='step')
        tgtphi.hist(TgtPhiMC,label="MC", bins=100, range=[-math.pi,math.pi], histtype='step')
        tgtphi.hist(OPhi,label="Origin", bins=100, range=[-math.pi,math.pi], histtype='step')
        tgtphi.set_xlabel("$\\Phi$")
        tgtphi.set_title("$\\Phi$")
        tgtphi.legend(loc="upper left")


        # plot Momentum
        nDeltaMomBins = 200
        nMomBins = 100
        momrange=(self.MomRange[0],107)
        momresorange=(-2.5,2.5)
        momresprange=(-10,5)
        #print(len(Mom[0]),Mom[0:10][0])
        #print(len(MomReso[0]),MomReso[0:10][0])
        momVal = [None]*3
        momReso = [None]*3
        momResp = [None]*3
        momRespFit = [None]*3
        MomRespHist = [None]*3
#        print("Mom[itarget]",Mom[itarget])

        fig, (tgtMom, momVal[0], momVal[1], momVal[2]) = plt.subplots(1,4,layout='constrained', figsize=(15,5))
        tgtMom.hist(Mom[itarget],label="ST", bins=nMomBins, range=momrange, histtype='step')
        tgtMom.set_xlabel("Extrapolated Momentum (MeV)")
        tgtMom.set_title("Upstream ST Foil")
        for isid in range(len(self.SIDs)) :
            momVal[isid].hist(Mom[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momrange, histtype='step')
            momVal[isid].set_xlabel("Fit Momentum (MeV)")
            momVal[isid].set_title(self.TrkLoc[isid])

        fig, (tgtMomResp,momResp[0], momResp[1], momResp[2]) = plt.subplots(1,4,layout='constrained', figsize=(15,5))
        tgtMomResp.hist(MomResp[itarget],label="ST", bins=nMomBins, range=momresprange, histtype='step')
        tgtMomResp.set_xlabel("Extrapolated - MC Origin Momentum (MeV)")
        tgtMomResp.set_title("Upstream ST Foil")
        for isid in range(len(self.SIDs)) :
            MomRespHist[isid] =  momResp[isid].hist(MomResp[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momresprange, histtype='step')
            momResp[isid].set_xlabel("Fit - MC Origin Momentum (MeV)")
            momResp[isid].set_title(self.TrkLoc[isid])

        fig, (momReso[0], momReso[1], momReso[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
        for isid in range(len(self.SIDs)) :
            momReso[isid].hist(MomReso[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momresorange, histtype='step')
            momReso[isid].set_xlabel("Fit - MC Momentum (MeV)")
            momReso[isid].set_title(self.TrkLoc[isid])
        #
        # response function fit
        #
        fig, (momRespFit[0],momRespFit[1],momRespFit[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
        for isid in range(len(self.SIDs)) :
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
            print("For SID=",self.SIDs[isid],"Trk fit parameters",popt)
            print("Trk fit covariance",pcov)
            momRespFit[isid].stairs(edges=momRespHist[1],values=momRespHist[0],label="$\\Delta$ P")
            momRespFit[isid].plot(momRespBinMid, fxn_expGauss(momRespBinMid, *popt), 'r-',label="EMG Fit")
            momRespFit[isid].legend()
            momRespFit[isid].set_title("EMG fit at "+self.TrkLoc[isid])
            momRespFit[isid].set_xlabel("Fit-MC Origin Momentum (MeV)")
            fig.text(self.FigX[isid], 0.5, f"$\\mu$ = {popt[1]:.3f}")
            fig.text(self.FigX[isid], 0.4, f"$\\sigma$ = {popt[2]:.3f}")
            fig.text(self.FigX[isid], 0.3, f"$\\lambda$ = {popt[3]:.3f}")
        plt.show()
