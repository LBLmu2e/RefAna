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
import HistUtil
import h5py
from scipy.stats import crystalball

def fxn_CrystalBall(x, amp, beta, m, loc, scale):
    pars = np.array([beta, m, loc, scale])
    return amp*crystalball.pdf(x,*pars)

def TargetFoil(tgtz):
    tgtz0 = -4300. # target center in detector coordinates
    tgtdz = 22.222222 # target spacing
    ntgt = 37 # number of target foils
    tgt0z = tgtz0 - 0.5*(ntgt-1)*tgtdz
    tgtnum = (tgtz-tgt0z)/tgtdz
    itgt = int(round(tgtnum))
    return itgt


class HistCe(object):
    def __init__(self,momrange,minNHits,minFitCon,minTrkQual):
        self.MomRange = momrange
        self.minNHits = minNHits
        self.minFitCon = minFitCon
        self.minTrkQual = minTrkQual


        nDeltaMomBins = 200
        nMomBins = 200
        momrange=(self.MomRange[0],107)
        momresorange=(-2.5,2.5)
        momresprange=(-10,5)
        #print(len(Mom[0]),Mom[0:10][0])
        #print(len(MomReso[0]),MomReso[0:10][0])

        self.TrkLoc = [None]*3
        self.HOriginMom = HistUtil.new_hist(name="OriginMom",label="MC Origin",bins=nMomBins, range=momrange,title="Momentum at Origin",xlabel="Momentum (MeV)")

        self.HTrkFitMom = [None]*3
        self.HTrkMCMom = [None]*3
        self.HTrkRespMom = [None]*3
        self.HTrkResoMom = [None]*3
        self.HTrkRefRespMom = [None]*3
        self.TrackerSIDs = [SID.TT_Front(), SID.TT_Mid(), SID.TT_Back()]
        for isid in range(len(self.TrackerSIDs)):
            loc = SID.SurfaceName(self.TrackerSIDs[isid])
        # momentum in tracker
            self.HTrkFitMom[isid] = HistUtil.new_hist(name=loc+"Mom",label="Fit",bins=nMomBins, range=momrange,title="Momentum at "+loc,xlabel="Momentum (MeV)")
            self.HTrkMCMom[isid] = HistUtil.new_hist(name=loc+"Mom",label="MC",bins=nMomBins, range=momrange,title="Momentum at "+loc,xlabel="Momentum (MeV)")
            self.HTrkResoMom[isid] = HistUtil.new_hist(name=loc+"Resolution",label="",bins=nDeltaMomBins, range=momresorange,title="Momentum Resolution at "+loc,xlabel="$\\Delta$ Momentum (MeV)")
            self.HTrkRespMom[isid] = HistUtil.new_hist(name=loc+"Response",label="All",bins=nDeltaMomBins, range=momresprange,title="Momentum Response at "+loc,xlabel="$\\Delta$ Momentum (MeV)")
            self.HTrkRefRespMom[isid] = HistUtil.new_hist(name=loc+"Response",label="Reflectable",bins=nDeltaMomBins, range=momresprange,title="Momentum Response at "+loc,xlabel="$\\Delta$ Momentum (MeV)")

        # target intersections
        rhorange = [20,80]
        self.HTgtRho = HistUtil.new_hist(name="HTgtRho",bins=100,range=rhorange,label="Fit",title="Target Rho",xlabel="Rho (mm)")
        self.HTgtRhoMC = HistUtil.new_hist(name="HTgtRho",bins=100,range=rhorange,label="MC",title="Target Rho",xlabel="Rho (mm)")
        self.HOriginRho = HistUtil.new_hist(name="HTgtRho",bins=100,range=rhorange,label="MC Origin",title="Target Rho",xlabel="Rho (mm)")
        foilrange = [-0.5,36.5]
        self.HTgtFoil = HistUtil.new_hist(name="HTgtFoil",bins=37,range=foilrange,label="Fit",title="Target Foil",xlabel="Foil (mm)")
        self.HTgtFoilMC = HistUtil.new_hist(name="HTgtFoil",bins=37,range=foilrange,label="MC",title="Target Foil",xlabel="Foil (mm)")
        self.HOriginFoil = HistUtil.new_hist(name="HTgtFoil",bins=37,range=foilrange,label="MC Origin",title="Target Foil",xlabel="Foil (mm)")
        costrange = [-0.8,0.8]
        self.HTgtCosT = HistUtil.new_hist(name="HTgtCosT",bins=100,range=costrange,label="Fit",title="Target Momentum Cos($\\Theta$)",xlabel="Cos($\\Theta$)")
        self.HTgtCosTMC = HistUtil.new_hist(name="HTgtCosT",bins=100,range=costrange,label="MC",title="Target Momentum Cos($\\Theta$)",xlabel="Cos($\\Theta$)")
        self.HOriginCosT = HistUtil.new_hist(name="HTgtCosT",bins=100,range=costrange,label="MC Origin",title="Target Momentum Cos($\\Theta$)",xlabel="Cos($\\Theta$)")

        ### BINS ###


        self.HDTgtMomB12 = HistUtil.new_hist(name="DMom",label="B12", bins=nMomBins, range=momresprange, xlabel="Mom - True Mom (MeV)", title="Ce at TT_Front B12")
        self.HDTgtMomB34 = HistUtil.new_hist(name="DMom",label="B34", bins=nMomBins, range=momresprange, xlabel="Mom - True Mom (MeV)", title="Ce at TT_Front B34")
        self.HDTgtMomB56 = HistUtil.new_hist(name="DMom",label="B56", bins=nMomBins, range=momresprange, xlabel="Mom - True Mom (MeV)", title="Ce at TT_Front B56")
        self.HDTgtMomB78 = HistUtil.new_hist(name="DMom",label="B78", bins=nMomBins, range=momresprange, xlabel="Mom - True Mom (MeV)", title="Ce at TT_Front B78")
        self.HDTgtMomB9p = HistUtil.new_hist(name="DMom",label="B9p", bins=nMomBins, range=momresprange, xlabel="Mom - True Mom (MeV)", title="Ce at TT_Front B9p")

    def Loop(self,files):
        elPDG = 11

        rfile = uproot.open(files[0]+":EventNtuple")
        print(rfile.keys())
        ibatch = 0
        np.set_printoptions(precision=5,floatmode='fixed')
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(files,filter_name="/evtinfo|trk|trksegs|trkmcsim|trksegsmc/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            runnum = batch['run']
            subrun = batch['subrun']
            event = batch['event']
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            trkQual = batch['trkqual.result']  # track fit quality
            trkMC = batch['trkmcsim']  # MC genealogy of particles
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            # should be 1 track/event
            assert(ak.sum(ak.count_nonzero(nhits,axis=1)!=1) == 0)
            Segs = segs[:,0]
            FitCon = fitcon[:,0]
            NHits = nhits[:,0]
            TrkQual = trkQual[:,0]
                        # now MC
            SegsMC = segsMC[:,0] # segments (of 1st MC match) of 1st track
            TrkMC = trkMC[:,0,0] # primary MC match of 1st track
            # basic consistency test
            assert((len(runnum) == len( Segs)) & (len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(NHits) == len(Segs)))
            goodMC = (TrkMC.pdg == elPDG) & (TrkMC.trkrel._rel == 0)
            OMom = TrkMC[goodMC].mom.magnitude()
            #goodMC = goodMC & (OMom>self.MomRange[0]) & (OMom < self.MomRange[1])
            #OMom = OMom[goodMC]

            SegsMC = SegsMC[goodMC]
            Segs = Segs[goodMC]
            NHits = NHits[goodMC]
            FitCon = FitCon[goodMC]
            TrkQual = TrkQual[goodMC]
            goodFit = (NHits >= self.minNHits) & (FitCon > self.minFitCon) & (TrkQual > self.minTrkQual)
            TSDASeg = Segs[Segs.sid == SID.TSDA() ]
            noTSDA = ak.num(TSDASeg)==0

            self.HOriginMom.fill(np.array(OMom))
            self.HOriginRho.fill(np.array(TrkMC[goodMC].pos.rho()))
            self.HOriginCosT.fill(np.array(TrkMC[goodMC].mom.cosTheta()))
            self.HOriginFoil.fill(np.array(list(map(TargetFoil,TrkMC[goodMC].pos.z()))))
            # sample the fits at the specified
            for isid in range(len(self.TrackerSIDs)) :
                sid = self.TrackerSIDs[isid]
                segs = Segs[(Segs.sid == sid) & (Segs.mom.z() > 0.0) ]
                mom = segs.mom.magnitude()
                mom = mom[(mom > self.MomRange[0]) & (mom < self.MomRange[1])]
                hasmom= ak.count_nonzero(mom,axis=1)==1
                segsMC = SegsMC[(SegsMC.sid == sid) & (SegsMC.mom.z() > 0.0) ]
                momMC = segsMC.mom.magnitude()
                hasMC = ak.count_nonzero(momMC,axis=1)==1
                good = hasMC & goodFit & hasmom
                reflectable = good & noTSDA
                goodmom = mom[good]
                goodmom = ak.flatten(goodmom,axis=1)
                refmom = mom[reflectable]
                refmom = ak.flatten(refmom,axis=1)
                goodmomMC = momMC[good]
                goodmomMC = ak.flatten(goodmomMC,axis=1)
                assert(len(goodmom) == len(goodmomMC) )
                self.HTrkFitMom[isid].fill(np.array(goodmom))
                self.HTrkMCMom[isid].fill(np.array(goodmomMC))
                momreso = goodmom - goodmomMC
                self.HTrkResoMom[isid].fill(np.array(momreso))
                momresp = goodmom - OMom[good]
                self.HTrkRespMom[isid].fill(np.array(momresp))
                momrefresp = refmom - OMom[reflectable]
                self.HTrkRefRespMom[isid].fill(np.array(momrefresp))

            # tracker front, binned response

            segs = Segs[(Segs.sid == SID.TT_Front()) & (Segs.mom.z() > 0.0)]
            mom = segs.mom.magnitude()
            mom = mom[(mom > self.MomRange[0]) & (mom < self.MomRange[1])]
            hasmom = ak.count_nonzero(mom,axis=1)==1

            #foil response
            tgtsegs = Segs[(Segs.sid == SID.ST_Foils())]
            tgtmom = tgtsegs.mom.magnitude()
            ntgts = ak.count(tgtmom,axis=1)
            goodtgt = (ntgts > 0)
            print(len(hasmom),len(goodFit),len(goodMC))
            ntgts = ntgts[hasmom & goodFit & goodMC]

            DMom = ak.flatten(mom[hasmom & goodFit & goodMC]) - OMom[hasmom & goodFit]

            # bin DMom based on number of foil hits ntgts

            ntgts_np = np.array(ntgts)
            #print(ntgts_np, len(ntgts_np))

            test_12 = np.logical_or(np.equal(ntgts_np, 1), np.equal(ntgts_np, 2))
            test_34 = np.logical_or(np.equal(ntgts_np, 3), np.equal(ntgts_np, 4))
            test_56 = np.logical_or(np.equal(ntgts_np, 5), np.equal(ntgts_np, 6))
            test_78 = np.logical_or(np.equal(ntgts_np, 7), np.equal(ntgts_np, 8))
            test_9p = np.greater_equal(ntgts_np, 9)

            #print(len(test_12))
            #print(len(DMom))

            B12 = DMom[test_12]
            B34 = DMom[test_34]
            B56 = DMom[test_56]
            B78 = DMom[test_78]
            B9p = DMom[test_9p]

            print(B12)

            self.HDTgtMomB12.fill(np.array(B12))
            self.HDTgtMomB34.fill(np.array(B34))
            self.HDTgtMomB56.fill(np.array(B56))
            self.HDTgtMomB78.fill(np.array(B78))
            self.HDTgtMomB9p.fill(np.array(B9p))

#            avgmom = ak.sum(tgtmom,axis=1)
#            avgmom = avgmom[goodtgt]
#            avgmom = avgmom/ntgts
#            Mom[itarget].extend(avgmom)
#            omomtgt = OMom[goodtgt]
#            MomResp[itarget].extend((avgmom-omomtgt))
            self.HTgtRho.fill(np.array(ak.flatten(tgtsegs.pos.rho())))
            self.HTgtCosT.fill(np.array(ak.flatten(tgtsegs.mom.cosTheta())))
            self.HTgtFoil.fill(np.array(list(map(TargetFoil,ak.flatten(tgtsegs.pos.z())))))

            tgtsegsmc = SegsMC[(SegsMC.sid == SID.ST_Foils())]
            self.HTgtRhoMC.fill(np.array(ak.flatten(tgtsegsmc.pos.rho())))
            self.HTgtCosTMC.fill(np.array(ak.flatten(tgtsegsmc.mom.cosTheta())))
            self.HTgtFoilMC.fill(np.array(list(map(TargetFoil,ak.flatten(tgtsegsmc.pos.z())))))

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
        print()


    def PlotTrackerMomentum(self):
        fig, (amom,areso,aresp) = plt.subplots(3,3,layout='constrained', figsize=(15,20))
        for isid in range(len(self.TrackerSIDs)) :
            HistUtil.plot(self.HTrkFitMom[isid],amom[isid])
            HistUtil.plot(self.HTrkMCMom[isid],amom[isid])
            HistUtil.plot(self.HTrkResoMom[isid],areso[isid])
            HistUtil.plot(self.HTrkRespMom[isid],aresp[isid])
            HistUtil.plot(self.HTrkRefRespMom[isid],aresp[isid])
        amom[0].legend(loc="upper left")
        aresp[0].legend(loc="upper left")

    def PlotTarget(self):
        fig, (arho,afoil,acost) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        # Rho
        HistUtil.plot(self.HTgtRho,arho)
        HistUtil.plot(self.HTgtRhoMC,arho)
        HistUtil.plot(self.HOriginRho,arho)
        arho.legend(loc="upper left")
        # Foil
        HistUtil.plot(self.HTgtFoil,afoil)
        HistUtil.plot(self.HTgtFoilMC,afoil)
        HistUtil.plot(self.HOriginFoil,afoil)
        # Cos(theta)
        HistUtil.plot(self.HTgtCosT,acost)
        HistUtil.plot(self.HTgtCosTMC,acost)
        HistUtil.plot(self.HOriginCosT,acost)

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
            HistUtil.save_hist(self.HOriginMom,hdf5file)
            for isid in range(len(self.TrackerSIDs)) :
                HistUtil.save_hist(self.HTrkFitMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkMCMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkResoMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkRespMom[isid],hdf5file)
                HistUtil.save_hist(self.HTrkRefRespMom[isid],hdf5file)
            HistUtil.save_hist(self.HTgtRho,hdf5file)
            HistUtil.save_hist(self.HTgtRhoMC,hdf5file)
            HistUtil.save_hist(self.HOriginRho,hdf5file)
            HistUtil.save_hist(self.HTgtFoil,hdf5file)
            HistUtil.save_hist(self.HTgtFoilMC,hdf5file)
            HistUtil.save_hist(self.HOriginFoil,hdf5file)
            HistUtil.save_hist(self.HTgtCosT,hdf5file)
            HistUtil.save_hist(self.HTgtCosTMC,hdf5file)
            HistUtil.save_hist(self.HOriginCosT,hdf5file)

            # save my new histograms

            HistUtil.save_hist(self.HDTgtMomB12,hdf5file)
            HistUtil.save_hist(self.HDTgtMomB34,hdf5file)
            HistUtil.save_hist(self.HDTgtMomB56,hdf5file)
            HistUtil.save_hist(self.HDTgtMomB78,hdf5file)
            HistUtil.save_hist(self.HDTgtMomB9p,hdf5file)


#            fig, (tgtMomResp,momResp[0], momResp[1], momResp[2]) = plt.subplots(1,4,layout='constrained', figsize=(15,5))
#            tgtMomResp.hist(MomResp[itarget],label="ST", bins=nMomBins, range=momresprange, histtype='step')
#            tgtMomResp.set_xlabel("Extrapolated - MC Origin Momentum (MeV)")
#            tgtMomResp.set_title("Upstream ST Foil")
#            for isid in range(len(self.TrackerSIDs)) :
#                MomRespHist[isid] =  momResp[isid].hist(MomResp[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momresprange, histtype='step')
#                momResp[isid].set_xlabel("Fit - MC Origin Momentum (MeV)")
#                momResp[isid].set_title(self.TrkLoc[isid])
#
#            fig, (momReso[0], momReso[1], momReso[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
#            for isid in range(len(self.TrackerSIDs)) :
#                momReso[isid].hist(MomReso[isid],label=self.TrkLoc[isid], bins=nMomBins, range=momresorange, histtype='step')
#                momReso[isid].set_xlabel("Fit - MC Momentum (MeV)")
#                momReso[isid].set_title(self.TrkLoc[isid])
#        #
#        # response function fit
#        #
#        fig, (momRespFit[0],momRespFit[1],momRespFit[2]) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
#        for isid in range(len(self.TrackerSIDs)) :
#            momRespHist = MomRespHist[isid]
#            momRespErrors = np.zeros(len(momRespHist[1])-1)
#            momRespBinMid = np.zeros(len(momRespHist[1])-1)
#            for ibin in range(len(momRespErrors)):
#                momRespBinMid[ibin] = 0.5*(momRespHist[1][ibin] + momRespHist[1][ibin+1])
#                momRespErrors[ibin] = max(1.0,math.sqrt(momRespHist[0][ibin]))
#            momRespIntegral = np.sum(momRespHist[0])
#        # initialize the fit parameters
#            mu_0 = np.mean(momRespBinMid*momRespHist[0]/momRespIntegral) # initial mean
#            var = np.sum(((momRespBinMid**2)*momRespHist[0])/momRespIntegral) - mu_0**2
#            sigma_0 = np.sqrt(var) # initial sigma
#            lamb_0 = sigma_0 # initial exponential (guess)
#            binsize = momRespHist[1][1]-momRespHist[1][0]
#            amp_0 = momRespIntegral*binsize # initial amplitude
#            p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
#        # fit, returing optimum parameters and covariance
#            popt, pcov = curve_fit(fxn_expGauss, momRespBinMid, momRespHist[0], p0, sigma=momRespErrors)
#            print("For SID=",self.TrackerSIDs[isid],"Trk fit parameters",popt)
#            print("Trk fit covariance",pcov)
#            momRespFit[isid].stairs(edges=momRespHist[1],values=momRespHist[0],label="$\\Delta$ P")
#            momRespFit[isid].plot(momRespBinMid, fxn_expGauss(momRespBinMid, *popt), 'r-',label="EMG Fit")
#            momRespFit[isid].legend()
#            momRespFit[isid].set_title("EMG fit at "+self.TrkLoc[isid])
#            momRespFit[isid].set_xlabel("Fit-MC Origin Momentum (MeV)")
#            fig.text(self.FigX[isid], 0.5, f"$\\mu$ = {popt[1]:.3f}")
#            fig.text(self.FigX[isid], 0.4, f"$\\sigma$ = {popt[2]:.3f}")
#            fig.text(self.FigX[isid], 0.3, f"$\\lambda$ = {popt[3]:.3f}")
#        plt.show()
