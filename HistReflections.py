#
# make histograms of reflecting particles
#
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
import math
from scipy import special
import SurfaceIds as SID
import h5py
import HistUtil

class HistReflections(object):
    def __init__(self,momrange,pdg,sid):
        # PDG cods of signal and background particles
        self.PDG = pdg
        PDGNames = {-13:"$\\mu^+$",-11:"$e^+$",11:"$e^-$",13:"$\\mu^-$"}
        self.PDGName = PDGNames[self.PDG]
        # setup cuts; these should be overrideable FIXME
        self.MinNHits = 15
        self.MinFitCon = 1.0e-6
        self.MaxDeltaT = 5.0 # nsec
        self.MomRange = momrange
        self.MinTQ = 0.2 # ANN output
        # Surface Ids
        self.SID = sid
        self.CompName = SID.SurfaceName(sid)
        # fit quality histograms
        self.HUpTQ = HistUtil.new_hist(name="HUpTQ",bins=100,range=[0.0,1.0],label="Up TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HDnTQ = HistUtil.new_hist(name="HDnTQ",bins=100,range=[0.0,1.0],label="Down TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HUpFitCon = HistUtil.new_hist(name="HUpFitCon",bins=100,range=[0.0,1.0],label="Up FitCon",title="Fit Consistency",xlabel="")
        self.HDnFitCon = HistUtil.new_hist(name="HDnFitCon",bins=100,range=[0.0,1.0],label="Down FitCon",title="Fit Consistency",xlabel="")
        self.HUpNHits = HistUtil.new_hist(name="HUpNHits",bins=100,range=[0.5,100.5],label="Up NActive",title="Fit N Hits",xlabel="N Hits")
        self.HDnNHits = HistUtil.new_hist(name="HDnNHits",bins=100,range=[0.5,100.5],label="Down NActive",title="Fit N Hits",xlabel="N Hits")

        # intersection histograms
        nNMatBins = 31
        NMatRange = [-0.5,30.5]
        self.HNST = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="All ST",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNIPA = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="All IPA",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNSTTgt = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="Target ST",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNIPATgt = HistUtil.new_hist(bins=nNMatBins,range=NMatRange,name="NInter",label="Target IPA",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        # Momentum histograms
        nMomBins = 100
        momrange=(40.0,220.0)
        nDeltaMomBins = 150
        deltaMomRange=(-10,5)
        nDeltaMomBins = 150
        nMomResBins = 150
        momResRange=(-3,3)
        nDeltaTimeBins = 100
        deltaTimeRange = [-8,8]
        nUpMatBins = 100
        upMatRange = [-8,1]
        self.HDnMom = HistUtil.new_hist(name="DnMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnTgtMom = HistUtil.new_hist(name="DnMom",label="$N_{ST}$>0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoTgtMom = HistUtil.new_hist(name="DnMom",label="$N_{ST}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoIPAMom = HistUtil.new_hist(name="DnMom",label="$N_{IPA}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoMatMom = HistUtil.new_hist(name="DnMom",label="No Material", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HUpMom = HistUtil.new_hist(name="UpMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpTgtMom = HistUtil.new_hist(name="UpMom",label="$N_{ST}$>0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoMatMom = HistUtil.new_hist(name="UpMom",label="No Material", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)

        # Momentum difference histograms, at upstream and downstream directions
        self.HDnDeltaMom = HistUtil.new_hist(name="DnDeltaMom",label="All", bins=nDeltaMomBins, range=deltaMomRange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+"Downstream $\\Delta$ Momentum at "+self.CompName)
        self.HDnDeltaTgtMom = HistUtil.new_hist(name="DnDeltaMom",label="$N_{ST}$>0", bins=nDeltaMomBins, range=deltaMomRange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+"Downstream $\\Delta$ Momentum at "+self.CompName)
        self.HDnDeltaNoMatMom = HistUtil.new_hist(name="DnDeltaMom",label="No Material", bins=nDeltaMomBins, range=deltaMomRange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+"Downstream $\\Delta$ Momentum at "+self.CompName)

        self.HUpDeltaMom = HistUtil.new_hist(name="UpDeltaMom",label="All", bins=nDeltaMomBins, range=deltaMomRange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+"Upstream $\\Delta$ Momentum at "+self.CompName)
        self.HUpDeltaTgtMom = HistUtil.new_hist(name="UpDeltaMom",label="$N_{ST}$>0", bins=nDeltaMomBins, range=deltaMomRange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+"Upstream $\\Delta$ Momentum at "+self.CompName)
        self.HUpDeltaNoMatMom = HistUtil.new_hist(name="UpDeltaMom",label="No Material", bins=nDeltaMomBins, range=deltaMomRange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+"Upstream $\\Delta$ Momentum at "+self.CompName)

        # Upstream material effect: compare fit and extrapolation (upstream and downstream fits)
        self.HUpMatUpExtrap = HistUtil.new_hist(name="UpMatUpExtrap",label="UpExtrap",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (All)")
        self.HUpMatTgtUpExtrap = HistUtil.new_hist(name="UpMatTgtUpExtrap",label="UpExtrap",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (Target)")
        self.HUpMatNoMatUpExtrap = HistUtil.new_hist(name="UpMatNoMatUpExtrap",label="UpExtrap",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (NoMat)")
        self.HUpMatDnExtrap = HistUtil.new_hist(name="UpMatDnExtrap",label="DnExtrap",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (All)")
        self.HUpMatTgtDnExtrap = HistUtil.new_hist(name="UpMatTgtDnExtrap",label="DnExtrap",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (Target)")
        self.HUpMatNoMatDnExtrap = HistUtil.new_hist(name="UpMatNoMatDnExtrap",label="DnExtrap",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (NoMat)")
        self.HUpMatFit = HistUtil.new_hist(name="UpMatFit",label="Fit",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (All)")
        self.HUpMatTgtFit = HistUtil.new_hist(name="UpMatTgtFit",label="Fit",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (Target)")
        self.HUpMatNoMatFit = HistUtil.new_hist(name="UpMatNoMatFit",label="Fit",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (NoMat)")
        # MC
        self.HUpMatMC = HistUtil.new_hist(name="UpMatMC",label="MC",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (All)")
        self.HUpMatTgtMC = HistUtil.new_hist(name="UpMatTgtMC",label="MC",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (Tgt)")
        self.HUpMatNoMatMC = HistUtil.new_hist(name="UpMatNoMatMC",label="MC",bins=nUpMatBins, range=upMatRange, xlabel="$\\Delta$ Momentum (MeV)",title=self.PDGName+"Upstream Material $\\Delta$ P (NoMat)")
        # Resolution
        self.HUpMomRes = HistUtil.new_hist(name="UpMomRes",label="All",bins=nMomResBins, range=momResRange, xlabel="Reco-MC Momentum (MeV)",title=self.PDGName+"Upstream Track Momentum Resolution at "+self.CompName)
        self.HDnMomRes = HistUtil.new_hist(name="DnMomRes",label="All",bins=nMomResBins, range=momResRange, xlabel="Reco-MC Momentum (MeV)",title=self.PDGName+"Downstream Track Momentum Resolution at "+self.CompName)

        # Time differences
        self.HdnDeltaTime = HistUtil.new_hist(name="DeltaTime",label="Upstream",bins=nDeltaTimeBins, range=deltaTimeRange, xlabel="Downstream - Upstream time (ns)",title=self.PDGName+" $\\Delta$ Time at "+self.CompName)
        self.HupDeltaTime = HistUtil.new_hist(name="DeltaTime",label="Downstream",bins=nDeltaTimeBins, range=deltaTimeRange, xlabel="Downstream - Upstream time (ns)",title=self.PDGName+" $\\Delta$ Time at "+self.CompName)

    def Print(self):
        print("HistReflections, nhits =",self.MinNHits,"Mom Range",self.MomRange,"Comparison at",self.CompName,"PDG",self.PDGName)

    def Loop(self,files,treename):
        # global counts
        NEvent = 0
        NGood = 0
        NMatch = 0
        NFinal = 0
        # append tree to files for uproot
        Files = [None]*len(files)
        for i in range(0,len(files)):
            Files[i] = files[i]+":"+treename
        ibatch = 0
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(Files,filter_name="/evtinfo|trk.trk|trkmc|trksegs|trkmcsim|trksegsmc|trkqual|trksegpars_lh/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            trkQual = batch['trkqual.result']  # track fit quality
            # Separate by upstream, downstream track
            upSegs = segs[:,0] # upstream track fits
            dnSegs = segs[:,1] # downstream track fits
            upFitCon = fitcon[:,0]
            dnFitCon = fitcon[:,1]
            upNhits = nhits[:,0]
            dnNhits = nhits[:,1]
            upTQ = trkQual[:,0]
            dnTQ = trkQual[:,1]
            # MC
            trksegsmc = batch['trksegsmc']
            upSegsMC = trksegsmc[:,0]
            dnSegsMC = trksegsmc[:,1]
            # basic consistency test
            assert((len(upSegs) == len(dnSegs)) & (len(upSegs) == len(upNhits)) & (len(upNhits) == len(dnNhits)) & (len(upTQ) == len(dnTQ)) & (len(upTQ) == len(upSegs)) )
            NEvent += len(upSegs)

            # select based on fit quality
            upGoodFit = (upNhits >= self.MinNHits) & (upFitCon > self.MinFitCon) & (upTQ > self.MinTQ)
            dnGoodFit = (dnNhits >= self.MinNHits) & (dnFitCon > self.MinFitCon) & (dnTQ > self.MinTQ)
            NGood +=  ak.count_nonzero(upGoodFit & dnGoodFit)
            # select the segments of interest and require consistency
            updnseg = (upSegs.sid == self.SID) & (upSegs.mom.Z() > 0.0) & upGoodFit
            dndnseg = (dnSegs.sid == self.SID) & (dnSegs.mom.Z() > 0.0) & dnGoodFit
            upupseg = (upSegs.sid == self.SID) & (upSegs.mom.Z() < 0.0) & upGoodFit
            dnupseg = (dnSegs.sid == self.SID) & (dnSegs.mom.Z() < 0.0) & dnGoodFit
            updncnt = ak.sum(updnseg,axis=1)
            upupcnt = ak.sum(upupseg,axis=1)
            dnupcnt = ak.sum(dnupseg,axis=1)
            dndncnt = ak.sum(dndnseg,axis=1)
            assert((len(upupcnt) == len(updncnt)) & (len(dndncnt) == len(dnupcnt)) & (len(upupcnt) == len(dndncnt)))
            test = [1]*len(upupcnt)
            goodMatch = ((updncnt == dndncnt) & (updncnt == test) & (upupcnt == dnupcnt) & (upupcnt == test))
            NMatch +=  ak.count_nonzero(goodMatch)
            # time difference
            updnTime = upSegs[updnseg & goodMatch].time
            dndnTime = dnSegs[dndnseg & goodMatch].time
            upupTime = upSegs[upupseg & goodMatch].time
            dnupTime = dnSegs[dnupseg & goodMatch].time
            dnDeltaTime = dndnTime-updnTime
            upDeltaTime = dnupTime-upupTime
            self.HdnDeltaTime.fill(np.array(ak.flatten(dnDeltaTime)))
            self.HupDeltaTime.fill(np.array(ak.flatten(upDeltaTime)))
            goodDeltaT = (abs(dnDeltaTime) < self.MaxDeltaT) & (abs(upDeltaTime) < self.MaxDeltaT)
            goodFinal = ak.any(goodMatch & goodDeltaT,highlevel=True, axis=1)
            NFinal +=  ak.count_nonzero(goodFinal)
            # extract properties to test
            updnMom = upSegs[updnseg & goodFinal].mom.magnitude()
            dndnMom = dnSegs[dndnseg & goodFinal].mom.magnitude()
            upupMom = upSegs[upupseg & goodFinal].mom.magnitude()
            dnupMom = dnSegs[dnupseg & goodFinal].mom.magnitude()
            # fit - extrapolated other fit
            dnDeltaMom = dndnMom - updnMom
            upDeltaMom = dnupMom - upupMom
            self.HUpMom.fill(np.array(ak.flatten(updnMom)))
            self.HDnMom.fill(np.array(ak.flatten(dndnMom)))
            self.HDnDeltaMom.fill(np.array(ak.flatten(dnDeltaMom)))
            self.HUpDeltaMom.fill(np.array(ak.flatten(upDeltaMom)))
            # upstream material effect
            upmatupextrap = updnMom - upupMom
            upmatdnextrap = dndnMom - dnupMom
            upmatfit = dndnMom - upupMom
            self.HUpMatUpExtrap.fill(np.array(ak.flatten(upmatupextrap)))
            self.HUpMatDnExtrap.fill(np.array(ak.flatten(upmatdnextrap)))
            self.HUpMatFit.fill(np.array(ak.flatten(upmatfit)))

            # count IPA and target intersections
            nfoil = ak.count_nonzero(upSegs.sid==SID.ST_Foils(),axis=1) + ak.count_nonzero(dnSegs.sid==SID.ST_Foils(),axis=1)
            self.HNST.fill(np.array(nfoil))
            nipa = ak.count_nonzero(upSegs.sid==SID.IPA(),axis=1) +  ak.count_nonzero(dnSegs.sid==SID.IPA(),axis=1)
            self.HNIPA.fill(np.array(nipa))
            # select fits
            hastgt = (nfoil>0)
            nomat = (nipa==0) & (nfoil==0)
            hasTgtInt = goodFinal & hastgt
            nfoiltgt = nfoil[hasTgtInt]
            self.HNSTTgt.fill(np.array(nfoiltgt))
            nipatgt = nipa[hasTgtInt]
            self.HNIPATgt.fill(np.array(nipatgt))
            updnTgtMom = updnMom[hasTgtInt]
            dndnTgtMom = dndnMom[hasTgtInt]
            upupTgtMom = upupMom[hasTgtInt]
            dnupTgtMom = dnupMom[hasTgtInt]
            self.HUpTgtMom.fill(np.array(ak.flatten(updnTgtMom)))
            self.HDnTgtMom.fill(np.array(ak.flatten(dndnTgtMom)))
            dndeltaTgtMom = dndnTgtMom - updnTgtMom
            updeltaTgtMom = dnupTgtMom - upupTgtMom
            self.HDnDeltaTgtMom.fill(np.array(ak.flatten(dndeltaTgtMom)))
            self.HUpDeltaTgtMom.fill(np.array(ak.flatten(updeltaTgtMom)))

            self.HUpFitCon.fill(np.array(upFitCon[hasTgtInt]))
            self.HDnFitCon.fill(np.array(dnFitCon[hasTgtInt]))
            self.HUpNHits.fill(np.array(upNhits[hasTgtInt]))
            self.HDnNHits.fill(np.array(dnNhits[hasTgtInt]))
            self.HUpTQ.fill(np.array(upTQ[hasTgtInt]))
            self.HDnTQ.fill(np.array(dnTQ[hasTgtInt]))

            upmatupextraptgt = updnTgtMom - upupTgtMom
            upmatdnextraptgt = dndnTgtMom - dnupTgtMom
            upmatfittgt = dndnTgtMom - upupTgtMom
            self.HUpMatTgtUpExtrap.fill(np.array(ak.flatten(upmatupextraptgt)))
            self.HUpMatTgtDnExtrap.fill(np.array(ak.flatten(upmatdnextraptgt)))
            self.HUpMatTgtFit.fill(np.array(ak.flatten(upmatfittgt)))

            # no material
            goodNoMat = goodFinal & nomat
            updnNoMatMom = updnMom[goodNoMat]
            dndnNoMatMom = dndnMom[goodNoMat]
            upupNoMatMom = upupMom[goodNoMat]
            dnupNoMatMom = dnupMom[goodNoMat]
            self.HUpNoMatMom.fill(np.array(ak.flatten(updnNoMatMom)))
            self.HDnNoMatMom.fill(np.array(ak.flatten(dndnNoMatMom)))
            dndeltaNoMatMom = dndnNoMatMom - updnNoMatMom
            updeltaNoMatMom = dnupNoMatMom - upupNoMatMom
            self.HDnDeltaNoMatMom.fill(np.array(ak.flatten(dndeltaNoMatMom)))
            self.HUpDeltaNoMatMom.fill(np.array(ak.flatten(updeltaNoMatMom)))

            upmatupextrapnomat = updnNoMatMom - upupNoMatMom
            upmatdnextrapnomat = dndnNoMatMom - dnupNoMatMom
            upmatfitnomat = dndnNoMatMom - upupNoMatMom
            self.HUpMatNoMatUpExtrap.fill(np.array(ak.flatten(upmatupextrapnomat)))
            self.HUpMatNoMatDnExtrap.fill(np.array(ak.flatten(upmatdnextrapnomat)))
            self.HUpMatNoMatFit.fill(np.array(ak.flatten(upmatfitnomat)))

            # MC
            updnsegmc = (upSegsMC.sid == self.SID) & (upSegsMC.mom.Z() > 0.0) & upGoodFit & goodFinal
            dndnsegmc = (dnSegsMC.sid == self.SID) & (dnSegsMC.mom.Z() > 0.0) & dnGoodFit & goodFinal
            upupsegmc = (upSegsMC.sid == self.SID) & (upSegsMC.mom.Z() < 0.0) & upGoodFit & goodFinal
            dnupsegmc = (dnSegsMC.sid == self.SID) & (dnSegsMC.mom.Z() < 0.0) & dnGoodFit & goodFinal
            updnMomMC = upSegsMC[updnsegmc].mom.magnitude()
            dndnMomMC = dnSegsMC[dndnsegmc].mom.magnitude()
            upupMomMC = upSegsMC[upupsegmc].mom.magnitude()
            dnupMomMC = dnSegsMC[dnupsegmc].mom.magnitude()
            updnGoodMC = ak.count_nonzero(updnMomMC,axis=1,keepdims=True)==1
            dndnGoodMC = ak.count_nonzero(dndnMomMC,axis=1,keepdims=True)==1
            upupGoodMC = ak.count_nonzero(upupMomMC,axis=1,keepdims=True)==1
            dnupGoodMC = ak.count_nonzero(dnupMomMC,axis=1,keepdims=True)==1
            upGoodMC = updnGoodMC & upupGoodMC
            dnGoodMC = dndnGoodMC & dnupGoodMC
            upmatmc = updnMomMC[upGoodMC] - upupMomMC[upGoodMC]
            dnmatmc = dndnMomMC[dnGoodMC] - dnupMomMC[dnGoodMC]
            self.HUpMatMC.fill(np.array(upmatmc))

            upres = upupMom[upupGoodMC] - upupMomMC[upupGoodMC]
            dnres = dndnMom[dndnGoodMC] - dndnMomMC[dndnGoodMC]
            self.HUpMomRes.fill(np.array(upres))
            self.HDnMomRes.fill(np.array(dnres))

            updnsegtgtmc = updnsegmc & hasTgtInt & upGoodMC
            dndnsegtgtmc = dndnsegmc & hasTgtInt & dnGoodMC
            upupsegtgtmc = upupsegmc & hasTgtInt & upGoodMC
            dnupsegtgtmc = dnupsegmc & hasTgtInt & dnGoodMC
            updnMomTgtMC = upSegsMC[updnsegtgtmc].mom.magnitude()
            dndnMomTgtMC = dnSegsMC[dndnsegtgtmc].mom.magnitude()
            upupMomTgtMC = upSegsMC[upupsegtgtmc].mom.magnitude()
            dnupMomTgtMC = dnSegsMC[dnupsegtgtmc].mom.magnitude()

            upmattgtmc = updnMomTgtMC - upupMomTgtMC
            dnmattgtmc = dndnMomTgtMC - dnupMomTgtMC
            self.HUpMatTgtMC.fill(np.array(ak.flatten(upmattgtmc)))

            updnsegnomatmc = updnsegmc & goodNoMat & upGoodMC
            dndnsegnomatmc = dndnsegmc & goodNoMat & dnGoodMC
            upupsegnomatmc = upupsegmc & goodNoMat & upGoodMC
            dnupsegnomatmc = dnupsegmc & goodNoMat & dnGoodMC
            updnMomNoMatMC = upSegsMC[updnsegnomatmc].mom.magnitude()
            dndnMomNoMatMC = dnSegsMC[dndnsegnomatmc].mom.magnitude()
            upupMomNoMatMC = upSegsMC[upupsegnomatmc].mom.magnitude()
            dnupMomNoMatMC = dnSegsMC[dnupsegnomatmc].mom.magnitude()

            upmatnomatmc = updnMomNoMatMC - upupMomNoMatMC
            dnmatnomatmc = dndnMomNoMatMC - dnupMomNoMatMC
            self.HUpMatNoMatMC.fill(np.array(ak.flatten(upmatnomatmc)))



        print("\nFrom", NEvent,"total events found", NGood, "with good reco,", NMatch,"matching reflections,", NFinal, "final selections and",self.HUpTgtMom.sum(), "with Target")

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
            HistUtil.save_hist(self.HUpNHits,hdf5file)
            HistUtil.save_hist(self.HDnNHits,hdf5file)
            HistUtil.save_hist(self.HUpFitCon,hdf5file)
            HistUtil.save_hist(self.HDnFitCon,hdf5file)
            HistUtil.save_hist(self.HUpTQ,hdf5file)
            HistUtil.save_hist(self.HDnTQ,hdf5file)

            HistUtil.save_hist(self.HNST,hdf5file)
            HistUtil.save_hist(self.HNSTTgt,hdf5file)
            HistUtil.save_hist(self.HNIPA,hdf5file)
            HistUtil.save_hist(self.HNIPATgt,hdf5file)
            HistUtil.save_hist(self.HDnMom,hdf5file)
            HistUtil.save_hist(self.HDnTgtMom,hdf5file)
            HistUtil.save_hist(self.HDnNoTgtMom,hdf5file)
            HistUtil.save_hist(self.HDnNoIPAMom,hdf5file)
            HistUtil.save_hist(self.HDnNoMatMom,hdf5file)
            HistUtil.save_hist(self.HUpMom,hdf5file)
            HistUtil.save_hist(self.HUpTgtMom,hdf5file)
            HistUtil.save_hist(self.HUpNoMatMom,hdf5file)
            #
            HistUtil.save_hist(self.HDnDeltaMom,hdf5file)
            HistUtil.save_hist(self.HUpDeltaMom,hdf5file)
            HistUtil.save_hist(self.HDnDeltaTgtMom,hdf5file)
            HistUtil.save_hist(self.HUpDeltaTgtMom,hdf5file)
            HistUtil.save_hist(self.HDnDeltaNoMatMom,hdf5file)
            HistUtil.save_hist(self.HUpDeltaNoMatMom,hdf5file)
            #
            HistUtil.save_hist(self.HUpMatUpExtrap,hdf5file)
            HistUtil.save_hist(self.HUpMatDnExtrap,hdf5file)
            HistUtil.save_hist(self.HUpMatFit,hdf5file)
            HistUtil.save_hist(self.HUpMatTgtUpExtrap,hdf5file)
            HistUtil.save_hist(self.HUpMatTgtDnExtrap,hdf5file)
            HistUtil.save_hist(self.HUpMatTgtFit,hdf5file)
            HistUtil.save_hist(self.HUpMatNoMatUpExtrap,hdf5file)
            HistUtil.save_hist(self.HUpMatNoMatDnExtrap,hdf5file)
            HistUtil.save_hist(self.HUpMatNoMatFit,hdf5file)

            HistUtil.save_hist(self.HUpMatMC,hdf5file)
            HistUtil.save_hist(self.HUpMatTgtMC,hdf5file)
            HistUtil.save_hist(self.HUpMatNoMatMC,hdf5file)

            HistUtil.save_hist(self.HUpMomRes,hdf5file)
            HistUtil.save_hist(self.HDnMomRes,hdf5file)
            #
            HistUtil.save_hist(self.HdnDeltaTime,hdf5file)
            HistUtil.save_hist(self.HupDeltaTime,hdf5file)

