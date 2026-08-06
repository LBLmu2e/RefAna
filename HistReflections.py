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
import MyHist
import h5py

class HistReflections(object):
    def __init__(self,momrange,pdg,sid):
        # PDG cods of signal and background particles
        self.PDG = pdg
        PDGNames = {-13:"$\\mu^+$",-11:"$e^+$",11:"$e^-$",13:"$\\mu^-$"}
        self.PDGName = PDGNames[self.PDG]
        # setup cuts; these should be overrideable FIXME
        self.MinNHits = 20
        self.MinFitCon = 1.0e-5
        self.MaxDeltaT = 5.0 # nsec
        self.MomRange = momrange
        self.MinTQ = 0.8 # ANN output
        # Surface Ids
        self.SID = sid
        self.CompName = SID.SurfaceName(sid)
        # fit quality histograms
        self.HUpTQ = MyHist.MyHist(name="HUpTQ",bins=100,range=[0.0,1.0],label="Up TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HDnTQ = MyHist.MyHist(name="HDnTQ",bins=100,range=[0.0,1.0],label="Down TrkQual",title="Track Quality",xlabel="ANN Result")
        self.HUpFitCon = MyHist.MyHist(name="HUpFitCon",bins=100,range=[0.0,1.0],label="Up FitCon",title="Fit Consistency",xlabel="")
        self.HDnFitCon = MyHist.MyHist(name="HDnFitCon",bins=100,range=[0.0,1.0],label="Down FitCon",title="Fit Consistency",xlabel="")
        self.HUpNHits = MyHist.MyHist(name="HUpNHits",bins=100,range=[0.5,100.5],label="Up NActive",title="Fit N Hits",xlabel="N Hits")
        self.HDnNHits = MyHist.MyHist(name="HDnNHits",bins=100,range=[0.5,100.5],label="Down NActive",title="Fit N Hits",xlabel="N Hits")

        # intersection histograms
        nNMatBins = 31
        NMatRange = [-0.5,30.5]
        self.HNST = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="All ST",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNIPA = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="All IPA",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNSTTgt = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="Target ST",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNIPATgt = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="Target IPA",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        # Momentum histograms
        nMomBins = 100
        momrange=(40.0,200.0)
        nDeltaMomBins = 200
        deltamomrange=(-10,5)
        nDeltaTimeBins = 100
        deltaTimeRange = [-8,8]
        self.HDnMom = MyHist.MyHist(name="DnMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnTgtMom = MyHist.MyHist(name="DnMom",label="$N_{ST}$>0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoTgtMom = MyHist.MyHist(name="DnMom",label="$N_{ST}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoIPAMom = MyHist.MyHist(name="DnMom",label="$N_{IPA}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoMatMom = MyHist.MyHist(name="DnMom",label="No Material", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HUpMom = MyHist.MyHist(name="UpMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpTgtMom = MyHist.MyHist(name="UpMom",label="$N_{ST}$>0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoMatMom = MyHist.MyHist(name="UpMom",label="No Material", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        # Momentum comparison histograms
        self.HDeltaMom = MyHist.MyHist(name="DeltaMom",label="All", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="$N_{ST}$>0", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Material", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)

        self.HdnDeltaTime = MyHist.MyHist(name="DeltaTime",label="Upstream",bins=nDeltaTimeBins, range=deltaTimeRange, xlabel="Downstream - Upstream time (ns)",title=self.PDGName+" $\\Delta$ Time at "+self.CompName)
        self.HupDeltaTime = MyHist.MyHist(name="DeltaTime",label="Downstream",bins=nDeltaTimeBins, range=deltaTimeRange, xlabel="Downstream - Upstream time (ns)",title=self.PDGName+" $\\Delta$ Time at "+self.CompName)

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
            # basic consistency test
            assert((len(upSegs) == len(dnSegs)) & (len(upSegs) == len(upNhits)) & (len(upNhits) == len(dnNhits)) & (len(upTQ) == len(dnTQ)) & (len(upTQ) == len(upSegs)) )
            NEvent += len(upSegs)

            # select based on fit quality
            upGoodFit = (upNhits >= self.MinNHits) & (upFitCon > self.MinFitCon) & (upTQ > self.MinTQ)
            dnGoodFit = (dnNhits >= self.MinNHits) & (dnFitCon > self.MinFitCon) & (dnTQ > self.MinTQ)

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
            # extract properties to test
            updnMom = upSegs[updnseg & goodMatch].mom.magnitude()
            dndnMom = dnSegs[dndnseg & goodMatch].mom.magnitude()
            upupMom = upSegs[upupseg & goodMatch].mom.magnitude()
            dnupMom = dnSegs[dnupseg & goodMatch].mom.magnitude()

            dnDeltaMom = dndnMom - updnMom
            upDeltaMom = dnupMom - upupMom
            goodMom = (dndnMom > self.MomRange[0]) & (dndnMom < self.MomRange[1]) & (updnMom > self.MomRange[0]) & (updnMom < self.MomRange[1])
            updnTime = upSegs[updnseg & goodMatch].time
            dndnTime = dnSegs[dndnseg & goodMatch].time
            upupTime = upSegs[upupseg & goodMatch].time
            dnupTime = dnSegs[dnupseg & goodMatch].time
            dnDeltaTime = dndnTime-updnTime
            upDeltaTime = dnupTime-upupTime
            self.HdnDeltaTime.fill(np.array(ak.flatten(dnDeltaTime)))
            self.HupDeltaTime.fill(np.array(ak.flatten(upDeltaTime)))

            goodDeltaT = (abs(dnDeltaTime) < self.MaxDeltaT) & (abs(upDeltaTime) < self.MaxDeltaT)
            goodFinal = goodMatch & goodDeltaT
            NGood +=  ak.count_nonzero(upGoodFit)
            NMatch +=  ak.count_nonzero(goodMatch)
            NFinal +=  ak.count_nonzero(goodFinal)
            #
            self.HUpMom.fill(np.array(ak.flatten(updnMom)))
            self.HDnMom.fill(np.array(ak.flatten(dndnMom)))
            self.HDeltaMom.fill(np.array(ak.flatten(dnDeltaMom)))
            # count IPA and target intersections
            nfoil = ak.count_nonzero(upSegs.sid==SID.ST_Foils(),axis=1) + ak.count_nonzero(dnSegs.sid==SID.ST_Foils(),axis=1)
            self.HNST.fill(np.array(nfoil))
            nipa = ak.count_nonzero(upSegs.sid==SID.IPA(),axis=1) +  ak.count_nonzero(dnSegs.sid==SID.IPA(),axis=1)
            self.HNIPA.fill(np.array(nipa))
            # select fits
            hastgt = (nfoil>0)
            nomat = (nipa==0) & (nfoil==0)
            hasTgtInt = ak.flatten(goodFinal & hastgt)
            nfoilsel = nfoil[hasTgtInt]
            self.HNSTTgt.fill(np.array(nfoilsel))
            nipasel = nipa[hasTgtInt]
            self.HNIPATgt.fill(np.array(nipasel))
            upTgtMom = updnMom[hasTgtInt]
            dnTgtMom = dndnMom[hasTgtInt]
            self.HUpTgtMom.fill(np.array(ak.flatten(upTgtMom)))
            self.HDnTgtMom.fill(np.array(ak.flatten(dnTgtMom)))
            deltaTgtMom = dnTgtMom - upTgtMom
            self.HDeltaTgtMom.fill(np.array(ak.flatten(deltaTgtMom)))

            self.HUpFitCon.fill(np.array(upFitCon[hasTgtInt]))
            self.HDnFitCon.fill(np.array(dnFitCon[hasTgtInt]))
            self.HUpNHits.fill(np.array(upNhits[hasTgtInt]))
            self.HDnNHits.fill(np.array(dnNhits[hasTgtInt]))
            self.HUpTQ.fill(np.array(upTQ[hasTgtInt]))
            self.HDnTQ.fill(np.array(dnTQ[hasTgtInt]))

            # no material
            goodNoMat = goodFinal & nomat
            upNoMatMom = updnMom[goodNoMat]
            dnNoMatMom = dndnMom[goodNoMat]
            self.HUpNoMatMom.fill(np.array(ak.flatten(upNoMatMom)))
            self.HDnNoMatMom.fill(np.array(ak.flatten(dnNoMatMom)))
            deltaNoMatMom = dnNoMatMom - upNoMatMom
            self.HDeltaNoMatMom.fill(np.array(ak.flatten(deltaNoMatMom)))

        print()
        print("From", NEvent,"total events found", NGood, "with good reco,", NMatch,"matching reflections,", NFinal, "final selections and",self.HUpTgtMom.integral(), "with Target")

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
            self.HUpNHits.save(hdf5file)
            self.HDnNHits.save(hdf5file)
            self.HUpFitCon.save(hdf5file)
            self.HDnFitCon.save(hdf5file)
            self.HUpTQ.save(hdf5file)
            self.HDnTQ.save(hdf5file)

            self.HNST.save(hdf5file)
            self.HNSTTgt.save(hdf5file)
            self.HNIPA.save(hdf5file)
            self.HNIPATgt.save(hdf5file)
            self.HDnMom.save(hdf5file)
            self.HDnTgtMom.save(hdf5file)
            self.HDnNoTgtMom.save(hdf5file)
            self.HDnNoIPAMom.save(hdf5file)
            self.HDnNoMatMom.save(hdf5file)
            self.HUpMom.save(hdf5file)
            self.HUpTgtMom.save(hdf5file)
            self.HUpNoMatMom.save(hdf5file)
            #
            self.HDeltaMom.save(hdf5file)
            self.HDeltaTgtMom.save(hdf5file)
            self.HDeltaNoMatMom.save(hdf5file)
            #
            self.HdnDeltaTime.save(hdf5file)
            self.HupDeltaTime.save(hdf5file)

