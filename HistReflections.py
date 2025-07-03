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
        # Surface Ids
        self.SID = sid
        self.CompName = SID.SurfaceName(sid)
        # intersection histograms
        nNMatBins = 20
        NMatRange = [-0.5,19.5]
        self.HNST = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="All ST",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNIPA = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="All IPA",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNSTTgt = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="Target ST",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        self.HNIPATgt = MyHist.MyHist(bins=nNMatBins,range=NMatRange,name="NInter",label="Target IPA",xlabel="N Intersections",title=self.PDGName+" Material Intersections")
        # Momentum histograms
        nMomBins = 100
        momrange=(40.0,200.0)
        nDeltaMomBins = 200
        deltamomrange=(-10,5)
        self.HDnMom = MyHist.MyHist(name="DnMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnTgtMom = MyHist.MyHist(name="DnMom",label="Target", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoTgtMom = MyHist.MyHist(name="DnMom",label="No Target", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoIPAMom = MyHist.MyHist(name="DnMom",label="No IPA", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoMatMom = MyHist.MyHist(name="DnMom",label="No Mat", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HUpMom = MyHist.MyHist(name="UpMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpTgtMom = MyHist.MyHist(name="UpMom",label="Target", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoTgtMom = MyHist.MyHist(name="UpMom",label="No Target", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoIPAMom = MyHist.MyHist(name="UpMom",label="No IPA", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoMatMom = MyHist.MyHist(name="UpMom",label="No Mat", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        # Momentum comparison histograms
        self.HDeltaMom = MyHist.MyHist(name="DeltaMom",label="All", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="Target", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaNoTgtMom = MyHist.MyHist(name="DeltaMom",label="No Target", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaNoIPAMom = MyHist.MyHist(name="DeltaMom",label="No IPA", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Mat", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)

    def Print(self):
        print("HistReflections, nhits =",self.MinNHits,"Mom Range",self.MomRange,"Comparison at",self.CompName,"PDG",self.PDGName)

    def Loop(self,files,treename):
        # global counts
        NReflect = 0
        NRecoReflect = 0
        NsigPartReflect = 0
        # append tree to files for uproot
        Files = [None]*len(files)
        for i in range(0,len(files)):
            Files[i] = files[i]+":"+treename
        ibatch = 0
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(Files,filter_name="/trk|trksegs|trkmcsim|gtrksegsmc/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            fitpdg = batch['trk.pdg']  # track fit consistency
            # compress out unneeded dimensions
            upSegs = segs[:,0] # upstream track fits
            dnSegs = segs[:,1] # downstream track fits
            upFitPDG = fitpdg[:,0]
            dnFitPDG = fitpdg[:,1]
            upFitCon = fitcon[:,0]
            dnFitCon = fitcon[:,1]
            upNhits = nhits[:,0]
            dnNhits = nhits[:,1]
            # basic consistency test
            assert((len(upSegs) == len(dnSegs)) & (len(upSegs) == len(upNhits)) & (len(upNhits) == len(dnNhits)) )
            # select fits that match PDG code
            upSigPart = (upFitPDG == self.PDG)
            dnSigPart = (dnFitPDG == self.PDG)
            sigPartFit = upSigPart & dnSigPart
            # find momentum to test
            upmom = upSegs[(upSegs.sid == self.SID) & (upSegs.mom.Z() < 0.0)].mom.magnitude()
            dnmom = dnSegs[(dnSegs.sid == self.SID) & (dnSegs.mom.Z() > 0.0)].mom.magnitude()
            # check up and down match
            updownmatch = ak.num(upmom) == ak.num(dnmom)
            # select based on fit quality
            upGoodFit = (upNhits >= self.MinNHits) & (upFitCon > self.MinFitCon)
            dnGoodFit = (dnNhits >= self.MinNHits) & (dnFitCon > self.MinFitCon)
            goodReco = updownmatch & upGoodFit & dnGoodFit & sigPartFit
            NReflect +=  ak.count_nonzero(upNhits)
            NRecoReflect += sum(goodReco)
            # select based on time difference at tracker entrance
            upEntTime = upSegs[(upSegs.sid==SID.TT_Front()) & (upSegs.mom.z() > 0.0) & goodReco ].time
            dnEntTime = dnSegs[(dnSegs.sid==SID.TT_Front()) & (dnSegs.mom.z() > 0.0) & goodReco ].time
            deltaEntTime = dnEntTime-upEntTime
            goodDeltaT = abs(deltaEntTime) < self.MaxDeltaT
            # total momentum of upstream and downstream fits at the comparison point
            upmom = upSegs[(upSegs.sid == self.SID) & (upSegs.mom.Z() < 0.0)].mom.magnitude()
            dnmom = dnSegs[(dnSegs.sid == self.SID) & (dnSegs.mom.Z() > 0.0)].mom.magnitude()
            # flatten
            upMom = np.array(ak.flatten(upmom[goodReco],axis=1))
            dnMom = np.array(ak.flatten(dnmom[goodReco],axis=1))
            if len(upMom) != len(dnMom):
                print()
                print("Upstream and Downstream fits don't match!",len(upMom),len(dnMom))
                continue
            # good fits
            goodFit = goodReco & goodDeltaT & updownmatch
            goodFit = ak.ravel(goodFit)
            NsigPartReflect += sum(goodFit)
            self.HUpMom.fill(upMom[goodFit])
            self.HDnMom.fill(dnMom[goodFit])
            deltaMom = dnMom - upMom
            self.HDeltaMom.fill(deltaMom[goodFit])
            # count IPA and target intersections
            nfoil = np.array(ak.count_nonzero(upSegs[goodReco].sid==SID.ST_Foils(),axis=1))
            self.HNST.fill(nfoil)
            nipa = np.array(ak.count_nonzero(upSegs[goodReco].sid==SID.IPA(),axis=1))
            self.HNIPA.fill(nipa)
            # select fits
            goodMom = (dnMom > self.MomRange[0]) & (dnMom < self.MomRange[1]) & (upMom > self.MomRange[0]) & (upMom < self.MomRange[1])
            hastgt = (nfoil>0)
            noipa = (nipa==0)
            notgt = (nfoil==0)
            nomat = notgt & noipa
            goodTgt = goodFit & goodMom & hastgt
            nfoilsel = nfoil[goodTgt]
            self.HNSTTgt.fill(nfoilsel)
            nipasel = nipa[goodTgt]
            self.HNIPATgt.fill(nipasel)
            upTgtMom = upMom[goodTgt]
            dnTgtMom = dnMom[goodTgt]
            self.HUpTgtMom.fill(upTgtMom)
            self.HDnTgtMom.fill(dnTgtMom)
            deltaTgtMom = dnTgtMom - upTgtMom
            self.HDeltaTgtMom.fill(deltaTgtMom)
            # no target
            goodNoTgt = goodFit & goodMom & notgt
            upNoTgtMom = upMom[goodNoTgt]
            dnNoTgtMom = dnMom[goodNoTgt]
            self.HUpNoTgtMom.fill(upNoTgtMom)
            self.HDnNoTgtMom.fill(dnNoTgtMom)
            deltaNoTgtMom = dnNoTgtMom - upNoTgtMom
            self.HDeltaNoTgtMom.fill(deltaNoTgtMom)
            # no IPA
            goodNoIPA = goodFit & noipa
            upNoIPAMom = upMom[goodNoIPA]
            dnNoIPAMom = dnMom[goodNoIPA]
            self.HUpNoIPAMom.fill(upNoIPAMom)
            self.HDnNoIPAMom.fill(dnNoIPAMom)
            deltaNoIPAMom = dnNoIPAMom - upNoIPAMom
            self.HDeltaNoIPAMom.fill(deltaNoIPAMom)
            # no material
            goodNoMat = goodFit & nomat
            upNoMatMom = upMom[goodNoMat]
            dnNoMatMom = dnMom[goodNoMat]
            self.HUpNoMatMom.fill(upNoMatMom)
            self.HDnNoMatMom.fill(dnNoMatMom)
            deltaNoMatMom = dnNoMatMom - upNoMatMom
            self.HDeltaNoMatMom.fill(deltaNoMatMom)

        print()
        print("From", NReflect,"total reflections found", NRecoReflect,"with good quality reco,", NsigPartReflect, "confirmed",self.PDGName,"and",self.HUpTgtMom.integral(), "Target",self.MomRange)

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
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
            self.HUpNoTgtMom.save(hdf5file)
            self.HUpNoIPAMom.save(hdf5file)
            self.HDeltaMom.save(hdf5file)
            self.HDeltaTgtMom.save(hdf5file)
            self.HDeltaNoTgtMom.save(hdf5file)
            self.HDeltaNoIPAMom.save(hdf5file)
            self.HDeltaNoMatMom.save(hdf5file)


