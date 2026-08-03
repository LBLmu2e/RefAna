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
        self.NUpHits = []
        self.NDnHits = []

        # intersection histograms
        nNMatBins = 15
        NMatRange = [-0.5,14.5]
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
        self.HDnTgtMom = MyHist.MyHist(name="DnMom",label="$N_{ST}$>0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoTgtMom = MyHist.MyHist(name="DnMom",label="$N_{ST}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoIPAMom = MyHist.MyHist(name="DnMom",label="$N_{IPA}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HDnNoMatMom = MyHist.MyHist(name="DnMom",label="No Material", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
        self.HUpMom = MyHist.MyHist(name="UpMom",label="All", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpTgtMom = MyHist.MyHist(name="UpMom",label="$N_{ST}$>0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoTgtMom = MyHist.MyHist(name="UpMom",label="$N_{ST}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoIPAMom = MyHist.MyHist(name="UpMom",label="$N_{IPA}$==0", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        self.HUpNoMatMom = MyHist.MyHist(name="UpMom",label="No Material", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
        # Momentum comparison histograms
        self.HDeltaMom = MyHist.MyHist(name="DeltaMom",label="All", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="$N_{ST}$>0", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaNoTgtMom = MyHist.MyHist(name="DeltaMom",label="$N_{ST}$==0", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaNoIPAMom = MyHist.MyHist(name="DeltaMom",label="$N_{IPA}$==0", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Material", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)

#        self.HUpTgtMomB12 = MyHist.MyHist(name="UpMom",label="B12", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
#        self.HUpTgtMomB34 = MyHist.MyHist(name="UpMom",label="B34", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
#        self.HUpTgtMomB56 = MyHist.MyHist(name="UpMom",label="B56", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
#        self.HUpTgtMomB78 = MyHist.MyHist(name="UpMom",label="B78", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
#        self.HUpTgtMomB9p = MyHist.MyHist(name="UpMom",label="B9p", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title=self.PDGName+" Upstream Momentum at "+self.CompName)
#
#        self.HDnTgtMomB12 = MyHist.MyHist(name="DnMom",label="B12", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
#        self.HDnTgtMomB34 = MyHist.MyHist(name="DnMom",label="B34", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
#        self.HDnTgtMomB56 = MyHist.MyHist(name="DnMom",label="B56", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
#        self.HDnTgtMomB78 = MyHist.MyHist(name="DnMom",label="B78", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
#        self.HDnTgtMomB9p = MyHist.MyHist(name="DnMom",label="B9p", bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title=self.PDGName+" Downstream Momentum at "+self.CompName)
#
#        self.HDeltaTgtMomB12 = MyHist.MyHist(name="DeltaMom",label="B12", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
#        self.HDeltaTgtMomB34 = MyHist.MyHist(name="DeltaMom",label="B34", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
#        self.HDeltaTgtMomB56 = MyHist.MyHist(name="DeltaMom",label="B56", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
#        self.HDeltaTgtMomB78 = MyHist.MyHist(name="DeltaMom",label="B78", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)
#        self.HDeltaTgtMomB9p = MyHist.MyHist(name="DeltaMom",label="B9p", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title=self.PDGName+" $\\Delta$ Momentum at "+self.CompName)

    def Print(self):
        print("HistReflections, nhits =",self.MinNHits,"Mom Range",self.MomRange,"Comparison at",self.CompName,"PDG",self.PDGName)

    def Loop(self,files,treename):
        # global counts
        NEvent = 0
        NReflect = 0
        NSelect = 0
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
            upTQ = trkQual[:,0]
            dnTQ = trkQual[:,1]
            # basic consistency test
            assert((len(upSegs) == len(dnSegs)) & (len(upSegs) == len(upNhits)) & (len(upNhits) == len(dnNhits)) & (len(upTQ) == len(dnTQ)) & (len(upTQ) == len(upSegs)) )
            NEvent += len(upSegs)

            # select based on fit quality
            upGoodFit = (upNhits >= self.MinNHits) & (upFitCon > self.MinFitCon) & (upTQ > self.MinTQ)
            dnGoodFit = (dnNhits >= self.MinNHits) & (dnFitCon > self.MinFitCon) & (dnTQ > self.MinTQ)

            # select the segments of interest: there should be one of these. Eventually make the momentum sign an option TODO:
            upent = (upSegs.sid == self.SID) & (upSegs.mom.Z() > 0.0) & upGoodFit
            dnent = (dnSegs.sid == self.SID) & (dnSegs.mom.Z() > 0.0) & dnGoodFit
            upentcnt = ak.sum(upent,axis=1)
            dnentcnt = ak.sum(dnent,axis=1)
            goodMatch = (upentcnt == dnentcnt)

            # extract properties to test
            upMom = upSegs[upent & goodMatch].mom.magnitude()
            dnMom = dnSegs[dnent & goodMatch].mom.magnitude()
            deltaMom = dnMom - upMom
            goodMom = (dnMom > self.MomRange[0]) & (dnMom < self.MomRange[1]) & (upMom > self.MomRange[0]) & (upMom < self.MomRange[1])
#            print(upMom,dnMom,len(upMom),len(dnMom))
            deltaEntMom = upMom -dnMom
            upEntTime = upSegs[upent & goodMatch].time
            dnEntTime = dnSegs[dnent & goodMatch].time
            deltaEntTime = dnEntTime-upEntTime
            goodDeltaT = abs(deltaEntTime) < self.MaxDeltaT
            goodFinal = goodMatch & goodMom & goodDeltaT
            NReflect +=  ak.count_nonzero(goodMatch)
            NSelect +=  ak.count_nonzero(goodMatch)
            #
            self.HUpMom.fill(np.array(ak.flatten(upMom)))
            self.HDnMom.fill(np.array(ak.flatten(dnMom)))
            self.HDeltaMom.fill(np.array(ak.flatten(deltaMom)))
            # count IPA and target intersections
            nfoil = ak.count_nonzero(upSegs.sid==SID.ST_Foils(),axis=1)
            self.HNST.fill(np.array(nfoil))
            nipa = ak.count_nonzero(upSegs.sid==SID.IPA(),axis=1)
            self.HNIPA.fill(np.array(nipa))
            # select fits
            hastgt = (nfoil>0)
            noipa = (nipa==0)
            notgt = (nfoil==0)
            nomat = notgt & noipa
            goodTgt = ak.flatten(goodMatch & goodMom & hastgt)
            nfoilsel = nfoil[goodTgt]
            self.HNSTTgt.fill(np.array(nfoilsel))
            nipasel = nipa[goodTgt]
            self.HNIPATgt.fill(np.array(nipasel))
            upTgtMom = upMom[goodTgt]
            dnTgtMom = dnMom[goodTgt]
            self.HUpTgtMom.fill(np.array(ak.flatten(upTgtMom)))
            self.HDnTgtMom.fill(np.array(ak.flatten(dnTgtMom)))
            deltaTgtMom = dnTgtMom - upTgtMom
            self.HDeltaTgtMom.fill(np.array(ak.flatten(deltaTgtMom)))
#            ### BINNED FITS ###
#
#            B12 = np.logical_or((nfoil == 1), (nfoil == 2))
#            upTgtMomB12 = upMom[B12]
#            dnTgtMomB12 = dnMom[B12]
#            self.HUpTgtMomB12.fill(upTgtMomB12)
#            self.HDnTgtMomB12.fill(dnTgtMomB12)
#            deltaTgtMomB12 = dnTgtMomB12 - upTgtMomB12
#            self.HDeltaTgtMomB12.fill(deltaTgtMomB12)
#
#            B34 = np.logical_or((nfoil == 3), (nfoil == 4))
#            upTgtMomB34 = upMom[B34]
#            dnTgtMomB34 = dnMom[B34]
#            self.HUpTgtMomB34.fill(upTgtMomB34)
#            self.HDnTgtMomB34.fill(dnTgtMomB34)
#            deltaTgtMomB34 = dnTgtMomB34 - upTgtMomB34
#            self.HDeltaTgtMomB34.fill(deltaTgtMomB34)
#
#            B56 = np.logical_or((nfoil == 5), (nfoil == 5))
#            upTgtMomB56 = upMom[B56]
#            dnTgtMomB56 = dnMom[B56]
#            self.HUpTgtMomB56.fill(upTgtMomB56)
#            self.HDnTgtMomB56.fill(dnTgtMomB56)
#            deltaTgtMomB56 = dnTgtMomB56 - upTgtMomB56
#            self.HDeltaTgtMomB56.fill(deltaTgtMomB56)
#
#            B78 = np.logical_or((nfoil == 7), (nfoil == 8))
#            upTgtMomB78 = upMom[B78]
#            dnTgtMomB78 = dnMom[B78]
#            self.HUpTgtMomB78.fill(upTgtMomB78)
#            self.HDnTgtMomB78.fill(dnTgtMomB78)
#            deltaTgtMomB78 = dnTgtMomB78 - upTgtMomB78
#            self.HDeltaTgtMomB78.fill(deltaTgtMomB78)
#
#            B9p = (nfoil >= 9)
#            upTgtMomB9p = upMom[B9p]
#            dnTgtMomB9p = dnMom[B9p]
#            self.HUpTgtMomB9p.fill(upTgtMomB9p)
#            self.HDnTgtMomB9p.fill(dnTgtMomB9p)
#            deltaTgtMomB9p = dnTgtMomB9p - upTgtMomB9p
#            self.HDeltaTgtMomB9p.fill(deltaTgtMomB9p)


            self.HUpFitCon.fill(np.array(upFitCon[goodTgt]))
            self.HDnFitCon.fill(np.array(dnFitCon[goodTgt]))
            self.HUpNHits.fill(np.array(upNhits[goodTgt]))
            self.HDnNHits.fill(np.array(dnNhits[goodTgt]))
            self.HUpTQ.fill(np.array(upTQ[goodTgt]))
            self.HDnTQ.fill(np.array(dnTQ[goodTgt]))

            self.NUpHits.extend(np.array(upNhits[goodTgt]))
            self.NDnHits.extend(np.array(dnNhits[goodTgt]))

            # no target
            goodNoTgt = goodMatch & goodMom & notgt
            upNoTgtMom = upMom[goodNoTgt]
            dnNoTgtMom = dnMom[goodNoTgt]
            self.HUpNoTgtMom.fill(np.array(ak.flatten(upNoTgtMom)))
            self.HDnNoTgtMom.fill(np.array(ak.flatten(dnNoTgtMom)))
            deltaNoTgtMom = dnNoTgtMom - upNoTgtMom
            self.HDeltaNoTgtMom.fill(np.array(ak.flatten(deltaNoTgtMom)))
            # no IPA
            goodNoIPA = goodMatch & noipa
            upNoIPAMom = upMom[goodNoIPA]
            dnNoIPAMom = dnMom[goodNoIPA]
            self.HUpNoIPAMom.fill(np.array(ak.flatten(upNoIPAMom)))
            self.HDnNoIPAMom.fill(np.array(ak.flatten(dnNoIPAMom)))
            deltaNoIPAMom = dnNoIPAMom - upNoIPAMom
            self.HDeltaNoIPAMom.fill(np.array(ak.flatten(deltaNoIPAMom)))
            # no material
            goodNoMat = goodMatch & nomat
            upNoMatMom = upMom[goodNoMat]
            dnNoMatMom = dnMom[goodNoMat]
            self.HUpNoMatMom.fill(np.array(ak.flatten(upNoMatMom)))
            self.HDnNoMatMom.fill(np.array(ak.flatten(dnNoMatMom)))
            deltaNoMatMom = dnNoMatMom - upNoMatMom
            self.HDeltaNoMatMom.fill(np.array(ak.flatten(deltaNoMatMom)))

        print()
        print("From", NEvent,"total events found", NReflect," matching reflections with good quality reco,", NSelect, "final selections and",self.HUpTgtMom.integral(), "with Target")

    def Plot(self):
        fig, (hits) = plt.subplots(1,1,layout='constrained', figsize=(5,5))
        hist = hits.hist2d(self.NUpHits,self.NDnHits,label="NHits",bins=[100,100],range=[[-0.5,99.5],[-0.5,99.5]],density=True,norm="linear")
        hits.set_title("Reflection Fit N Hits")
        hits.set_xlabel("Upstream N Hits")
        hits.set_ylabel("Downstream N hits")


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
            self.HUpNoTgtMom.save(hdf5file)
            self.HUpNoIPAMom.save(hdf5file)
            self.HDeltaMom.save(hdf5file)
            self.HDeltaTgtMom.save(hdf5file)
            self.HDeltaNoTgtMom.save(hdf5file)
            self.HDeltaNoIPAMom.save(hdf5file)
            self.HDeltaNoMatMom.save(hdf5file)

#            self.HUpTgtMomB12.save(hdf5file)
#            self.HUpTgtMomB34.save(hdf5file)
#            self.HUpTgtMomB56.save(hdf5file)
#            self.HUpTgtMomB78.save(hdf5file)
#            self.HUpTgtMomB9p.save(hdf5file)
#
#            self.HDnTgtMomB12.save(hdf5file)
#            self.HDnTgtMomB34.save(hdf5file)
#            self.HDnTgtMomB56.save(hdf5file)
#            self.HDnTgtMomB78.save(hdf5file)
#            self.HDnTgtMomB9p.save(hdf5file)
#
#            self.HDeltaTgtMomB12.save(hdf5file)
#            self.HDeltaTgtMomB34.save(hdf5file)
#            self.HDeltaTgtMomB56.save(hdf5file)
#            self.HDeltaTgtMomB78.save(hdf5file)
#            self.HDeltaTgtMomB9p.save(hdf5file)
