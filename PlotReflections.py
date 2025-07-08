#
# plot histograms of reflecting particles
#
from matplotlib import pyplot as plt
import MyHist
import h5py
class PlotReflections(object):
    def __init__(self,savefile):
        self.HUpTQ = MyHist.MyHist(name="HUpTQ",label="Up TrkQual",file=savefile)
        self.HDnTQ = MyHist.MyHist(name="HDnTQ",label="Down TrkQual",file=savefile)
        self.HUpFitCon = MyHist.MyHist(name="HUpFitCon",label="Up FitCon",file=savefile)
        self.HDnFitCon = MyHist.MyHist(name="HDnFitCon",label="Down FitCon",file=savefile)
        self.HUpNHits = MyHist.MyHist(name="HUpNHits",label="Up NActive",file=savefile)
        self.HDnNHits = MyHist.MyHist(name="HDnNHits",label="Down NActive",file=savefile)

        self.HNST = MyHist.MyHist(name="NInter",label="All ST",file=savefile)
        self.HNIPA = MyHist.MyHist(name="NInter",label="All IPA",file=savefile)
        self.HNSTTgt = MyHist.MyHist(name="NInter",label="Target ST",file=savefile)
        self.HNIPATgt = MyHist.MyHist(name="NInter",label="Target IPA",file=savefile)
        self.HDnMom = MyHist.MyHist(name="DnMom",label="All",file=savefile)
        self.HDnTgtMom = MyHist.MyHist(name="DnMom",label="Target",file=savefile)
        self.HDnNoTgtMom = MyHist.MyHist(name="DnMom",label="No Target",file=savefile)
        self.HDnNoIPAMom = MyHist.MyHist(name="DnMom",label="No IPA",file=savefile)
        self.HDnNoMatMom = MyHist.MyHist(name="DnMom",label="No Mat",file=savefile)
        self.HUpMom = MyHist.MyHist(name="UpMom",label="All",file=savefile)
        self.HUpTgtMom = MyHist.MyHist(name="UpMom",label="Target",file=savefile)
        self.HUpNoTgtMom = MyHist.MyHist(name="UpMom",label="No Target",file=savefile)
        self.HUpNoIPAMom = MyHist.MyHist(name="UpMom",label="No IPA",file=savefile)
        self.HUpNoMatMom = MyHist.MyHist(name="UpMom",label="No Mat",file=savefile)
        self.HDeltaMom = MyHist.MyHist(name="DeltaMom",label="All",file=savefile)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="Target",file=savefile)
        self.HDeltaNoTgtMom = MyHist.MyHist(name="DeltaMom",label="No Target",file=savefile)
        self.HDeltaNoIPAMom = MyHist.MyHist(name="DeltaMom",label="No IPA",file=savefile)
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Mat",file=savefile)

    def PlotQuality(self):
        fig, (anhit,afc,atq) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upnhit = self.HUpNHits.plot(anhit)
        dnnhit = self.HDnNHits.plot(anhit)
        anhit.legend(loc="upper right")
        upfc = self.HUpFitCon.plot(afc)
        dnfc = self.HDnFitCon.plot(afc)
        afc.legend(loc="upper right")
        uptq = self.HUpTQ.plot(atq)
        dntq = self.HDnTQ.plot(atq)
        atq.legend(loc="upper right")

    def PlotIntersections(self):
        fig, (cmat,cselmat) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
        nipa = self.HNIPA.plot(cmat)
        nst = self.HNST.plot(cmat)
        cmat.legend(loc="upper right")
        nipasel = self.HNIPATgt.plot(cselmat)
        nstsel = self.HNSTTgt.plot(cselmat)
        cselmat.legend(loc="upper right")

    def PlotMomentum(self):
        fig, (upMom, dnMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upmom = self.HUpMom.plot(upMom)
        uptgtmom = self.HUpTgtMom.plot(upMom)
        upnotgtmom = self.HUpNoTgtMom.plot(upMom)
        upnoipamom = self.HUpNoIPAMom.plot(upMom)
        upnomatmom = self.HUpNoMatMom.plot(upMom)
        upMom.legend(loc="upper right")
        dnmom = self.HDnMom.plot(dnMom)
        dntgtmom = self.HDnTgtMom.plot(dnMom)
        dnnotgtmom = self.HDnNoTgtMom.plot(dnMom)
        dnnoipamom = self.HDnNoIPAMom.plot(dnMom)
        dnnomatmom = self.HDnNoMatMom.plot(dnMom)
        dnMom.legend(loc="upper right")
        delmom = self.HDeltaMom.plot(deltaMom)
        deltgtmom = self.HDeltaTgtMom.plot(deltaMom)
        delnotgtmom = self.HDeltaNoTgtMom.plot(deltaMom)
        delnoipamom = self.HDeltaNoIPAMom.plot(deltaMom)
        delnomatmom = self.HDeltaNoMatMom.plot(deltaMom)
        deltaMom.legend(loc="upper right")
