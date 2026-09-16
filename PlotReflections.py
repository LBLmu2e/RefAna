#
# plot histograms of reflecting particles
#
from matplotlib import pyplot as plt
import HistUtil
import math
import FitFunctions
from scipy.optimize import curve_fit

class PlotReflections(object):
    def __init__(self,savefile):
        self.HUpTQ = HistUtil.load_hist(name="HUpTQ",label="Up TrkQual",file=savefile)
        self.HDnTQ = HistUtil.load_hist(name="HDnTQ",label="Down TrkQual",file=savefile)
        self.HUpFitCon = HistUtil.load_hist(name="HUpFitCon",label="Up FitCon",file=savefile)
        self.HDnFitCon = HistUtil.load_hist(name="HDnFitCon",label="Down FitCon",file=savefile)
        self.HUpNHits = HistUtil.load_hist(name="HUpNHits",label="Up NActive",file=savefile)
        self.HDnNHits = HistUtil.load_hist(name="HDnNHits",label="Down NActive",file=savefile)

        self.HNST = HistUtil.load_hist(name="NInter",label="All ST",file=savefile)
        self.HNIPA = HistUtil.load_hist(name="NInter",label="All IPA",file=savefile)
        self.HNSTTgt = HistUtil.load_hist(name="NInter",label="Target ST",file=savefile)
        self.HNIPATgt = HistUtil.load_hist(name="NInter",label="Target IPA",file=savefile)
        self.HDnMom = HistUtil.load_hist(name="DnMom",label="All",file=savefile)
        self.HDnTgtMom = HistUtil.load_hist(name="DnMom",label="$N_{ST}$>0",file=savefile)
        self.HDnNoMatMom = HistUtil.load_hist(name="DnMom",label="No Material",file=savefile)
        self.HUpMom = HistUtil.load_hist(name="UpMom",label="All",file=savefile)
        self.HUpTgtMom = HistUtil.load_hist(name="UpMom",label="$N_{ST}$>0",file=savefile)
        self.HUpNoMatMom = HistUtil.load_hist(name="UpMom",label="No Material",file=savefile)
        self.HDnDeltaMom = HistUtil.load_hist(name="DnDeltaMom",label="All",file=savefile)
        self.HDnDeltaTgtMom = HistUtil.load_hist(name="DnDeltaMom",label="$N_{ST}$>0",file=savefile)
        self.HDnDeltaNoMatMom = HistUtil.load_hist(name="DnDeltaMom",label="No Material",file=savefile)
        self.HUpDeltaMom = HistUtil.load_hist(name="UpDeltaMom",label="All",file=savefile)
        self.HUpDeltaTgtMom = HistUtil.load_hist(name="UpDeltaMom",label="$N_{ST}$>0",file=savefile)
        self.HUpDeltaNoMatMom = HistUtil.load_hist(name="UpDeltaMom",label="No Material",file=savefile)
        #
        self.HUpMatUpExtrap = HistUtil.load_hist(name="UpMatUpExtrap",label="UpExtrap",file=savefile)
        self.HUpMatDnExtrap = HistUtil.load_hist(name="UpMatDnExtrap",label="DnExtrap",file=savefile)
        self.HUpMatFit = HistUtil.load_hist(name="UpMatFit",label="Fit",file=savefile)
        self.HUpMatTgtUpExtrap = HistUtil.load_hist(name="UpMatTgtUpExtrap",label="UpExtrap",file=savefile)
        self.HUpMatTgtDnExtrap = HistUtil.load_hist(name="UpMatTgtDnExtrap",label="DnExtrap",file=savefile)
        self.HUpMatTgtFit = HistUtil.load_hist(name="UpMatTgtFit",label="Fit",file=savefile)
        self.HUpMatNoMatUpExtrap = HistUtil.load_hist(name="UpMatNoMatUpExtrap",label="UpExtrap",file=savefile)
        self.HUpMatNoMatDnExtrap = HistUtil.load_hist(name="UpMatNoMatDnExtrap",label="DnExtrap",file=savefile)
        self.HUpMatNoMatFit = HistUtil.load_hist(name="UpMatNoMatFit",label="Fit",file=savefile)
        #
        self.HUpMatMC = HistUtil.load_hist(name="UpMatMC",label="MC",file=savefile)
        self.HUpMatTgtMC = HistUtil.load_hist(name="UpMatTgtMC",label="MC",file=savefile)
        self.HUpMatNoMatMC = HistUtil.load_hist(name="UpMatNoMatMC",label="MC",file=savefile)
        #
        self.HUpMomRes = HistUtil.load_hist(name="UpMomRes",label="All",file=savefile)
        self.HDnMomRes = HistUtil.load_hist(name="DnMomRes",label="All",file=savefile)
        #
        self.HupDeltaTime = HistUtil.load_hist(name="DeltaTime",label="Upstream",file=savefile)
        self.HdnDeltaTime = HistUtil.load_hist(name="DeltaTime",label="Downstream",file=savefile)


    def PlotQuality(self):
        fig, (anhit,afc,atq) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upnhit = HistUtil.plot(self.HUpNHits,anhit)
        dnnhit = HistUtil.plot(self.HDnNHits,anhit)
        anhit.legend(loc="upper right")
        upfc = HistUtil.plot(self.HUpFitCon,afc)
        dnfc = HistUtil.plot(self.HDnFitCon,afc)
        afc.legend(loc="upper right")
        uptq = HistUtil.plot(self.HUpTQ,atq)
        dntq = HistUtil.plot(self.HDnTQ,atq)
        atq.legend(loc="upper right")

    def PlotIntersections(self):
        fig, (cmat,cselmat) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        nipa = HistUtil.plot(self.HNIPA,cmat)
        nst = HistUtil.plot(self.HNST,cmat)
        cmat.legend(loc="upper right")
        nipasel = HistUtil.plot(self.HNIPATgt,cselmat)
        nstsel = HistUtil.plot(self.HNSTTgt,cselmat)
        cselmat.legend(loc="upper right")

    def PlotMomentum(self):
        fig, (upMom, dnMom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        upmom = HistUtil.plot(self.HUpMom,upMom)
        uptgtmom = HistUtil.plot(self.HUpTgtMom,upMom)
        upnomatmom = HistUtil.plot(self.HUpNoMatMom,upMom)
        upMom.legend(loc="upper right")
        dnmom = HistUtil.plot(self.HDnMom,dnMom)
        dntgtmom = HistUtil.plot(self.HDnTgtMom,dnMom)
        dnnomatmom = HistUtil.plot(self.HDnNoMatMom,dnMom)
        dnMom.legend(loc="upper right")

    def PlotDeltaMomentum(self):
        fig, (upDMom, dnDMom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        dndelmom = HistUtil.plot(self.HDnDeltaMom,dnDMom)
        dndeltgtmom = HistUtil.plot(self.HDnDeltaTgtMom,dnDMom)
        dndelnomatmom = HistUtil.plot(self.HDnDeltaNoMatMom,dnDMom)
        dnDMom.legend(loc="upper right")
        updelmom = HistUtil.plot(self.HUpDeltaMom,upDMom)
        updeltgtmom = HistUtil.plot(self.HUpDeltaTgtMom,upDMom)
        updelnomatmom = HistUtil.plot(self.HUpDeltaNoMatMom,upDMom)
        upDMom.legend(loc="upper right")

    def PlotMomRes(self):
        fig, (upMomRes, dnMomRes) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        dnmomres = HistUtil.plot(self.HDnMomRes,dnMomRes)
        CB = FitFunctions.CrystalBall()
        HistUtil.fit(self.HDnMomRes,CB,subplot=dnMomRes)
        dnMomRes.legend(loc="upper right")
        upmomres = HistUtil.plot(self.HUpMomRes,upMomRes)
        RCB = FitFunctions.CrystalBall(reverse=True)
        HistUtil.fit(self.HUpMomRes,RCB,subplot=upMomRes)
        upMomRes.legend(loc="upper right")

    def PlotUpstreamMat(self):
        fig, (all, tgt, nomat) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upxtrapdmom = HistUtil.plot(self.HUpMatUpExtrap,all)
        dnxtrapdmom = HistUtil.plot(self.HUpMatDnExtrap,all)
        fitdmom = HistUtil.plot(self.HUpMatFit,all)
        mcdmom = HistUtil.plot(self.HUpMatMC,all)
        all.legend(loc="upper left")
        upxtrapdmom = HistUtil.plot(self.HUpMatTgtUpExtrap,tgt)
        dnxtrapdmom = HistUtil.plot(self.HUpMatTgtDnExtrap,tgt)
        fitdmom = HistUtil.plot(self.HUpMatTgtFit,tgt)
        mcdmom = HistUtil.plot(self.HUpMatTgtMC,tgt)
        tgt.legend(loc="upper left")
        upxtrapdmom = HistUtil.plot(self.HUpMatNoMatUpExtrap,nomat)
        dnxtrapdmom = HistUtil.plot(self.HUpMatNoMatDnExtrap,nomat)
        fitdmom = HistUtil.plot(self.HUpMatNoMatFit,nomat)
        mcdmom = HistUtil.plot(self.HUpMatNoMatMC,nomat)
        nomat.legend(loc="upper left")

    def PlotTime(self):
        fig, (deltaTime) = plt.subplots(1,1,layout='constrained', figsize=(5,5))
        updtime = HistUtil.plot(self.HupDeltaTime,deltaTime)
        dndtime = HistUtil.plot(self.HdnDeltaTime,deltaTime)
        deltaTime.legend(loc="upper right")
