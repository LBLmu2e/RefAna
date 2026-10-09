#
# class to verify the integrity of the reflections tuple
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

class RefCheck(object):
    def __init__(self,sid):
        self.SID = sid

    def Print(self):
        print("Reflection test, SID=",self.SID)

    def Loop(self,files,treename):
        # append tree to files for uproot
        Files = [None]*len(files)
        for i in range(0,len(files)):
            Files[i] = files[i]+":"+treename
        ibatch = 0
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(Files,filter_name="/evtinfo|trk.trk|trksegs|trkmcsim|trksegsmc/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            segs = batch['trksegs'] # track fit samples
            upSegs = segs[:,0] # upstream track fits
            dnSegs = segs[:,1] # downstream track fits
            upSegsMC = trksegsmc[:,0]
            dnSegsMC = trksegsmc[:,1]

            upTrkMC = trkMC[:,0,0] # primary MC match of upstream track
            dnTrkMC = trkMC[:,1,0] # primary MC match of downstream track
            print(upTrkMC)
            print(dnTrkMC)


            trksegsmc = batch['trksegsmc']
            run = batch['run']
            subrun = batch['subrun']
            event = batch['event']

            # basic consistency test
            assert(len(upSegs) == len(dnSegs))
            # times at specified surface for different fits, directions
            upupEntTime = upSegs[(upSegs.sid==self.SID) & (upSegs.mom.z() < 0.0) ].time
            updnEntTime = upSegs[(upSegs.sid==self.SID) & (upSegs.mom.z() > 0.0) ].time
            dndnEntTime = dnSegs[(dnSegs.sid==self.SID) & (dnSegs.mom.z() > 0.0) ].time
            dnupEntTime = dnSegs[(dnSegs.sid==self.SID) & (dnSegs.mom.z() < 0.0) ].time

            updnsegmc = (upSegsMC.sid == self.SID) & (upSegsMC.mom.Z() > 0.0)
            dndnsegmc = (dnSegsMC.sid == self.SID) & (dnSegsMC.mom.Z() > 0.0)
            upupsegmc = (upSegsMC.sid == self.SID) & (upSegsMC.mom.Z() < 0.0)
            dnupsegmc = (dnSegsMC.sid == self.SID) & (dnSegsMC.mom.Z() < 0.0)

            # check for missing intersections
            upupgood = ak.firsts(upupEntTime)
            updngood = ak.firsts(updnEntTime)
            dnupgood = ak.firsts(dnupEntTime)
            dndngood = ak.firsts(dndnEntTime)
            #
            missingupup = ak.is_none(upupgood)
            missingupdn = ak.is_none(updngood)
            missingdndn = ak.is_none(dndngood)
            missingdnup = ak.is_none(dnupgood)

            if np.count_nonzero(missingupup)> 0:
                print(f"Missing TT_front inter: upup {np.count_nonzero(missingupup)}")
                for i in range(0,len(missingupup)):
                    if ( missingupup[i] ):
                        print(f"\"{run[i]}:{subrun[i]}:{event[i]}\",")
            if np.count_nonzero(missingupdn)> 0:
                print(f"Missing TT_front inter: updn {np.count_nonzero(missingupdn)}")
                for i in range(0,len(missingupdn)):
                    if ( missingupdn[i] ) :
                        print(f"\"{run[i]}:{subrun[i]}:{event[i]}\",")
            if np.count_nonzero(missingdnup)> 0:
                print(f"Missing TT_front inter: dnup {np.count_nonzero(missingdnup)}")
                for i in range(0,len(missingdnup)):
                    if ( missingdnup[i] ) :
                        print(f"\"{run[i]}:{subrun[i]}:{event[i]}\",")
            if np.count_nonzero(missingdndn)> 0:
                print(f"Missing TT_front inter: dndn {np.count_nonzero(missingdndn)}")
                for i in range(0,len(missingdndn)):
                    if ( missingdndn[i] ):
                        print(f"\"{run[i]}:{subrun[i]}:{event[i]}\",")


        # Histogram basic reflection selection properties (todo)
        # Time differences
        nDeltaTimeBins = 100
        deltaTimeRange = [-20,20]
        self.HdnDeltaTime = HistUtil.new_hist(name="DeltaTime",label="Upstream",bins=nDeltaTimeBins, range=deltaTimeRange, xlabel="Downstream - Upstream time (ns)",title=self.PDGName+" $\\Delta$ Time at "+self.CompName)
        self.HupDeltaTime = HistUtil.new_hist(name="DeltaTime",label="Downstream",bins=nDeltaTimeBins, range=deltaTimeRange, xlabel="Downstream - Upstream time (ns)",title=self.PDGName+" $\\Delta$ Time at "+self.CompName)



        print("\nDone processing")
