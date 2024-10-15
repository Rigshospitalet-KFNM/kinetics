# Python Standard library
import re
from typing import List

# Third party packages
from pydicom import Dataset
import numpy as np
import nibabel as nib

from scipy.linalg import toeplitz
from scipy.interpolate import pchip_interpolate
from scipy.optimize import minimize

# Kinetics packages
from kinetic.models import modelling


# Constants # These are from args of the process script
FIT_START = 0
FIT_END = 0
PADDING = 0

# STEP 1 -
def monoexp(time,idif,params):
    """ 1-comp model with delay
    """

    f    = params[0] # flow/K1
    lmb = params[1] # Distribution Volume
    deltaT = params[2] # ATT

    Hct = 0

    dt = time[1]-time[0]

    # Shift idif
    if deltaT:
        # idif_shift = np.interp(time,time+deltaT,idif)
        idif_shift = pchip_interpolate(time+deltaT,idif,time)
    else:
        idif_shift = idif

    # Residual Impulse Function
    RF = np.exp((-(1-Hct)*f*time)/lmb)

    # Create A matrix
    row = np.zeros_like(idif_shift)
    row[0] = idif[0]
    A = toeplitz(idif_shift,row)

    return f*dt*np.matmul(A,RF)



def func_to_minimise(params, x, y, idif):
    y_pred = monoexp(x,idif,params)
    # return np.sum((y_pred - y) ** 2)

    temp = y_pred - y

    sumsquare = np.dot(temp.T,temp)/(y.size-1)
    K=1

    # If deltaT is zero then ss is unaffected; if deltaT is long ss decrease. Thus long deltaT is favourized
    # sumsquare = sumsquare-0.5*(1-np.exp(-K*np.abs(params[2])))*sumsquare
    sumsquare = sumsquare*(1-0.5*(1-np.exp(-K*params[2])))

def func_to_minimise_att(params, x, y, idif, weights):
    # y_pred = monoexp(x,idif,params)
    # # return np.sum((y_pred - y) ** 2)

    # temp = y_pred[weights==1] - y

    # sumsquare = np.dot(temp.T,temp)/(y.size-1)
    # K=1

    # # If deltaT is zero then ss is unaffected; if deltaT is long ss decrease. Thus long deltaT is favourized
    # # sumsquare = sumsquare-0.5*(1-np.exp(-K*np.abs(params[2])))*sumsquare
    # sumsquare = sumsquare*(1-0.5*(1-np.exp(-K*params[2])))

    # return sumsquare

    y_pred = monoexp(x,idif,params)

    return np.linalg.norm(y_pred[weights==1] - y)


def fit_monoexp_att(time: np.array, idif: np.array, tac: np.ndarray, p0:np.ndarray = None, endtime:float = None, return_fit: bool=False, bounds=((0,300/6000),(0, 1),(0,10))):
    """Calculate delay based on 1 compartment fit
    """

    if p0 is None:
        # Start guess
        p0 = [30/6000, 0.5, 1.5]

    # Get smallest frame duration
    dt = np.amin(np.diff(time))

    # Handle non-uniform sampled data
    if np.unique(np.diff(time)).size > 1:

        # Interpolate time to smallest dt
        time_interp = np.arange(time[0],endtime,dt)

        # Interpolate IDIF
        idif = np.interp(time_interp,time,idif)

        # Create a weighting vector that only includes measured timepoints
        weights = np.zeros_like(time_interp)

        for i in range(time.size):
            # for each original MidFrameTime find nearest interpolated point and set to one
            timepoint = np.abs(time_interp-time[i]).argmin()

            # Set weight to 1 in the point
            weights[timepoint] = 1

        time = time_interp
    else:
        weights = np.zeros_like(time)+1

    # Call to minimization using bounds
    res = minimize(func_to_minimise_att, p0, args=(time, tac[0:np.sum(weights,dtype=int)], idif, weights), method='Nelder-Mead', bounds=bounds)

    if return_fit:
        y_pred = monoexp(time,idif,res.x)
        return res.x, y_pred[weights==1]
    return res.x



def one_compartment_model(pet: List[Dataset]):
  hdr_label, deck_label = dcm.dcmread_folder(filelist_label)
  deck_label = np.squeeze(deck_label)

  hdr, deck = dcm.dcmmlist_read(list(filtered))
  regionidx = totalsegmentator.get_regionidx_v2(REGION)

  regionmask = (deck_label == regionidx)
  regionmask = roi.keep_largest_cluster(regionmask)

  slices = list(range(slices[0] - PADDING, slices[0], 1)) + slices + list(range(slices[-1] + 1, slices[-1] + PADDING + 1, 1))
  regionmask = regionmask[:, :, slices]

  ### Extract mean timeseries inside regionmask ###
  regioncurve = np.mean(deck[regionmask], axis=0)
  # Get time info from header
  FrameTimesStart = np.array(hdr['FrameTimesStart'])
  FrameDuration = np.array(hdr['FrameDuration'])
  FrameTimesEnd = FrameTimesStart + FrameDuration
  MidFrameTime = FrameTimesStart + FrameDuration / 2.0


  def tacread(filename: str):
    """ Read Time Activity Curve (tac) file

    Parameters
    ----------
    filename : str
        filename of tacfile

    """

    # Read header row
    f = open(filename)
    header = f.readline().split('\t')

    # Read data
    tac = np.loadtxt(filename, delimiter='\t', skiprows=1)

    units = {}

    # Get timeunit from first column header
    units[0] = re.search(r"(?<=\[)[^)]*(?=\])",header[0]).group(0)

    # Get concentration unit from second column header
    units[1] = re.search(r"(?<=\[)[^)]*(?=\])",header[1]).group(0)

    return tac, header, units



  # read in tac to get an idea of datarange
  tac, header, units = tacread(IDIFPATH)

  # check if user gave input for fitting
  if FIT_START != 0:
    start_time = 0
    print('Selected starttime different from 0 s. Proceeding with', str(start_time), 's.')
  else:
    start_time = 0
    if FIT_END != 0:
      if FIT_END * 60 > max(MidFrameTime):
        end_time = max(MidFrameTime)
        print('Selected endtime outside of datarange. Proceeding with', str(end_time))
      else:
        end_time = FIT_END * 60  # s
    else:
      end_time = 180  # s

  # end frame corresponding to the selected endtime
  end_frame = max(np.argwhere(MidFrameTime <= end_time))[0]+1


  if 'start' in header[0]:
    FrameTimesStart = np.array(tac[:, 0])
    FrameTimesEnd = np.array(tac[:, 1])
    FrameDuration = FrameTimesEnd - FrameTimesStart
    idif = tac[:, -1]

  # determine delay based on a 1cmp model
  max_idx = np.argmax(idif)

  popt, curve_fit = fit_monoexp_att(MidFrameTime, idif, regioncurve, p0 = None, endtime = MidFrameTime[max_idx+5], return_fit=True)


  idif = pchip_interpolate(MidFrameTime,idif,MidFrameTime-popt[2])


  K1, k2, Va = modelling.onecmp(deck[..., :end_frame], idif[:end_frame], MidFrameTime[:end_frame] / 60.0, thr=0.01)


  return K1, k2, Va
