import numpy
from scipy import integrate
from scipy.optimize import nnls

def onecmp(deck: numpy.ndarray, aif: numpy.ndarray, midframetime: numpy.ndarray, thr: float = 0.05, regularization: bool=False):
  """ imgflow is adapted from turku tpclib
  """

  imadim = deck.shape
  nvox = int(numpy.prod(imadim[:-1]))

  # Find threshold to remove background and speed-up calculations
  # Pixels with AUC less than (threshold/100 x max AUC) are set to zero (Default is 5%)
  auc = numpy.trapz(deck, x=midframetime, axis=-1)
  thr_limit = numpy.amax(auc) * thr

  deck[auc < thr_limit] = 0

  # Arterial inumpyut curve
  cp = aif

  # Integrate inumpyut curve in minutes
  cpi = integrate.cumulative_trapezoid(cp, midframetime, initial=0)

  # Allocate solution array
  out = numpy.zeros((imadim[:-1] + (3,)))

  specified = numpy.nonzero(deck.any(axis=-1))
  for x, y, z in specified:
    # Integrate tissue
    cti = integrate.cumulative_trapezoid(deck[x, y, z, :], midframetime, initial=0)

    # if AUC at the end is <= zero, then forget this pixel
    if cti[-1] <= 0:
      continue

    # Create A matrix and multiply ctissue by -1
    A = numpy.stack((cti * -1, cpi, cp), axis=1)

    # Non-negative least squares from scipy.optimize
    out[x, y, z, :], rnorm = nnls(A, deck[x,y,z,:])

  # Store data
  k2 = out[..., 0]
  K1 = out[..., 1]
  Va = out[..., 2]

  return K1, k2, Va