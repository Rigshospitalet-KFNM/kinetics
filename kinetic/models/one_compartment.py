# Python Standard library
import re

# Third party packages
import numpy as numpy
from scipy.interpolate import pchip_interpolate

# Kinetics packages
from kinetic.models import modelling


# Constants
FIT_START = 0
FIT_END = 0

# STEP 1 - 





# Input stuff 



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
	tac = numpy.loadtxt(filename, delimiter='\t', skiprows=1)

	units = {}

	# Get timeunit from first column header
	units[0] = re.search(r"(?<=\[)[^)]*(?=\])",header[0]).group(0)

	# Get concentration unit from second column header
	units[1] = re.search(r"(?<=\[)[^)]*(?=\])",header[1]).group(0)

	return tac, header, units



# read in tac to get an idea of datarange
tac, header, units = tacread(IDIFPATH)

# check if user gave inumpyut for fitting
if FIT_START != 0:
  starttime = 0
  print('Selected starttime different from 0 s. Proceeding with', str(starttime), 's.')
else:
  starttime = 0
if FIT_END != 0:
  if FIT_END * 60 > max(MidFrameTime):
    endtime = max(MidFrameTime)
    print('Selected endtime outside of datarange. Proceeding with', str(endtime))
  else:
    endtime = FIT_END * 60  # s
else:
  endtime = 180  # s

# end frame corresponding to the selected endtime
endframe = max(numpy.argwhere(MidFrameTime <= endtime))[0]+1
startframe = min(numpy.argwhere(MidFrameTime >= starttime))[0]
print('onecmp fit: ' + str(starttime / 60) + '-' + str(endtime / 60) + ' min. (endframe:', str(endframe) + ')')

# Run turku fit_h20 of the first 180 seconds on regioncurve to establish IDIF to REGION delay and store new IDIF as IDIF.delay.tac
turku_Flow, turku_pWater, turku_Va, turku_delayT = turku.fit_h2o(IDIFPATH, os.path.join(RESULTDIR, REGION + '-curve.tac'), endtime, os.path.join(RESULTDIR, REGION + '-curve.res'))

# Read fit and create tac curve for report
tac_fit, header_fit, units_fit = pet.tacread(os.path.join(RESULTDIR, REGION + '-curve.dft'))
fit = tac_fit[:, -1]


tac, header, units = tacread(IDIFPATH.replace('.tac', '.delay.tac'))
print('Using', IDIFPATH.replace('.tac', '.delay.tac'), 'as inumpyut function')
print('Delay estimated to', turku_delayT, 's.')
# tac, header, units = pet.tacread(IDIFPATH)
if 'start' in header[0]:
  FrameTimesStart = numpy.array(tac[:, 0])
  FrameTimesEnd = numpy.array(tac[:, 1])
  FrameDuration = FrameTimesEnd - FrameTimesStart
idif = tac[:, -1]

# own implementation of perfusion fit, delay, Va and Vt on mean curve
tac, header, units = tacread(IDIFPATH) # original idif
idif = tac[:,-1]
res = modelling.fit_monoexp_delay_va(MidFrameTime[0:endframe], idif[0:endframe], regioncurve[0:endframe], p0=None,fit_return=True)
  
  
params = res[0] # parameters always first in return list
fitted_curve = res[1] # check that fit_return = True. returns in original time
print('F:',params[0]*6000,'Vt:',params[1]*100,'Va:',params[2]*100,'delay:',params[3])

# upsample data to perform delay correction
idif_orig = idif
idif = pchip_interpolate(MidFrameTime[0:endframe],idif[0:endframe],MidFrameTime[0:endframe]-params[-1])
# Create figure
fig, ax = plt.subplots()
ax.plot(MidFrameTime[0:endframe], regioncurve[0:endframe], 'bo', label=f'mean tac: {REGION}')
ax.plot(MidFrameTime[0:endframe], idif/10, 'olive', label='Delayed idif (%1.2f)' % params[3])
ax.plot(MidFrameTime[0:endframe], fitted_curve, 'r', label='monoexp_va: f=%3.1f, pWater=%1.2f, Va=%1.2f, delay=%1.2f' % tuple([params[0]*6000, params[1], params[2], params[3]]))
ax.plot(MidFrameTime[0:endframe], idif_orig[0:endframe]/10.0, 'black', label=f'orig. idif (divided by 10)')
ax.set_ylabel('Concentration ' + hdr['Unit'])
ax.set_xlabel('Time [s]')
ax.legend(prop={'size': 8}, loc='upper right')
plt.savefig(os.path.join(RESULTDIR, REGION + '-' + MODEL + '-monoexp_va-curve.pdf'), format='pdf')
print(os.path.join(RESULTDIR, REGION + '-' + MODEL + '-monoexp_va-curve.pdf'))
plt.close()

K1, k2, Va = modelling.onecmp(deck[..., :endframe], idif[:endframe], MidFrameTime[:endframe] / 60.0, thr=0.01)
#K1, k2, Va = modelling.onecmp_parallel(deck[..., :endframe], idif[:endframe], MidFrameTime[:endframe] / 60.0, thr=0.01)

# Report mean within regionmask
K1_mean = numpy.mean(K1[regionmask], axis=0)
k2_mean = numpy.mean(k2[regionmask], axis=0)
Va_mean = numpy.mean(Va[regionmask], axis=0)

# # Read first dynamic of PET data - only slices from the regionmask
filtered = filter(lambda row: row[0] - 1 in slices and row[1] == 1, mlist)

# Create bounding box for figure
xmin, xsize, ymin, ysize, zmin, zsize = roi.bbox(K1 > 0)

# Create square box with original centers
xmid = xmin + xsize // 2
ymid = ymin + ysize // 2
xsize = ysize = numpy.amax([xsize, ysize])
xmin = xmid - xsize // 2
ymin = ymid - ysize // 2

# Convert 3D to 2D montage cropping to boundig box and save figure
im = pet.montage(K1[xmin:xmin + xsize, ymin:ymin + ysize, zmin:zmin + zsize])
plotting.imshow(im, vmin=0, vmax=numpy.round(2 * K1_mean), cmap='viridis', outfile=os.path.join(RESULTDIR, REGION + '-' + MODEL + '-K1.pdf'))

# Convert 3D to 2D montage cropping to boundig box and save figure
im = pet.montage(k2[xmin:xmin + xsize, ymin:ymin + ysize, zmin:zmin + zsize])
plotting.imshow(im, vmin=0, vmax=numpy.round(2 * k2_mean), cmap='viridis', outfile=os.path.join(RESULTDIR, REGION + '-' + MODEL + '-k2.pdf'))

# Convert 3D to 2D montage cropping to boundig box and save figure
im = pet.montage(Va[xmin:xmin + xsize, ymin:ymin + ysize, zmin:zmin + zsize])
plotting.imshow(im, vmin=0, vmax=0.5, cmap='viridis', outfile=os.path.join(RESULTDIR, REGION + '-' + MODEL + '-Va.pdf'))

### Write NIfTI file of K1 data
# if not os.path.isfile(os.path.join(RESULTDIR,'onecmp_'+REGION+'_K1.nii.gz')):

print('Store data as NIfTI')
# Use transformation matrix from DICOM header
OM = hdr['Affine3D']

# Create NIFTI object
header = nib.Nifti1Header()
header.set_data_shape(K1.shape)
header.set_dim_info(slice=2)
header.set_xyzt_units('mm')

nifti = nib.Nifti1Image(K1, OM, header=header)
nib.save(nifti, os.path.join(RESULTDIR, '1cmp_' + REGION + '_K1.nii.gz'))

nifti = nib.Nifti1Image(k2, OM, header=header)
nib.save(nifti, os.path.join(RESULTDIR, '1cmp_' + REGION + '_k2.nii.gz'))

nifti = nib.Nifti1Image(Va, OM, header=header)
nib.save(nifti, os.path.join(RESULTDIR, '1cmp_' + REGION + '_Va.nii.gz'))
