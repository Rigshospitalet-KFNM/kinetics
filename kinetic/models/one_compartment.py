# Python Standard library
import re

# Third party packages
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
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
endframe = max(np.argwhere(MidFrameTime <= endtime))[0]+1
startframe = min(np.argwhere(MidFrameTime >= starttime))[0]

tac, header, units = tacread(IDIFPATH)
if 'start' in header[0]:
	FrameTimesStart = np.array(tac[:, 0])
	FrameTimesEnd = np.array(tac[:, 1])
	FrameDuration = FrameTimesEnd - FrameTimesStart
idif = tac[:, -1]

# determine delay based on a 1cmp model
maxidx = np.argmax(idif)
TTP = MidFrameTime[maxidx]
print(f'TTP in IDIF: {TTP} s')
popt, curve_fit = att.fit_monoexp_att(MidFrameTime, idif, regioncurve, p0 = None, endtime = MidFrameTime[maxidx+5], return_fit=True)
print(f'Perfusion from 1-comp modelling: {popt[0]*6000:.1f} ml/100 ml/min')
print(f'Delay estimated from 1cmp modelling: {popt[2]:.2f} s')
deltaT = popt[-1]
# upsample data to perform delay correction
idif_orig = idif
idif = pchip_interpolate(MidFrameTime,idif,MidFrameTime-popt[2])
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(MidFrameTime,regioncurve,'bo', label='Average {} TAC'.format(REGION))
ax.plot(MidFrameTime[0:curve_fit.size],curve_fit,'r', label=f'Fit')
ax.plot(MidFrameTime,idif/(np.max(idif)/np.max(regioncurve)),'g', label='idif (scaled)')
ax.plot(MidFrameTime,idif_orig/(np.max(idif)/np.max(regioncurve)),'k', label='idif (orig,scaled)')
ax.text(20,0,f'$K_{{1}}$ = {popt[0]*6000:.1f} ml/100ml/min, $V_{{d}}$ = {popt[1]:.2f} $\\Delta T$ = {popt[2]:.2f} s')
ax.set_xlim([0, 60])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Concentration [kBq/ml]')
ax.legend()
plt.savefig(os.path.join(RESULTDIR,'1cmp_'+REGION+'-delay.pdf'))
plt.close()
print(os.path.join(RESULTDIR,'1cmp_'+REGION+'-delay.pdf'))

K1, k2, Va = modelling.onecmp(deck[..., :endframe], idif[:endframe], MidFrameTime[:endframe] / 60.0, thr=0.01)
#K1, k2, Va = modelling.onecmp_parallel(deck[..., :endframe], idif[:endframe], MidFrameTime[:endframe] / 60.0, thr=0.01)

# Report mean within regionmask
K1_mean = np.mean(K1[regionmask], axis=0)
k2_mean = np.mean(k2[regionmask], axis=0)
Va_mean = np.mean(Va[regionmask], axis=0)

try:
	# save .txt files with results
	with open(os.path.join(RESULTDIR,'results.txt'),'w+',encoding='UTF8') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['K1','k2','Va','delay'])
		writer.writerow([K1_mean,k2_mean,Va_mean,deltaT])
		print('Wrote', os.path.join(RESULTDIR,'results.txt'))
except Exception as e:
	print(e)

# # Read first dynamic of PET data - only slices from the regionmask
filtered = filter(lambda row: row[0] - 1 in slices and row[1] == 1, mlist)

# Create bounding box for figure
xmin, xsize, ymin, ysize, zmin, zsize = roi.bbox(K1 > 0)

# Create square box with original centers
xmid = xmin + xsize // 2
ymid = ymin + ysize // 2
xsize = ysize = np.amax([xsize, ysize])
xmin = xmid - xsize // 2
ymin = ymid - ysize // 2

# Convert 3D to 2D montage cropping to boundig box and save figure
im = pet.montage(K1[xmin:xmin + xsize, ymin:ymin + ysize, zmin:zmin + zsize])
plotting.imshow(im, vmin=0, vmax=np.round(2 * K1_mean), cmap='viridis', outfile=os.path.join(RESULTDIR, REGION + '-' + MODEL + '-K1.pdf'))

# Convert 3D to 2D montage cropping to boundig box and save figure
im = pet.montage(k2[xmin:xmin + xsize, ymin:ymin + ysize, zmin:zmin + zsize])
plotting.imshow(im, vmin=0, vmax=np.round(2 * k2_mean), cmap='viridis', outfile=os.path.join(RESULTDIR, REGION + '-' + MODEL + '-k2.pdf'))

# Convert 3D to 2D montage cropping to boundig box and save figure
im = pet.montage(Va[xmin:xmin + xsize, ymin:ymin + ysize, zmin:zmin + zsize])
plotting.imshow(im, vmin=0, vmax=0.5, cmap='viridis', outfile=os.path.join(RESULTDIR, REGION + '-' + MODEL + '-Va.pdf'))

# print dirs to convenience
print('Saved .pdfs to', os.path.join(RESULTDIR))

# Create Histograms with values within regionmask
counts, bins = np.histogram(K1[regionmask])
fig, ax = plt.subplots(2, 2, constrained_layout=True)
counts, bins = np.histogram(K1[regionmask], bins=100)
ax[0, 0].hist(bins[:-1], bins, weights=counts)
ax[0, 0].set_xlabel('ml blood/ml tissue/min')
ax[0, 0].set_ylabel('Counts')
ax[0, 0].set_title(f'K1: {K1_mean:.2f}')
counts, bins = np.histogram(k2[regionmask], bins=100)
ax[0, 1].hist(bins[:-1], bins, weights=counts)
ax[0, 1].set_xlabel('1/min')
ax[0, 1].set_title(f'k2: {k2_mean:.2f}')
counts, bins = np.histogram(Va[regionmask], bins=100)
ax[1, 0].hist(bins[:-1], bins, weights=counts)
ax[1, 0].set_xlabel('ml blood/ml tissue')
ax[1, 0].set_ylabel('Counts')
ax[1, 0].set_title(f'Va: {Va_mean:.2f}')

plt.savefig(os.path.join(RESULTDIR, REGION + '-' + MODEL + '-hist.pdf'), format='pdf')
plt.close()
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

if DEBUG: # copy the parameter maps to debug folder
	try:
		shutil.copy(os.path.join(RESULTDIR, '1cmp_' + REGION + '_K1.nii.gz'),os.path.join(RESULTDIR,'DEBUG'))
		shutil.copy(os.path.join(RESULTDIR, '1cmp_' + REGION + '_k2.nii.gz'),os.path.join(RESULTDIR,'DEBUG'))
		shutil.copy(os.path.join(RESULTDIR, '1cmp_' + REGION + '_Va.nii.gz'),os.path.join(RESULTDIR,'DEBUG'))
	except Exception as e:
		print(e)
	else:
		print('Copied parameter maps to',os.path.join(RESULTDIR,'DEBUG'))