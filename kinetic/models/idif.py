


class ImageDerivedInputFunction:
  def __init__(self):
    pass

###
#
# #!/usr/bin/python

import os
import argparse
import numpy as np
import json
import csv
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation

from rhkinetics.lib import dcm
from rhkinetics.lib import plotting
from rhkinetics.lib import roi
from rhkinetics.lib import moose, totalsegmentator
from rhkinetics.lib import pet

__scriptname__ = 'idif'
__version__ = '0.3.1'
__author__ = 'Ulrich Lindberg'

def aorta_segment(aortamask: np.ndarray=bool):
    """ Segment aorta in four segments with value:
        1. Aorta Ascendens
        2. Aorta Arch
        3. Aorta Descendens (upper)
        4. Aorta Descendens (lower)
    """

    # Get image dimensions
    xdim,ydim,nslices = aortamask.shape

    # Allocate aortamask_segmented
    aortamask_segmented = aortamask.astype(int)

    # Loop over all axial slices and count number of clusters
    nclusters = np.zeros((nslices,),dtype=int)

    for slc in range(nslices):
        label_img, nclusters[slc] = label(aortamask[:,:,slc], return_num=True)

    # Compute volume within each slice
    volume = np.count_nonzero(aortamask,axis=(0,1))
    print(np.mean(volume[nclusters==1]),np.mean(volume[nclusters==2]))

    # Correct
    nclusters[nclusters>2] = 2

    # Find islands of connected ones in nclusters
    #islands = np.flatnonzero(np.diff(np.r_[0,nclusters,0])!=0).reshape(-1,2) - [0,1]
    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False],nclusters==1,[False]))))[0].reshape(-1,2)

    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    start_longest_seq_1 = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]

    # Aorta descendens is the biggest island of connected slices
    slices_desc = range(start_longest_seq_1,nslices)

    # Segment aorta descendens lower
    aortamask_segmented[:,:,slices_desc] = aortamask[:,:,slices_desc] * 4

    # Aorta arch is the biggest island of connected twos
    nclusters_twos = nclusters
    nclusters_twos[nclusters_twos==1] = 0

    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(np.diff(np.hstack(([False],nclusters==2,[False]))))[0].reshape(-1,2)
    # Get the island lengths, whose argmax would give us the ID of longest island.
    # Start index of that island would be the desired output
    start_longest_seq_2 = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]
    slices_arch = range(0,start_longest_seq_2)
    slices_two = range(start_longest_seq_2,start_longest_seq_1)

    # Find upper descending part of aorta by finding connection with lower part
    label_img_2, nclusters = label(aortamask[:,:,slices_two[0]:slices_desc[0]+1], return_num=True)
    for cluster in range(1,nclusters+1):
        clustersize = np.sum((label_img_2==cluster))
        if np.sum((label_img_2==cluster)*(aortamask_segmented[:,:,slices_two[0]:slices_desc[0]+1]==4)):
            aortamask_segmented[:,:,slices_two[0]:slices_desc[0]] =  aortamask_segmented[:,:,slices_two[0]:slices_desc[0]] + (label_img_2[:,:,0:-1]==cluster)*2

    #### TEMP: Set Aorta Arch to value 2
    #aortamask_segmented[:,:,0:slices_two[0]] = aortamask[:,:,0:slices_two[0]] * 2
    aortamask_segmented[:,:,slices_arch] = aortamask[:,:,slices_arch] * 2
    ##########
    return aortamask_segmented


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(prog='QUADRA_IDIF', description='Image derived input function of dynamic PET data')
    parser.add_argument('-v','--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    # Required arguments
    parser.add_argument('-i','--data', help='Input directory (DICOM)', required=True)
    parser.add_argument('-s','--segmentation', help='Organ segmentation directory (DICOM)', required=True)
    parser.add_argument('-o','--outdir', help='Output directory', required=True)

    # Parse arguments
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    ### SEGMENTAION ###
    # Recursively locate all DICOM files inside label directory
    filelist_label = []
    for dirpath, subdirs, files in os.walk(args.segmentation):
        filelist_label.extend(os.path.join(dirpath, x) for x in files if x.endswith(('.IMA', '.dcm')))

    # Read segmentation data into deck
    hdr_label, deck_label = dcm.dcmread_folder(filelist_label)
    deck_label = np.squeeze(deck_label) # data is loaded with a 4th dimension - THIS SHOULD BE FIXED IN quadra_reslice.py !!!

    xdim,ydim,nslices = deck_label.shape

    # Make sure that slice orientation is superior->inferior
    if hdr_label['dimlabel'][-2::] == 'is':
        print('Flipping slice orientation')
        deck_label = deck_label[:,:,::-1]
        #hdr_label['dimlabel'][-2::] = 'si' #DOES NOT WORK

    print(f"Label dimensions: {deck_label.shape}")

    # Get label value for aorta in segmentation
    method = hdr_label['SeriesDescription']
    if method == 'TotalSegmentator':
        regionidx = totalsegmentator.get_regionidx('aorta') # totalsegmentator
    elif method == 'TotalSegmentator-v2':
        regionidx = totalsegmentator.get_regionidx_v2('aorta') # totalsegmentator-v2
    elif method == 'MOOSE':
        regionidx = moose.get_regionidx('aorta') # MOOSE
    elif method == 'SegThor':
        regionidx = 4 # segthor
    else:
        raise SystemExit('No header label match')

    print(f'Method: {method}, Aorta label: {regionidx}')

    # Get only aorta from segmentation mask
    aortamask = (deck_label == regionidx)

    # Assert that aorta mask is only one cluster
    #aortamask = roi.keep_largest_cluster(aortamask)

    # Get slices containing region
    xmin, xsize, ymin, ysize, zmin, zsize = roi.bbox(aortamask)
    # Get slices containing region
    slicesum = np.sum(aortamask.astype(int), axis=(0, 1))
    slices_nonzero = [i for i, e in enumerate(slicesum) if e > 0]
    aortamask = aortamask[:,:,zmin:zmin+zsize]
    if hdr_label['dimlabel'][-2::] == 'si':
        slices = range(zmin,zmin+zsize)
    else:
        # Flip slices
        slices = range(-1*(zmin+zsize-1-nslices),-1*(zmin-nslices)+1)

    ### Read Dynamical Data (Only slices within aorta) ###
    # Create mlist dictionary of all files in PET directory
    # Check if file already exist
    print("Load PET data:")
    if os.path.isfile(args.data+'.tsv'):
        print('  Reading existing mlist file')
        mlist = dcm.dcmmlistread(args.data+'.tsv')
    else:
        study_dict = dcm.dcmmlist(filelist)

        # Get StudyInstanceUID(s)
        StudyInstanceUID = list(study_dict.keys())

        if len(StudyInstanceUID) > 1:
            raise SystemExit('  More than one DICOM study in PET directory... Aborting!')

        # Get SeriesInstanceUID(s)
        SeriesInstanceUID = list(study_dict[StudyInstanceUID[0]].keys())

        if len(SeriesInstanceUID) > 1:
            raise SystemExit('  More than one DICOM series in PET directory... Aborting!')

        mlist = study_dict[StudyInstanceUID[0]][SeriesInstanceUID[0]]['mlist']

    # Read only slices with aorta
    filtered = filter(lambda row: row[0]-1 in slices, mlist)

    # Read dynamical PET data
    hdr_pet, deck = dcm.dcmmlist_read(list(filtered))

    # Get Spatial information
    xdim,ydim,nslices,nframes = deck.shape
    voxdim = np.array(([hdr_pet['PixelSpacing'][0], hdr_pet['PixelSpacing'][1], hdr_pet['SliceThickness']]))

    if hdr_pet['dimlabel'][-2::] == 'is':
        deck = deck[:,:,::-1,:]

    # Get time info from header
    FrameTimesStart = np.array(hdr_pet['FrameTimesStart'])
    FrameDuration = np.array(hdr_pet['FrameDuration'])

    # Create SUV from 5 to 40 seconds
    print(f"Create SUV map. TotalDose: {hdr_pet['RadionuclideInjectedDose']/1e6} MBq, PatientWeight: {hdr_pet['PatientWeight']} kg")
    # Create SUV from first 40 second frames
    suvframes = FrameDuration==np.unique(FrameDuration)[0]
    #suvframes = FrameTimesStart < 40

    SUV = pet.suv(deck[:,:,:,suvframes],hdr_pet['RadionuclideInjectedDose'],hdr_pet['PatientWeight'])

    # Compute median SUV value inside aortamask
    SUV_median = np.median(SUV[aortamask])
    print(f'Median SUV inside Aorta Mask: {SUV_median}')


    # Threshold aortamask with median(SUV)/1.5
    aortamask = aortamask*np.int8(SUV>SUV_median/1.5)

    # Count number of clusters
    nclusters = roi.count_clusters(aortamask)
    print(f'Number of Clusters found in Aorta Segmentation: {nclusters}')

    if nclusters > 1:
        # Handle the mystery
        print(' Handling multiple clusters')

        # Keep only cluster above threshold
        volthreshold=20
        print(f'Reomving cluster(s) with volume lower than {volthreshold*np.prod(voxdim)/1000:.2f} ml')
        aortamask_tmp, nclusters = roi.threshold_clusters(aortamask, volthreshold=volthreshold)

        print(f'  Remaining clusters: {nclusters}')

        # Still have multiple clusters - now try to extrapolate
        if nclusters > 1:
            print('Extrapolation')

            # Loop over axial slices and count number of clusters
            nrois = np.zeros((nslices,),dtype=int)
            for slc in range(nslices-1,-1,-1):
                nrois[slc] = roi.count_clusters(aortamask[:,:,slc])

                if nrois[slc] == 0 or np.count_nonzero(aortamask[:,:,slc])<3:
                    # Get bounding box for the two slices below
                    xmin_tmp, xsize_tmp, ymin_tmp, ysize_tmp, _, _ = roi.bbox(aortamask[:,:,slc+1:slc+3])

                    xmid_tmp = xmin_tmp+xsize_tmp//2
                    ymid_tmp = ymin_tmp+ysize_tmp//2

                    # Keep only largest cluster if multiple
                    maskimg = roi.keep_largest_cluster(SUV[xmid_tmp-5:xmid_tmp+5,ymid_tmp-5:ymid_tmp+5,slc]>SUV_median/1.5)

                    # Run region props on binary mask add SUV image for weighted centroid estimation
                    regions = regionprops(label(maskimg), intensity_image=SUV[xmid_tmp-5:xmid_tmp+5,ymid_tmp-5:ymid_tmp+5,slc])
                    for props in regions:
                        aortamask[xmid_tmp-5+int(props.centroid_weighted[0])-3:xmid_tmp-5+int(props.centroid_weighted[0])+4,
                            ymid_tmp-5+int(props.centroid_weighted[1])-3:ymid_tmp-5+int(props.centroid_weighted[1])+4,slc] = 1

                # Dilate and threshold to account for aortavoxels outside segmentation
                aortamask_dilated = binary_dilation(aortamask[:,:,slc])
                aortamask[:,:,slc] = aortamask_dilated*np.int8(SUV[:,:,slc]>SUV_median/1.5)

    # Create figure of SUV overlayed with aorta VOI
    fig, ax = plt.subplots(1,2, constrained_layout=True)
    # Sag
    ax[0].imshow(np.transpose(np.max(SUV[::-1,:,:],axis=1), (1, 0)), vmin=0, vmax=2*SUV_median, cmap='gray_r',
        aspect=hdr_pet['SliceThickness']/hdr_pet['PixelSpacing'][1])
    plotting.plot_outlines(np.transpose(np.max(aortamask[::-1,:,:],axis=1), (1, 0)).T, ax=ax[0], lw=0.5, color='r')
    ax[0].axis('off')
    # Cor
    ax[1].imshow(np.transpose(np.max(SUV,axis=0), (1, 0)), vmin=0, vmax=SUV_median*2, cmap='gray_r',
        aspect=hdr_label['SliceThickness']/hdr_label['PixelSpacing'][0])
    plotting.plot_outlines(np.transpose(np.max(aortamask,axis=0), (1, 0)).T, ax=ax[1], lw=0.5, color='r')
    ax[1].axis('off')
    plt.suptitle('Aorta Segmentation')
    plt.savefig(os.path.join(args.outdir,'segmentation_corr.pdf'))
    print(os.path.join(args.outdir,'segmentation_corr.pdf'))

    ### Segment aorta in four segments ###
    segments = ['Aorta asc', 'Aortic arch', 'Aorta desc (upper)', 'Aorta desc (lower)']
    aortamask_segmented = aorta_segment(aortamask)

    # Create bounding box for figure
    xmin, xsize, ymin, ysize, zmin, zsize = roi.bbox(aortamask_segmented>0)

    # Create square box with original centers
    xmid = xmin+xsize//2
    ymid = ymin+ysize//2
    xsize = ysize = np.amax([xsize, ysize])
    xmin = xmid-xsize//2
    ymin = ymid-ysize//2

    ### Create VOI inside aorta arch of approx 1 mL ###
    thr = 1000//np.prod(voxdim)

    # Allocate
    VOI = np.zeros((xdim,ydim,nslices,4), dtype=bool)
    idif = np.zeros((nframes,4))

    print('Looping over each segment')
    N = int(np.round((1000/(np.prod(voxdim)*3*3))/2)*2)
    print(f"Length of VOI: {N} slices")




    for seg in range(4):
        if seg in [0,2,3]:
            # Create slice profile in z-direction
            # Get median value of each slice
            slicemedian = np.zeros(nslices)
            for slc in range(nslices):
                if np.sum(aortamask_segmented[:,:,slc]==seg+1):
                    SUV_segment = SUV[:,:,slc]*(aortamask_segmented[:,:,slc]==seg+1)
                    slicemedian[slc] = np.median(SUV_segment[SUV_segment>0])

            # Sliding window average of slice profile
            middleslc = np.argmax(np.convolve(slicemedian, np.ones(N)/N, mode='valid'))+N//2

            # Position 3x3xN VOI within segment with detected center slice
            for slc in range(middleslc-N//2,middleslc+N//2):
                if np.sum(aortamask_segmented[:,:,slc]==seg+1):
                    x0,y0 = roi.cog(SUV[:,:,slc]*(aortamask_segmented[:,:,slc]==seg+1))
                    VOI[y0-1:y0+2,x0-1:x0+2,slc,seg] = 1
        else:
        	# Aortic Arch
            # Data-driven approach to find VOI based on maximum thresholding
            prc = 99.99
            volume = 0

            while volume <= thr:
                VOI[:,:,:,seg] = (SUV*(aortamask_segmented==seg+1))>=np.percentile(SUV[aortamask_segmented==seg+1],prc)
                label_img, nclusters = label(VOI[...,seg], return_num=True)

                clustersize = np.zeros((nclusters,))

                for cluster in range(nclusters):
                    clustersize[cluster] = np.sum(label_img==cluster+1)

                # Find largest cluster
                maxclusteridx = np.argmax(clustersize)+1
                VOI[:,:,:,seg] = label_img==maxclusteridx

                volume = clustersize[maxclusteridx-1]
                prc -= 0.5

            print(f"volume: {volume*np.prod(voxdim):.2f} mm3")
            print(f"threshold: {prc}")

        ### Extract IDIF as the mean inside the VOI ###
        idif[:,seg] = np.mean(deck[np.squeeze(VOI[:,:,:,seg])], axis=0)

        ### Write NIfTI file of SUV data
        #niftifile = os.path.join(args.outdir,'VOI_segment-'+str(seg+1)+'.nii.gz')
        #if not os.path.isfile(niftifile):
        #    print('Store VOI as NIfTI')
        #    # Use transformation matrix from DICOM header
        #    OM = hdr_pet['Affine3D']

        #    # Create NIFTI object
        #    header = nib.Nifti1Header()
        #    header.set_data_shape(VOI[:,:,:,seg].shape)
        #    header.set_dim_info(slice=2)
        #   header.set_xyzt_units('mm')

        #    ### TODO: LIKELY MORE INFO SHOULD BE ADDED TO THE NIFTI FILE HERE ###
        #    nifti =  nib.Nifti1Image(VOI[:,:,:,seg], OM, header=header)
        #    nib.save(nifti, niftifile)

        # Save VOI as numpy array
        #np.save(voifile,VOI)

        # Write IDIF to file
        #pet.tacwrite(FrameTimesStart,FrameDuration,idif[:,seg],'Bq/cc',os.path.join(args.outdir,'QUADRA_segment-'+str(seg+1)+'_IDIF.tac'),['idif'])
        pet.tacwrite(FrameTimesStart,FrameDuration,idif[:,seg],'Bq/cc',os.path.join(args.outdir,'IDIF_'+method.lower()+'_segment-'+str(seg+1)+'.tac'),['idif'])