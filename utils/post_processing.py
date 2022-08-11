import os
import glob
import regex
import numpy as np
import pandas as pd
import nibabel as nib


def find_targeted_files(path):
    '''
    Find and save files of interest. In this case, we look for short-axis
    nifti files.
    '''
    patt = ['cine_short_axis', 'cine_short_axis_6MM', 'CINE_EC*_apex', 'EC_*_FIL', 'EC_*_10slices',
            'CINE_EC_barrido', 'CINE_EC', 'cine_*_EC', 'SHORT_AXIS', '*CINE_EJE_CORTO', 
            'FUNCION_VI', '*_#SA', '*SAX*', 'sa_cine', 'sa_*']
    # Remove suffices containing slice number that will affect the next steps
    # in filenames as in sa_cine(1).nii.gz or sa_cine_slc1.nii.gz, for example
    slc_patt = r'(SLC\d+)|(slc\d+)|(\(\d+\))'
    filepaths = []
    for p in patt:
        filepaths.extend(list(glob.iglob(os.path.join(path, '**', p + '.nii.gz'), recursive=True)))

    files = {}
    for f in sorted(filepaths):
        # Skip contour files
        if '_label' in f: continue
        basename = os.path.basename(f)
        basename = regex.sub(slc_patt, '', basename)
        if basename not in files.keys():
            files[basename] = [f]
        else:
            files[basename].append(f)

    return files


def postProcess(path):
    '''
    Look for short-axis nifti slices and grouped them together in one
    4D file.
    '''
    files = find_targeted_files(path)
    print('--->', files)

    for k in files.keys():
        if len(files[k]) < 2: continue
        post_process_files(files[k], basename=k)


def post_process_files(files, basename):
    'Take list of files and build a 4D image if possible.'
    if len(files) > 1:
        ims = [nib.load(f) for f in files]
        zs = pd.Series([im.header.structarr['qoffset_z'] for im in ims])
        mask = np.ones(len(ims)) == 1
        mks = None
        try:
            mks = [nib.load(regex.sub(r'\.nii', '_label.nii', f)) for f in files]
            mkus = [nib.load(regex.sub(r'\.nii', '_label_upsample.nii', f)) for f in files]
            # Filter new and old slices by mask existence
            mask = [np.unique(mk.get_fdata()).size > 1 for mk in mks]
            zs = zs[mask]
            filt_mks = np.asarray(mks)[zs.index]
            filt_mkus = np.asarray(mkus)[zs.index]
        except Exception:
            print('Label not found for files during post-processing.')

        #zs = zs.drop_duplicates(keep='first')
        filt_ims = np.asarray(ims)[zs.index]
        if np.any(np.greater([im.shape[2] for im in ims], 1)):
            # There is already a 4D image and repeated slices
            # probably due to the quality of the acquisition.
            # We try to concatenate them.
            print('4D image found already!')
            # If only one nifti has mask, skip postprocess
            if np.sum(mask) == 1: return 0
            # If other nitfis are 4D, skip postprocess as well
            if ims[1].get_fdata().ndim > 3: return 0
            im1 = ims[0].get_fdata()[:]
            # List of z positions per slice
            im_dict = ims[0].header.structarr
            zpos = np.asarray([im_dict['qoffset_z'] + im_dict['srow_z'][2]*i for i in range(ims[0].shape[-2])])
            im_array = np.asarray([
                *[np.expand_dims(im1[...,i,:], axis=2) for i in range(im1.shape[2])][::-1],
                *[ims[i].get_fdata()[:] for i in range(1,len(ims))]
            ])
            newim = np.concatenate(im_array, axis=2)

            if mks is not None:
                mk1 = mks[0].get_fdata()[:]
                mku1 = mkus[0].get_fdata()[:]
                mk_array = np.asarray([
                    *[np.expand_dims(mk1[...,i,:], axis=2) for i in range(mk1.shape[2])][::-1],
                    *[mks[i].get_fdata()[:] for i in range(1,len(mks))]
                ])
                newmk = np.concatenate(mk_array, axis=2)
                mku_array = np.asarray([
                    *[np.expand_dims(mku1[...,i,:], axis=2) for i in range(mku1.shape[2])][::-1],
                    *[mkus[i].get_fdata()[:] for i in range(1,len(mkus))]
                ], dtype=np.uint8)
                newmku = np.concatenate(mku_array, axis=2)
        else:
            # All nifti files are 3D and we need to combine them together
            print('Set of 3D images found')
            newim = np.repeat(np.zeros(ims[0].get_fdata().shape), filt_ims.shape[0], axis=2)
            oims = filt_ims[np.argsort(zs)]
            if mks is not None:
                newmk = np.repeat(np.zeros(mks[0].get_fdata().shape), filt_mks.shape[0], axis=2)
                newmku = np.repeat(np.zeros(mkus[0].get_fdata().shape, np.float16), filt_mkus.shape[0], axis=2)
                omks = filt_mks[np.argsort(zs)]
                omkus = filt_mkus[np.argsort(zs)]
            for i in range(len(oims)):
                try:
                    newim[...,i,:] = oims[i].get_fdata().squeeze()[:]
                    if mks is not None:
                        newmk[...,i,:] = omks[i].get_fdata().squeeze()[:]
                        newmku[...,i,:] = omkus[i].get_fdata().squeeze()[:]
                except ValueError: # Shape missmatch
                    print('ValueError: shape missmatch during post processing'
                          ' for files related to {}'.format(files[0]))
                    continue

        # Post-processed cine short axis
        outf = os.path.join(os.path.dirname(files[0]), 'pp_'+ basename)
        nim = nib.Nifti1Image(newim, None, ims[0].header)
        nib.save(nim, outf)
        if mks is not None:
            nmk = nib.Nifti1Image(newmk, None, mks[0].header)
            nib.save(nmk, regex.sub(r'\.nii', '_label.nii', outf))
            nmku = nib.Nifti1Image(newmku, None, mkus[0].header)
            nib.save(nmku, regex.sub(r'\.nii', '_label_upsample.nii', outf))
        
    return 1
