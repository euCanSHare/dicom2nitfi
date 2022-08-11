import os
import cv2
import glob
import pickle
import argparse

import numpy as np

try:
    from utils.parse_cvi42 import parse as parse_cvi
except Exception:
    from parse_cvi42 import parse as parse_cvi


def parseContours(patient_dir, new_dir):
    """
    Find and parse contours from cvi42 files.
    Returns true if files were found and false otherwise.
    """
    # Obtain cvi42wsx or cvi42ws file
    files = list(glob.iglob(os.path.join(patient_dir, '*.cvi42ws*')))
    if len(files) != 0:
        cvi42_file = files[0]
        print('cvi42 xml file is', cvi42_file)
        # Parse file
        parse_cvi(cvi42_file, new_dir)
        return True

    return False


def getContour(contour_pickle, X, Y):
    '''
    Construct contour from points in pickle file and return
    in given dimensions.
    '''
    # The image annotation by default upsamples the image and then
    # annotate on the upsampled image.
    up = 4
    # Check whether there is a corresponding contour file for this dicom
    if os.path.exists(contour_pickle):
        with open(contour_pickle, 'rb') as f:
            contours = pickle.load(f)

            # Labels
            # short axis
            lv_endo = 1
            lv_epi = 2
            rv_endo = 3
            papil = 4
            enh_ref_myo = 6
            ref_myo = 7
            excl_enh = 10
            no_reflow = 20
            # Long axis
            la_endo = 4
            ra_endo = 5

            # Fill the contours in order
            # RV endocardium first, then LV epicardium,
            # then LV endocardium, then RA and LA.
            #
            # Issue: there is a problem in very rare cases,
            # e.g. eid 2485225, 2700750, 2862965, 2912168,
            # where LV epicardial contour is not a closed contour. This problem
            # can only be solved if we could have a better definition of contours.
            # Thanks for Elena Lukaschuk and Stefan Piechnik for pointing this out.

            # We skip the last point in the contours from cvi, otherwise
            # the polygon may present problems when closing. 
            print('----------->', contours.keys())
            ordered_contours = []
            if 'sarvendocardialContour' in contours:
                ordered_contours += [(contours['sarvendocardialContour'], rv_endo)]
            if 'larvendocardialContour' in contours:
                ordered_contours += [(contours['larvendocardialContour'][:-1], rv_endo)]

            if 'saepicardialContour' in contours:
                ordered_contours += [(contours['saepicardialContour'], lv_epi)]
            if 'saepicardialOpenContour' in contours:
                ordered_contours += [(contours['saepicardialOpenContour'], lv_epi)]

            # Close LV epicardium in long axis by taking the closest
            # points to the endocardium contour
            if 'laendocardialContour' in contours:
                aux = contours['laepicardialContour'].copy()
                start_closest = min(contours['laendocardialContour'], key=lambda x: np.linalg.norm(x-aux[0]))
                aux = np.concatenate(([start_closest], aux))
                end_closest = min(contours['laendocardialContour'], key=lambda x: np.linalg.norm(x-aux[-1]))
                aux = np.concatenate((aux, [end_closest]))
                contours['laepicardialContour'] = aux

            if 'laepicardialContour' in contours:
                ordered_contours += [(contours['laepicardialContour'][:-1], lv_epi)]
            if 'laepicardialOpenContour' in contours:
                ordered_contours += [(contours['laepicardialOpenContour'], lv_epi)]

            if 'saendocardialContour' in contours:
                ordered_contours += [(contours['saendocardialContour'], lv_endo)]
            if 'laendocardialContour' in contours:
                ordered_contours += [(contours['laendocardialContour'][:-1], lv_endo)]
            if 'saendocardialOpenContour' in contours:
                ordered_contours += [(contours['saendocardialOpenContour'], lv_endo)]
            if 'laendocardialOpenContour' in contours:
                ordered_contours += [(contours['laendocardialOpenContour'][:-1], lv_endo)]

            if 'saEnhancementReferenceMyoContour' in contours:
                ordered_contours += [(contours['saEnhancementReferenceMyoContour'], enh_ref_myo)]
            if 'saReferenceMyoContour' in contours:
                ordered_contours += [(contours['saReferenceMyoContour'], ref_myo)]

            if 'excludeEnhancementAreaContour' in contours:
                ordered_contours += [(contours['excludeEnhancementAreaContour'], excl_enh)]

            if 'noReflowAreaContour' in contours:
                ordered_contours += [(contours['noReflowAreaContour'], no_reflow)]

            if 'laraContour' in contours:
                ordered_contours += [(contours['laraContour'], ra_endo)]

            if 'lalaContour' in contours:
                ordered_contours += [(contours['lalaContour'], la_endo)]

            # if 'sapapilMuscContour' in contours:
            #     ordered_contours += [(contours['sapapilMuscContour'], papil)]

            # cv2.fillPoly requires the contour coordinates to be integers.
            # However, the contour coordinates are floating point number since
            # they are drawn on an upsampled image by 4 times.
            # We multiply it by 4 to be an integer. Then we perform fillPoly on
            # the upsampled image as cvi42 does. This leads to a consistent volume
            # measurement as cvi2. If we perform fillPoly on the original image, the
            # volumes are often over-estimated by 5~10%.
            # We found that it also looks better to fill polygons on the upsampled
            # space and then downsample the label map than fill on the original image.
            lab_up = np.zeros((Y * up, X * up))
            for c, l in ordered_contours:
                coord = np.round(c * up).astype(np.int32)
                # print('> coords', c[:10])
                # print('> coords', c[-10:])
                # Remove outlier points in contours. 
                # For some unknown reason, some outlier points appear.
                # b = np.linalg.norm(coord - np.mean(coord, axis=0), axis=1)
                # coord = coord[(b < np.mean(b) + 3*np.std(b))&(b > np.mean(b) - 3*np.std(b))]
                cv2.fillPoly(lab_up, [coord], l)

            return lab_up[::up, ::up].transpose(), lab_up.transpose()


def convert_contours(directory, output_dir):
    'Conver pickle files with countor coordinates to numpy arrays.'
    pickles = sorted(glob.iglob(os.path.join(directory, '*.pickle')))
    for f in pickles:
        print('> Parsing file', f)
        od = os.path.join(output_dir, os.path.basename(f)[:-7])
        os.makedirs(od, exist_ok=True)
        cnt, cnt_up = getContour(f, 256, 256)
        np.save(os.path.join(od, os.path.basename(f)[:-7]+'.npy'), cnt)
        np.save(os.path.join(od, os.path.basename(f)[:-7]+'_up.npy'), cnt_up)
        # break




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert DICOM files to NIFTI format.')
    parser.add_argument('directory', type=str, help='Directory path to contour file to be pased.')
    parser.add_argument('output', type=str, help='Path to output directory.')
    args = parser.parse_args()

    convert_contours(args.directory, args.output)
