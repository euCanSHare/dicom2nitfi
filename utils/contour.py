import os
import cv2
import glob
import pickle

import numpy as np

from utils.parse_cvi42 import parse as parse_cvi


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
            enh_ref_myo = 5
            ref_myo = 6
            excl_enh = 10
            no_reflow = 20
            # Long axis
            la_endo = 1
            ra_endo = 2

            # Fill the contours in order
            # RV endocardium first, then LV epicardium,
            # then LV endocardium, then RA and LA.
            #
            # Issue: there is a problem in very rare cases,
            # e.g. eid 2485225, 2700750, 2862965, 2912168,
            # where LV epicardial contour is not a closed contour. This problem
            # can only be solved if we could have a better definition of contours.
            # Thanks for Elena Lukaschuk and Stefan Piechnik for pointing this out.
            ordered_contours = []
            if 'sarvendocardialContour' in contours:
                ordered_contours += [(contours['sarvendocardialContour'], rv_endo)]

            if 'saepicardialContour' in contours:
                ordered_contours += [(contours['saepicardialContour'], lv_epi)]
            if 'saepicardialOpenContour' in contours:
                ordered_contours += [(contours['saepicardialOpenContour'], lv_epi)]

            if 'saendocardialContour' in contours:
                ordered_contours += [(contours['saendocardialContour'], lv_endo)]
            if 'saendocardialOpenContour' in contours:
                ordered_contours += [(contours['saendocardialOpenContour'], lv_endo)]

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
                coord = np.round(c * up).astype(np.int)
                # Remove outlier points in contours. 
                # For some unknown reason, some outlier points appear.
                # b = np.linalg.norm(coord - np.mean(coord, axis=0), axis=1)
                # coord = coord[(b < np.mean(b) + 3*np.std(b))&(b > np.mean(b) - 3*np.std(b))]
                cv2.fillPoly(lab_up, [coord], l)

            return lab_up[::up, ::up].transpose(), lab_up.transpose()
