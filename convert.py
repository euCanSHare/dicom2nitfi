"""
    DICOM converting module.

    This module reads DICOM files for a subject,
    looks for the correct series (sometimes there are more than one series for one slice),
    stack the slices into a volume and save as a nifti image.

    pydicom is used for reading DICOM images. However, I have found that very rarely it could
    fail in reading certain DICOM images, perhaps due to the DICOM format, which has no standard
    and vary between manufacturers and machines.

    IMPORTANT!!
    The structure of the project SHOULD BE
        dataset folder
          --> patientA
            --> cvi42wsx file
            --> folderA with series
            --> folderB with series
            --> etc.
          --> patientB
          --> etc.
"""
import os, regex, sys, glob, pydicom, zipfile
import pickle, json, shutil, argparse
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk

from utils.contour import parseContours, getContour
from utils.post_processing import postProcess


class DICOM_Dataset(object):
    """ Class for managing DICOM datasets """
    def __init__(self, input_dir, output_dir, post_process=False):
        """
            Initialise data
            Group series by study type or field of view.
        """
        # Set output dir
        self.output_dir = output_dir
        
        # If input folder is a zip file, unzip it
        _, ext = os.path.splitext(input_dir)
        if ext == '.zip':
            zip_ref = zipfile.ZipFile(input_dir, 'r')
            zip_ref.extractall(os.path.dirname(input_dir))
            zip_ref.close()
            input_dir = os.path.dirname(input_dir)

        # Find patients and look for subdirs for each of them
        self.patient_dict = {}
        patients = os.listdir(input_dir)
        for p in sorted(patients):
            if '_transformed' in p:
                continue
            folder = os.path.join(input_dir, p)
            if not os.path.isdir(folder):
                continue
            print('Patient found', p)
            self.patient_dict[p] = []

            new_dir = os.path.join(input_dir, p + '_transformed')
            if os.path.exists(new_dir):
                os.system('rm -rf "{}"'.format(new_dir))
            os.mkdir(new_dir)

            # Process and extract contours
            self.cvi42_lb = parseContours(folder, new_dir)
            if self.cvi42_lb:
                contour_dir = os.path.join(new_dir, 'contours')
                avail_cont = [f.rstrip('.pickle') for f in os.listdir(contour_dir)]

            # Go into each subdir, list DICOM series, and move them to new folders
            filedf = pd.DataFrame(columns=['sop','siuid','src','dst','at','copy'])
            for root, _dir, files in os.walk(folder):
                for _file in files:
                    if _file == 'DICOMDIR':
                        continue
                    try:
                        ds = pydicom.dcmread(os.path.join(root, _file))
                    except pydicom.errors.InvalidDicomError:
                        # Handle invalid dicom files. These are found to be the empty files (0 bytes).
                        print('Found invalid dicom file {}. Skipping!'.format(os.path.join(root, _file)))
                        continue
                    # serDesc = ds.SeriesDescription
                    if not ds.__contains__('SeriesDescription') and not ds.__contains__('SequenceName'):
                        serDesc = ds.StudyDescription if ds.__contains__('StudyDescription') else 'unknown'
                    else:
                        serDesc = ds.SeriesDescription if ds.__contains__('SeriesDescription') else ds.SequenceName
                    serDesc = regex.sub( r' |-|\\|\/', '_', serDesc)
                    serDesc = regex.sub(r'\(|\)', '', serDesc)
                    desc = ds.SeriesInstanceUID
                    if 'Workspace' in serDesc:
                        # Ignore cvi42 workspace dicom
                        # Maybe this file can be used to extract contours as well
                        continue
                    try:
                        pos = np.array([float(x) for x in ds.ImagePositionPatient])
                        pos[:2] = -pos[:2]
                    except Exception:
                        print('ERROR: ImagePositionPatient attribute does not exist.')
                        continue

                    instNumber = int(ds.InstanceNumber)
                    if ds.__contains__('SliceLocation'):
                        slcLoc = ds.SliceLocation
                    else: # Choose patient position coordinate
                        slcLoc = pos[2]

                    try:
                        ffile = os.path.join(new_dir, desc, 'img{0}-{1:.4f}.dcm'.format(str(instNumber).zfill(4), slcLoc))
                    except AttributeError: # SliceLocation does not exist
                        print('ERROR: SliceLocation does not exist.')
                        continue

                    # Save new file
                    # Copy file if it does not exist yet or it's not set for copy
                    copy = False
                    if ffile not in filedf.loc[filedf['copy'] == True, 'dst'].values:
                        copy = True

                    new_row = {
                        'sop': ds.SOPInstanceUID,
                        'siuid': desc,
                        'src': os.path.join(root, _file),
                        'dst': ffile,
                        'at': ds.AcquisitionTime if ds.__contains__('AcquisitionTime') else '',
                        'copy': copy
                    }
                    filedf = filedf.append(new_row, ignore_index=True)

                    if not copy:
                        # This particular slice was already present in filedict
                        # This means we found a corrected version of some series 
                        # (probably due to some artifact or bad acquisition)
                        print('Repeated series', desc, os.path.basename(ffile))
                        at = ds.AcquisitionTime
                        if not self.cvi42_lb:
                            alternative_file = filedf.loc[(filedf['dst'] == ffile) & (filedf['at'] != ffile), 'src'].values[0]
                            ds2 = pydicom.dcmread(alternative_file)
                            # Skip file if it is an old version of an existing one
                            if float(ds.AcquisitionTime) < float(ds2.AcquisitionTime):
                                continue
                        else:
                            # When contours exist, keep images according to those files.
                            # If this particular image is contoured, we keep it.
                            # Although we must keep all temporal phases associated to this one
                            # that are linked by the AcquisitionTime.
                            if ds.SOPInstanceUID not in avail_cont: continue
                            ats = filedf.loc[filedf['dst'] == ffile, 'at'].values
                            other_at = list(set(ats) - set([at]))
                            if len(other_at) == 0: continue
                            other_at = other_at[0]
                            filedf.loc[filedf['at'] == other_at, 'copy'] = False
                            # Set copy to true for all slices for this acquisition time
                            filedf.loc[filedf['at'] == at, ['copy']] = True

            # Copy definitive files around
            for _, row in filedf.iterrows():
                if not row['copy']: continue
                os.makedirs(os.path.dirname(row['dst']), exist_ok=True)
                shutil.copy(row['src'], row['dst'])

            self.transformDicoms(new_dir, p)

            # File postprocessing
            if post_process:
                try:
                    postProcess(new_dir)
                except Exception:
                    continue
            
            # Delete origin folder
            os.system('rm -rf {}'.format(os.path.join(input_dir, p)))

        # Remove DICOM files from converted folders
        for f in glob.iglob(os.path.join(input_dir, '*_transformed', '**', '*.dcm'), recursive=True):
            os.remove(f)


    def transformDicoms(self, dicom_dir, patient):
        avail_cont = []
        self.cvi42_lb_used = False
        if self.cvi42_lb:
            contour_dir = os.path.join(dicom_dir, 'contours')
            avail_cont = [f.rstrip('.pickle') for f in os.listdir(contour_dir)]
        # Go through each folder and transform it to nifti files
        processed = []
        for s in sorted(os.listdir(dicom_dir)):
            SKIP = False
            print('Series', s)
            if s == 'contours':
                continue

            # Look for series that have the same name, but for the last three digits
            # or the last code
            spl = s.split('.')
            prefix = ''
            if len(spl[-1]) > 3:
                prefix = spl[-1]
            elif len(spl[-2]) > 3:
                prefix = s
            else:
                prefix = '.'.join(spl[:-3])
            # Process each series once
            if prefix in processed:
                print('Series already considered! {}. Skipping!'.format(prefix))
                continue
            processed.append(prefix)
            matchSeries = list(glob.iglob(os.path.join(dicom_dir, prefix + '.*')))
            # Set destination folder to the series
            folder = os.path.join(dicom_dir, s)
            files = []
            if len(matchSeries) < 2:
                files = list(glob.iglob(os.path.join(folder, '*.dcm')))
            else:
                for ms in sorted(matchSeries):
                    files.extend(list(glob.iglob(os.path.join(ms, '*.dcm'))))
            # Compute "normalised" file name. Avoid instance numbers that are not sequential
            files_normalised = []
            for zpos in np.unique([os.path.basename(f)[8:].rstrip('.dcm') for f in files]):
                aux_files = sorted([f for f in files if os.path.basename(f)[8:].rstrip('.dcm') == zpos])
                aux_norm = [regex.sub(r'img\d{4}\-', 'img{}-'.format(str(idx+1).zfill(4)), os.path.basename(f)) for idx, f in enumerate(aux_files)]
                files_normalised.extend(list(zip(aux_files, aux_norm)))
            # List of files with Z value in position 2
            file_list = [[f, fnorm, float(os.path.basename(f)[8:].rstrip('.dcm'))] for f, fnorm in files_normalised]

            # Vertical slices (Z) and time steps (T)
            Z = len(set([float(os.path.basename(i)[8:].rstrip('.dcm')) for i in files]))
            T = 1 + (len(files)-Z) // Z
            print('T', T, ' Z', Z)
            # print('-> files', files)
            if T*Z < len(files):
                print('DICOM files seem to come from different cine images or have different time steps.')
                print('Series will be grouped separately.')
                print('Work in progress...')
                # TODO
                continue

            files_time = []
            # Order file list so that the slices go from earlier to later times
            # and for each time step we have increasing Z component
            file_list = sorted(file_list, key=lambda x: x[1])
            if T >= 2 and Z >= 2:
                for t in range(T):
                    file_list[t*Z:(t+1)*Z] = sorted(file_list[t*Z:(t+1)*Z], key=lambda x: x[2])
            elif T == 1: # Order by Z in case there is no temporal axis (just in case). It fails on DE sax images
                file_list = sorted(file_list, key=lambda x: x[2])

            # Create default volume and fill it with corresponding images
            ds = pydicom.dcmread(file_list[0][0])
            if not ds.__contains__('SeriesDescription') and not ds.__contains__('SequenceName'):
                serDesc = ds.StudyDescription if ds.__contains__('StudyDescription') else 'unknown'
            else:
                serDesc = ds.SeriesDescription if ds.__contains__('SeriesDescription') else ds.SequenceName
            if serDesc == 'MESURES': continue
            serDesc = regex.sub( r' |-|\\|\/', '_', serDesc)
            serDesc = regex.sub(r'\(|\)', '', serDesc)
            Y, X = ds.Rows, ds.Columns
            volume = np.zeros((X,Y,Z,T), dtype='float32')
            label  = np.zeros((X,Y,Z,T), dtype='uint8')
            label_up  = np.zeros((X*4,Y*4,Z,T), dtype='uint8')
            for idx, fi in enumerate(file_list):
                f = fi[0]
                ds = pydicom.dcmread(f)
                reader = sitk.ImageFileReader()
                reader.SetFileName(f)
                try:
                    im = sitk.GetArrayFromImage(reader.Execute()).squeeze()
                except Exception: 
                    print('ERROR: Unable to read file with SimpleITK')
                    SKIP = True # Skip series
                    break
                try:
                    volume[..., idx%Z, idx//Z] = im.transpose()
                except ValueError:
                    print('Debug: Problem with transposing image. Saving as it comes.')
                    volume[..., idx%Z, idx//Z] = im
                if ds.SOPInstanceUID in avail_cont:
                    self.cvi42_lb_used = True # Contouring found
                    lab_down, lab_up = getContour(os.path.join(contour_dir, '{}.pickle'.format(ds.SOPInstanceUID)), X, Y)
                    label[..., idx%Z, idx//Z] = lab_down
                    label_up[..., idx%Z, idx//Z] = lab_up

                t = 0
                if ds.__contains__('TriggerTime'):
                    t = ds.TriggerTime
                files_time.append(t)
            
            # If SimpleITK failed to read some dicom file, we skip this series
            if SKIP: continue

            dx = float(ds.PixelSpacing[1])
            dy = float(ds.PixelSpacing[0])
            # Temporal spacing
            dt = (files_time[Z-1] - files_time[0]) * 1e-3

            # DICOM coordinate (LPS)
            #  x: left
            #  y: posterior
            #  z: superior
            # Nifti coordinate (RAS)
            #  x: right
            #  y: anterior
            #  z: superior
            # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
            # Refer to
            # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
            # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

            # The coordinate of the upper-left voxel of the first and second slices
            ds = pydicom.dcmread(file_list[0][0])
            pos_ul = np.array([float(x) for x in ds.ImagePositionPatient])
            pos_ul[:2] = -pos_ul[:2]

            # Image orientation
            axis_x = np.array([float(x) for x in ds.ImageOrientationPatient[:3]])
            axis_y = np.array([float(x) for x in ds.ImageOrientationPatient[3:]])
            axis_x[:2] = -axis_x[:2]
            axis_y[:2] = -axis_y[:2]

            if Z >= 2:
                # Read a dicom file at the second slice
                d2 = pydicom.dcmread(file_list[1][0])
                pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
                pos_ul2[:2] = -pos_ul2[:2]
                axis_z = pos_ul2 - pos_ul
                axis_z = axis_z / np.linalg.norm(axis_z)
            else:
                axis_z = np.cross(axis_x, axis_y)

            # Determine the z spacing
            if hasattr(ds, 'SpacingBetweenSlices'):
                dz = float(ds.SpacingBetweenSlices)
            elif Z >= 2:
                print('Debug: can not find attribute SpacingBetweenSlices. '
                      'Calculate from two successive slices.')
                dz = float(np.linalg.norm(pos_ul2 - pos_ul))
            else:
                print('Debug: can not find attribute SpacingBetweenSlices. '
                      'Use attribute SliceThickness instead.')
                dz = float(ds.SliceThickness)

            # Affine matrix which converts the voxel coordinate to world coordinate
            affine = np.eye(4)
            affine[:3,0] = axis_x * dx
            affine[:3,1] = axis_y * dy
            affine[:3,2] = axis_z * dz
            affine[:3,3] = pos_ul
            nii = nib.Nifti1Image(volume, affine)
            nii.header['pixdim'][4] = dt
            nii.header['sform_code'] = 1
            nib.save(nii, os.path.join(self.output_dir, '{}.nii.gz'.format(serDesc)))
            aux_file = {}
            aux_file['file_path'] = os.path.join(self.output_dir, '{}.nii.gz'.format(serDesc))
            if self.cvi42_lb:
                nii_lb = nib.Nifti1Image(label, affine)
                nii_lb.header['pixdim'][4] = dt
                nii_lb.header['sform_code'] = 1
                nib.save(nii_lb, os.path.join(self.output_dir, '{}_label.nii.gz'.format(serDesc)))
                nii_lb_up = nib.Nifti1Image(label_up, affine)
                nii_lb_up.header['pixdim'][4] = dt
                nii_lb_up.header['sform_code'] = 1
                nib.save(nii_lb_up, os.path.join(self.output_dir, '{}_label_upsample.nii.gz'.format(serDesc)))
                aux_file['mask_path'] = os.path.join(self.output_dir, '{}_label.nii.gz'.format(serDesc))
                aux_file['upsample_mask_path'] = os.path.join(self.output_dir, '{}_label_upsample.nii.gz'.format(serDesc))

        # Save information of patient
        studyDate = ds.StudyDate if ds.__contains__('StudyDate') else ''
        patientBD = ds.PatientBirthDate[:4] if ds.__contains__('PatientBirthDate') else ''
        age = ''
        if ds.__contains__('PatientAge'):
            age = ds.PatientAge
        elif studyDate != '' and patientBD != '':
            # Compute age at the time of scan
            age = int(studyDate[:4]) - int(patientBD)

        # Image view and use of contrast
        patt = ['cine_short_axis', 'cine_short_axis_6MM', 'CINE_EC.*_apex', 'EC_.*_FIL', 'EC_.*_10slices',
                'CINE_EC_barrido', 'CINE_EC', 'cine_.*_EC', 'SHORT_AXIS', 'CINE_EJE_CORTO', 
                'FUNCION_VI', '.*\_\#SA', '.*SAX.*', 'viabilidad']
        sa = np.any([regex.search(p, serDesc) for p in patt])
        patt = ['viabilidad', '.*RTG']
        ge = np.any([regex.search(p, serDesc) for p in patt])
        patt = ['.*2C.*', '.*2_C.*']
        c2 = np.any([regex.search(p, serDesc) for p in patt])
        patt = ['.*3C.*', '.*3_C.*']
        c3 = np.any([regex.search(p, serDesc) for p in patt])
        patt = ['.*4C.*', '.*4_C.*']
        c4 = np.any([regex.search(p, serDesc) for p in patt])
        view = ''
        contrast = False
        if c2:
            view = '2chambers'
        elif c3:
            view = '3chambers'
        elif c4:
            view = '4chambers'
        elif sa:
            view = 'short_axis'

        if ge:
            contrast = True

        info = {
            'age': age,
            'patientID': ds.PatientID if ds.__contains__('PatientID') else '',
            'sex': ds.PatientSex if ds.__contains__('PatientSex') else '',
            'weight': ds.PatientWeight if ds.__contains__('PatientWeight') else '',
            'size': ds.PatientSize if ds.__contains__('PatientSize') else '',
            'vendor': ds.Manufacturer if ds.__contains__('Manufacturer') else '',
            'institution': ds.InstitutionName if ds.__contains__('InstitutionName') else '',
            'modality': ds.Modality if ds.__contains__('Modality') else '',
            'bodyPartExamined': ds.BodyPartExamined if ds.__contains__('BodyPartExamined') else '',
            'view': view, 'contrast': contrast,
            'studyDate': studyDate
        }
        # with open(os.path.join(folder, '..', 'info.json'), 'w') as f:
        #     json.dump(info, f)
        
        aux_file.update(info)
        self.patient_dict[patient].append(aux_file)


    def getNiftis(self):
        return self.patient_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert DICOM files to NIFTI format.')
    parser.add_argument('path', type=str, help='Directory path to dataset that must be converted.')
    parser.add_argument('--pp', type=bool, default=False, help='Whether or not to apply post-processing to the final images.')
    args = parser.parse_args()

    DICOM_Dataset(args.path, args.pp)
