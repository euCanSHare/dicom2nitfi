import os
import glob
import uuid
import pydicom


dcm_keys_to_keep = [
    ('0008', '0005'), ('0008', '0008'), ('0008', '0016'), ('0008', '0018'), ('0008', '0020'),
    ('0008', '0021'), ('0008', '0022'), ('0008', '0023'), ('0008', '0030'), ('0008', '0031'),
    ('0008', '0032'), ('0008', '0033'), ('0008', '0060'), ('0008', '0070'), ('0008', '0080'),
    ('0008', '0081'), ('0008', '1010'), ('0008', '1030'), ('0008', '1032'), ('0008', '103e'),
    ('0008', '1090'), ('0010', '0010'), ('0010', '0020'), ('0010', '0030'), ('0010', '0040'),
    ('0010', '1010'), ('0010', '1020'), ('0018', '0020'), ('0018', '0021'), ('0018', '0022'),
    ('0018', '0023'), ('0018', '0024'), ('0018', '0025'), ('0018', '0050'), ('0018', '0080'),
    ('0018', '0081'), ('0018', '0083'), ('0018', '0084'), ('0018', '0085'), ('0018', '0086'),
    ('0018', '0087'), ('0018', '0089'), ('0018', '0091'), ('0018', '0093'), ('0018', '0094'),
    ('0018', '0095'), ('0018', '1000'), ('0018', '1020'), ('0018', '1030'), ('0018', '1060'),
    ('0018', '1062'), ('0018', '1251'), ('0018', '1310'),
    ('0018', '1312'), ('0018', '1314'), ('0018', '1315'), ('0018', '1316'), ('0018', '1318'),
    ('0018', '5100'), ('0020', '000d'), ('0020', '000e'), ('0020', '0010'), ('0020', '0011'),
    ('0020', '0012'), ('0020', '0013'), ('0020', '0032'), ('0020', '0037'), ('0020', '0052'),
    ('0020', '1040'), ('0020', '1041'), ('0020', '4000'), ('0028', '0002'), ('0028', '0004'),
    ('0028', '0010'), ('0028', '0011'), ('0028', '0030'), ('0028', '0100'), ('0028', '0101'),
    ('0028', '0102'), ('0028', '0103'), ('0028', '0106'), ('0028', '0107'), ('0028', '1050'),
    ('0028', '1051'), ('0028', '1055'), ('7fe0', '0010')
]


def anonymize_dicoms(root_dir, ext_id):
    for folder in os.listdir(root_dir):
        folder = os.path.join(root_dir, folder)
        # Study Instance UID
        stu_uid = '2.25.' + str(uuid.uuid4().int)
        # Series Instance UID
        ser_uid = '2.25.' + str(uuid.uuid4().int)
        # Frame of reference UID
        for_uid = '2.25.' + str(uuid.uuid4().int)
        # Read all dcm files and copy them to dst folder
        for i, dcm_file in enumerate(glob.iglob(os.path.join(folder, '*.dcm'))):
            ds = pydicom.dcmread(dcm_file)
            # Set SOP Instance and Class UID
            sop_uid = '2.25.' + str(uuid.uuid4().int)
            sop_cl_uid = '2.25.' + str(uuid.uuid4().int)
            ds.SOPInstanceUID = sop_uid
            ds.SOPClassUID = sop_cl_uid
            # Set name from External id
            ds.PatientName = ext_id
            ds.PatientID = ext_id
            # Delete all private information
            sdt = ds.StudyDate[:4] + '0101'
            ds.StudyDate = sdt
            ds.SeriesDate = sdt
            ds.SeriesDescription = 'SA'
            ds.SeriesInstanceUID = ser_uid
            ds.AcquisitionDate = sdt
            ds.ContentDate = sdt
            ds.StudyID = 'MR' + sdt
            ds.StudyDescription = 'MR_Heart'
            ds.StudyInstanceUID = stu_uid
            ds.FrameOfReferenceUID = for_uid

            ds.InstitutionName = ''
            ds.InstitutionAddress = ''
            ds.ReferringPhysicianName = ''
            ds.PerformingPhysicianName = ''
            ds.OperatorsName = ''
            ds.PatientBirthDate = ds.PatientBirthDate[:4] + '0101'
            ds.SequenceName = ''
            ds.ProtocolName = ''
            ds.SoftwareVersions = ''

            ks = list(ds.keys())
            for k in ks:
                if k not in dcm_keys_to_keep:
                    del ds[k]

            # Overwrite file
            pydicom.dcmwrite(dcm_file, ds)
