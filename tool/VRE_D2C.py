#!/usr/bin/env python3

"""
.. See the NOTICE file distributed with this work for additional information
   regarding copyright ownership.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os

from basic_modules.metadata import Metadata
from utils import logger
from basic_modules.tool import Tool

from convert import DICOM_Dataset


class D2C_RUNNER(Tool):
    """
    Tool for segmenting a file
    """
    MASKED_KEYS = {
        'execution',
        'project',
        'description'
    }  # arguments from config.json

    def __init__(self, configuration=None):
        """
        Init function
        """
        logger.info("VRE D2C Workflow runner")
        Tool.__init__(self)

        if configuration is None:
            configuration = {}

        self.configuration.update(configuration)

        # Arrays are serialized
        for k, v in self.configuration.items():
            if isinstance(v, list):
                self.configuration[k] = ' '.join(v)

        self.populable_outputs = []


    def run(self, input_files, input_metadata, output_files):
        """
        The main function to run the compute_metrics tool.

        :param input_files: List of input files - In this case there are no input files required.
        :param input_metadata: Matching metadata for each of the files, plus any additional data.
        :param output_files: List of the output files that are to be generated.
        :type input_files: dict
        :type input_metadata: dict
        :type output_files: dict
        :return: List of files with a single entry (output_files), List of matching metadata for the returned files
        (output_metadata).
        :rtype: dict, dict
        """
        try:
            # Set and check execution directory. If not exists the directory will be created.
            execution_path = os.path.abspath(self.configuration.get('execution', '.'))
            execution_parent_dir = os.path.dirname(execution_path)
            if not os.path.isdir(execution_parent_dir):
                os.makedirs(execution_parent_dir)

            # Update working directory to execution path
            os.chdir(execution_path)
            logger.debug("Execution path: {}".format(execution_path))

            # Set file names for output files (with random name if not predefined)
            # output_path = ''
            # for ofile in output_files:
            #     if ofile["file_path"] is not None:
            #         pop_output_path = os.path.abspath(ofile["file_path"])
            #         if output_path == '':
            #             output_path = os.path.dirname(pop_output_path)
            #         self.populable_outputs.append(pop_output_path)
            #     else:
            #         errstr = "The output_file[{}] can not be located. Please specify its expected path.".format(key)
            #         logger.error(errstr)
            #         raise Exception(errstr)

            logger.debug("Init execution of the conversion tool")
            # Prepare file paths
            subjects = {}
            for key in input_files.keys():
                if key != 'bioimage':
                    logger.debug('Invalid key "{}". Should be bioimage.'.format(key))
                    continue
                for _file in input_files[key]:
                    # Convert DICOM images
                    aux = DICOM_Dataset(_file)
                    subjects.update(aux.getNiftis())

            output_files = []
            out_meta = []
            for subj in subjects.keys():
                for _file in subjects[subj]:
                    if os.path.isfile(_file["file_path"]):
                        meta = Metadata()
                        meta.file_path = _file["file_path"]  # Set file_path for output files
                        meta.data_type = 'bioimage'
                        meta.file_type = 'NIFTI'
                        meta.meta_data = _file
                        out_meta.append(meta)
                        output_files.append({
                            'name': 'bioimage', 'file_path': _file['file_path']
                        })

                        if 'mask_path' in _file:
                            meta = Metadata()
                            meta.file_path = _file["mask_path"]  # Set file_path for output files
                            meta.data_type = 'image_mask'
                            meta.file_type = 'NIFTI'
                            # Set sources for output files
                            meta.sources = [_file["file_path"]]
                            meta.meta_data = _file
                            out_meta.append(meta)
                            output_files.append({
                                'name': 'image_mask', 'file_path': _file['mask_path']
                            })

                            meta = Metadata()
                            meta.file_path = _file["upsample_mask_path"]  # Set file_path for output files
                            meta.data_type = 'image_mask'
                            meta.file_type = 'NIFTI'
                            meta.meta_data = _file
                            # Set sources for output files
                            meta.sources = [_file["file_path"]]
                            out_meta.append(meta)
                            output_files.append({
                                'name': 'image_mask', 'file_path': _file['upsample_mask_path']
                            })
                    else:
                        logger.warning("Output not found. Path \"{}\" does not exist".format(_file["file_path"]))

            output_metadata = {'output_files': out_meta}
            logger.debug("Output metadata created")

            return output_files, output_metadata

        except Exception:
            errstr = "VRE D2C RUNNER pipeline failed. See logs"
            logger.fatal(errstr)
            raise Exception(errstr)
