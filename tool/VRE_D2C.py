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

from eval import run


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
            output_path = ''
            for ofile in output_files:
                if ofile["file_path"] is not None:
                    pop_output_path = os.path.abspath(ofile["file_path"])
                    if output_path == '':
                        output_path = os.path.dirname(pop_output_path)
                    self.populable_outputs.append(pop_output_path)
                else:
                    errstr = "The output_file[{}] can not be located. Please specify its expected path.".format(key)
                    logger.error(errstr)
                    raise Exception(errstr)

            logger.debug("Init execution of the Segmentation")
            # Prepare file paths
            for key in input_files.keys():
                if key == 'model':
                    model = input_files[key]
                elif key == 'images':
                    datasets = input_files[key]
                else:
                    logger.debug('Unrecognized key {}'.format(key))
                    continue

            # Segment images
            run(model, datasets, output_path)

            out_meta = []
            for ofile in output_files:
                if os.path.isfile(ofile["file_path"]):
                    meta = Metadata()
                    meta.file_path = ofile["file_path"]  # Set file_path for output files
                    meta.data_type = 'image_mask'
                    meta.file_type = 'NIFTI'

                    # Set sources for output files
                    meta.sources = [ofile["file_path"].rstrip('_mask.nii.gz') + '.nii.gz']

                    # Append new element in output metadata
                    logger.info('Update metadata value {}'.format(meta.file_path))
                    out_meta.append(meta)

                else:
                    logger.warning("Output not found. Path {} does not exist".format(ofile["file_path"]))

            output_metadata = {'output_files': out_meta}
            logger.debug("Output metadata created")

            return output_files, output_metadata

        except Exception:
            errstr = "VRE D2C RUNNER pipeline failed. See logs"
            logger.fatal(errstr)
            raise Exception(errstr)
