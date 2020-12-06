import os, re, sys
import json, shutil

wd = os.path.dirname(os.path.realpath(__file__))
tool_dir = os.path.abspath(os.path.join(wd, '..'))
sys.path.append(tool_dir)

from main import main_json


DEFAULT_FILES = ['config.json', 'in_metadata.json', 'expected_out_metadata.json']


def create_local_files(json_name):
    '''
    Change keyword in json files by absolute path for
    the tool in each specific machine.
    '''
    filename = os.path.join(tool_dir, 'tests', 'testfiles', json_name)
    with open(filename, 'r') as _f:
        content = _f.read()
    # Substitute local tool path
    tmp_file = re.sub(r'\$\d{1}', tool_dir, content)
    # Save to new json file
    tmp_filename = re.sub(r'\.json', '_tmp.json', filename)
    with open(tmp_filename, 'w') as _f:
        _f.write(tmp_file)

    return True


def test_cmr():
    '''Test CMR studies.'''
    os.makedirs(os.path.join(wd, 'runTMP'), exist_ok=True)
    
    # Create temporal files used for this tool
    for _file in DEFAULT_FILES:
        create_local_files(_file)

    # Run tool with test files
    input_args = list(map(
        lambda x: 
            os.path.join(wd, 'testfiles', x.split('.')[0] + '_tmp.json'), 
        DEFAULT_FILES
    ))
    output_file = os.path.join(wd, 'runTMP', 'cmr_out_metadata.json')
    main_json(*input_args[:2], output_file)

    # Check output
    # Compare runTMP/cmr_out_metadata.json with cmr_expected_out_metadata.json
    with open(output_file, 'r') as _f:
        output = _f.read()
    with open(input_args[2], 'r') as _f:
        expected = _f.read()
    
    assert output == expected, 'Output files differ from expected.'

    # Remove output dir and tmp files
    for _f in input_args:
        os.remove(_f)
    shutil.rmtree(os.path.join(wd, 'runTMP'))

    return True


if __name__ == "__main__":
    test_cmr()