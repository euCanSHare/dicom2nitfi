import os
import zlib
import regex
import pickle
import numpy as np
from xml.dom import minidom


def keepElementNodes(nodes):
    """ Get the element nodes """
    nodes2 = []
    for node in nodes:
        if node.nodeType == node.ELEMENT_NODE:
            nodes2 += [node]
    return nodes2


def parseContours(node):
    """
        Parse a Contours object. Each Contours object may contain several contours.
        We first parse the contour name, then parse the points and pixel size.
        """
    contours = {}
    for child in keepElementNodes(node.childNodes):
        contour_name = child.getAttribute('Hash:key')
        for child2 in keepElementNodes(child.childNodes):
            if child2.getAttribute('Hash:key') == 'Points':
                points = []
                for child3 in keepElementNodes(child2.childNodes):
                    x = float(child3.getElementsByTagName('Point:x')[0].firstChild.data)
                    y = float(child3.getElementsByTagName('Point:y')[0].firstChild.data)
                    points += [[x, y]]
            if child2.getAttribute('Hash:key') == 'SubpixelResolution':
                sub = int(child2.firstChild.data)
        points = np.array(points)
        points /= sub
        contours[contour_name] = points
    return contours


def traverseNode(node, uid_contours):
    """ Traverse the nodes """
    child = node.firstChild
    while child:
        if child.nodeType == child.ELEMENT_NODE:
            # This is where the information for each dicom file starts
            if child.getAttribute('Hash:key') == 'ImageStates':
                for child2 in keepElementNodes(child.childNodes):
                    # UID for the dicom file
                    uid = child2.getAttribute('Hash:key')
                    for child3 in keepElementNodes(child2.childNodes):
                        if child3.getAttribute('Hash:key') == 'Contours':
                            contours = parseContours(child3)
                            if contours:
                                uid_contours[uid] = contours
        traverseNode(child, uid_contours)
        child = child.nextSibling


def parseXmlFile(xml_name, output_dir):
    """ Parse a cvi42 xml file """
    dom = minidom.parse(xml_name)
    uid_contours = {}
    traverseNode(dom, uid_contours)

    # Save the contours for each dicom file
    cont_dir = os.path.join(output_dir, 'contours')
    os.mkdir(cont_dir)
    for uid, contours in uid_contours.items():
        with open(os.path.join(cont_dir, '{0}.pickle'.format(uid)), 'wb') as f:
            pickle.dump(contours, f)


def findContours(str_file):
    """
    Find C.o.n.t.o.u.r. key on hex file.
    Contour points are preceded by keywords
    such as Contour or Contour00X.
    """
    # Byte string to look for in corpus
    a = b'Contour'
    b = list(zip([a[i:i+1] for i in range(len(a))], [b'\x00']*len(a)))
    cntr = b''.join([i for j in b for i in j])

    # Clean contour associated to RefPoint
    # e.g. RV may have two contact points with MYO
    pos = 0
    f = 0
    while True:
        f = str_file[pos+1:].find(b'R\x00e\x00f\x00P\x00o\x00i\x00n\x00t\x00')
        if f < 0: break
        pos += f+1
        str_file = str_file[:pos-100] + str_file[pos+20:]
        pos -= 100

    # Extract contour points
    f = 0
    pos = 0
    uid = b'' # Identifier for a slice
    uid_contours = {}
    while True:
        f = str_file[pos+1:].find(cntr)
        if f < 0: break # No match found
        pos += f+1
        # Skip lines that contain the word "Contours".
        # These tags do not contain points.
        if b'Contours' in str_file[pos:pos+21].replace(b'\x00', b''): continue
        # Find contour name, i.e. alphabetic characters after Contour keyword.
        sr = regex.search(rb'[a-zA-Z\d]+$', str_file[pos-100:pos].replace(b'\x00', b''))
        contour = sr.group().decode('utf-8') + 'Contour'
        # Find uid. A sequence of numbers and dots of the form 1.5.12.145...
        sr = regex.search(rb'(?<=\x00{2}\x00.)((\x00\d)+\x00\.)+(\x00\d)+(?=\x00{3})', str_file[pos-300:pos])
        if sr is not None:
            uid = sr.group().replace(b'\x00', b'').decode('utf-8')
        elif uid in uid_contours.keys():
            if contour in uid_contours[uid]:
                # If this contour was already found, we did not find the correct uid.
                # Look further ahead for uid.
                sr = regex.search(rb'(?<=\x00{4}\x00[a-zA-Z\`\&\,])((\x00\d)+\x00\.)+(\x00\d)+(?=\x00{3})', str_file[pos-550:pos])
                if sr is not None:
                    uid = sr.group().replace(b'\x00', b'').decode('utf-8')
                else:
                    print('Warning! No uid found for contour with name "{}" at position {}'.format(
                        contour, pos))
                    continue
        # Start of contour points (FREE could be another keyword for automatic contours)
        sr = regex.search(rb'FREE', str_file[pos:pos+50])
        if sr is None: # When LINE is found, we skip it
            continue
        dif = sr.end()
        # Number of points in contour
        num = int.from_bytes(str_file[pos+dif+7:pos+dif+9], byteorder='big')
        # Save points to array staring from init.
        init = pos+dif+9
        points = []
        for i in range(num):
            x = int.from_bytes(str_file[init+2:init+4], byteorder='big')
            init += 4
            y = int.from_bytes(str_file[init+2:init+4], byteorder='big')
            init += 4
            points.append((x,y))

        # SubpixelResolution is usually 4, but sometimes it can be 2.
        # Should find a way to detect this.
        sub = 4
        points = np.array(points, dtype=np.float)
        points /= sub
        if uid not in uid_contours.keys():
            uid_contours[uid] = {}
        uid_contours[uid][contour] = points

    return uid_contours


def parseHexFile(filepath, output_dir):
    """Parse a cvi42 hexadecimal file"""
    # Uncompressed file in Zlib format
    str_object1 = open(filepath, 'rb').read()
    str_object2 = zlib.decompress(str_object1)
    decomp_name = filepath[:-8] + '.decoded'
    f = open(decomp_name, 'wb')
    f.write(str_object2)
    f.close()

    # Parse file in here
    uid_contours = findContours(str_object2)

    # Save the contours for each dicom file
    cont_dir = os.path.join(output_dir, 'contours')
    os.mkdir(cont_dir)
    for uid, contours in uid_contours.items():
        with open(os.path.join(cont_dir, '{0}.pickle'.format(uid)), 'wb') as f:
            pickle.dump(contours, f)



def parse(filepath, output_dir):
    '''
    Parse cvi42 file to obtain a contour. It creates
    a folder where final contour files are saved.
    Parameters:
        filepath: Path to the desired cvi42 file.
        output_dir: Directory where output files are to be saved.
    Return:
        None
    '''
    _, ext = os.path.splitext(filepath)

    if ext == '.cvi42ws': # Workspace file decoded and in HEX format
        parseHexFile(filepath, output_dir)
    elif ext == '.cvi42wsx': # Workspace in xml format
        parseXmlFile(filepath, output_dir)
    else:
        print('ERROR: Contour format "{}" from file "{}" not understood!'.format(
            ext, filepath
        ))
