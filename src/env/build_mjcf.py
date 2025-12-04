import xml.etree.ElementTree as ET
from src.mujoco_helper.utils import prettify
import random

def build_mjcf_based_on_config(base_xml_path,recp_names, obj_names):
    """
    This function is used to build a mjcf file based on the config file.
    The config file is a list of dictionaries, each dictionary contains the following keys:
    - language_instruction: str
    - conditions: dict
        - objects: list of dict
            - names: tuple
            - relation: str
        - gripper: dict
            - state: str
            - pose: tuple
    - xml_file: str
    """
    # Load the base xml file
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    # Add receptacles
    for recp_name in recp_names:
        file_path = './recp/%s/model_new.xml'%(recp_name)
        include_tag = ET.Element('include',attrib={'file':file_path})
        root.append(include_tag)
    for obj_name in obj_names:
        file_path = './objects/%s/model_new.xml'%(obj_name)
        include_tag = ET.Element('include',attrib={'file':file_path})
        root.append(include_tag)
    xml_string = prettify(root) # indent xml
    xml_path = "./asset/temp.xml"
    with open(xml_path,'w') as f:
        f.write(xml_string)
    return xml_path
