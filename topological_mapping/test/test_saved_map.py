from ament_index_python.packages import get_package_share_directory
import unittest
import os
import pathlib
import numpy as np
map_folder = os.path.join(get_package_share_directory('topological_mapping'),'map.npy') 
map = np.load(map_folder,allow_pickle=True) 
def test_map():
    flag = False    
    for row in range(map.shape[0]):
            for colum in range(map.shape[1]):
                for state in range(map.shape[2]):
                    for i in range(10):
                        if map[row,colum,state,i][0] != 0: 
                            print(map[row,colum,state,i][0])
                            flag = True
    assert flag

def test_all_states_saved():
    subdir_count = 0
    states_count = 0
    rootdir = str(pathlib.Path(__file__).parent.resolve()).removesuffix('/test') + '/data/'
    for subdir, dirs, files in os.walk(rootdir):
        subdir_count += 1
    for row in range(map.shape[0]):
            for colum in range(map.shape[1]):
                for state in range(map.shape[2]):
                    for i in range(10):
                        if map[row,colum,state,i][0] != 0: 
                            states_count += 1
    assert subdir_count == states_count

    
                                                             

       