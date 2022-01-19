# -*- coding: utf-8 -*-
"""This module saves and retrieves data, dealing with overwriting.

It could be divided into 2 main sections:
    
    (1) making new directories and free files to avoid overwriting 
    (`new_dir`, `free_file`)
    (2) saving data into files with the option of not overwriting 
    (`saveplot`, `savetext`, `savewav`)
    (3) loading or retrieving data saved with this module 
    (`retrieve_header`, `retrieve_footer`)
    
new_dir : function
    Makes and returns a new related directory to avoid overwriting.
free_file : function
    Returns a name for a new file to avoid overwriting.
saveplot : function
    Saves a matplotlib.pyplot plot on an image file (i.e: 'png').
savetxt : function
    Saves some np.array like data on a '.txt' file.
savewav : function
    Saves a PyAudio encoded audio on a '.wav' file.
saveanimation : function
    Saves a matplotlib.animation object as '.gif' or '.mp4'.
@author: Vall
"""

from datetime import datetime
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from socket import gethostname
import v_utilities as vu

#%%

def init_system(interactive=False):
    
    if interactive:
        sysname = input("Choose a nickname for your current PC; i.e. MC\n\n")
        answer = input(f"Is this repository's directory correct?\n{os.getcwd()}\nAnswer yes or no (Y/N)\n\n")
        if "y" in answer.lower():
            syshome = os.getcwd()
        else:
            syshome = input("Copy and paste your code repository's full directory;\n"+
                            "i.e. C:\\Users\\Me\\SimplePixelSimulations\n\n")
        home = input("Copy and paste your results folder full directory;\n"+
                     "i.e. C:\\Users\\Me\\SimplePixelSimulationsResults\n\n")
                        
        if not os.path.isdir(home):
            os.makedirs(home)
            print("Results folder was not found so it has been created")
            
        sysname_definition = {gethostname(): sysname}
        syshome_definition = {gethostname(): syshome}
        home_definition = {gethostname(): home}
    
    else:
        sysname_definition = {"Nano": "SC", "vall": "MC", "else": "TC"}
        home_definition = {"Nano": "/home/nanofisica/Documents/Vale/PyMeepResults",
                           "SC": "/home/vall/Documents/Thesis/PyMeepResults", 
                           "vall": "/nfs/home/vpais/PyMeepResults"}
        syshome_definition = {"Nano": "/home/nanofisica/Documents/Vale/PyMeepPlasmonics",
                              "SC": "/home/vall/Documents/Thesis/PyMeepPlasmonics", 
                              "vall": "/nfs/home/vpais/PyMeepPlasmonics"}
        
    default_system = dict(sysname_definition=sysname_definition,
                          home_definition=home_definition,
                          syshome_definition=syshome_definition)
    try:
        savetxt(os.path.join(syshome, "SystemDirectories.txt"), 
                np.array([]), footer=default_system, overwrite=True)
    except UnboundLocalError:
        savetxt(os.path.join(os.getcwd(), "SystemDirectories.txt"), 
                np.array([]), footer=default_system, overwrite=True)        
    
    return

#%%

def add_or_edit_system(sysname, syshome, home=None):
    
    default_path = os.getcwd()
    
    current_system = retrieve_footer(os.path.join(default_path, 
                                                  "SystemDirectories.txt"))
    
    hostname = gethostname()
    
    if hostname in current_system["sysname"].keys():
        print("Current host's system will be updated")
    else:
        print("New system entry will be created for current host")
        
    if home is None:
        home = current_system["home"][hostname]
    
    current_system["sysname"][hostname] = sysname
    current_system["syshome"][hostname] = syshome
    current_system["home"][hostname] = home

#%%

def setup_system():
    
    default_path = os.getcwd()
    
    current_system = retrieve_footer(os.path.join(default_path, 
                                                  "SystemDirectories.txt"))
    
    sysname_definition = current_system["sysname_definition"]
    home_definition = current_system["home_definition"]
    syshome_definition = current_system["syshome_definition"]
    
    return sysname_definition, home_definition, syshome_definition


#%% 

def get_sys_name():
    """Returns system name according to which CPU is running"""
    
    string = gethostname()
    try: return sysname_definition[string]
    except: 
        try: return sysname_definition["else"]
        except: raise ValueError("Your PC must appear inside sysname definition")

#%%

def get_home():

    """Returns home path for results according to which CPU is running"""
    
    string = gethostname()
    try: return home_definition[string]
    except: 
        try: return home_definition["else"]
        except: raise ValueError("Your PC must appear inside home definition")

#%%

def get_sys_home():

    """Returns home path for repository according to which CPU is running"""
    
    string = gethostname()
    try: return syshome_definition[string]
    except: 
        try: return syshome_definition["else"]
        except: raise ValueError("Your PC must appear inside syshome definition")
        
#%%

def datetime_dir(path, strftime="(%Y-%m-%d)(%H:%M:%S)"):
    
    """Returns a path decorated with timestamp at the end.
    
    Takes the path to both a file or a directory.
    
    Parameters
    ----------
    path : str
        Original path.
    strftime="(%Y-%m-%d)(%H:%M:%S)" : str
        Date and time formatter.
    
    Returns
    -------
    new_dir : str
        New path, containing the timestamp.
    
    See Also
    --------
    datetime.date.strftime
    
    """
    
    date = datetime.now().strftime(strftime)
    
    base = os.path.split(path)[0]
    name = os.path.splitext(os.path.split(path)[-1])[0]
    extension = os.path.splitext(path)[-1]
    
    new_path = os.path.join(base, name+date+extension)
    
    return new_path

#%%

def new_dir(my_dir, newformat='{}_{}'):
    
    """Makes and returns a new directory to avoid overwriting.
    
    Takes a directory name 'my_dir' and checks whether it already 
    exists. If it doesn't, it returns 'dirname'. If it does, it 
    returns a related unoccupied directory name. In both cases, 
    the returned directory is initialized.
    
    Parameters
    ----------
    my_dir : str
        Desired directory (should also contain full path).
    
    Returns
    -------
    new_dir : str
        New directory (contains full path)
    
    Yields
    ------
    new_dir : directory
    
    """
    
    sepformat = newformat.split('{}')
    base = os.path.split(my_dir)[0]
    
    new_dir = my_dir
    while os.path.isdir(new_dir):
        new_dir = os.path.basename(new_dir)
        new_dir = new_dir.split(sepformat[-2])[-1]
        try:
            new_dir = new_dir.split(sepformat[-1])[0]
        except ValueError:
            new_dir = new_dir
        try:
            new_dir = newformat.format(my_dir, str(int(new_dir)+1))
        except ValueError:
            new_dir = newformat.format(my_dir, 2)
        new_dir = os.path.join(base, new_dir)
    os.makedirs(new_dir)
        
    return new_dir

#%%

def free_file(my_file, newformat='{}_{}'):
    
    """Returns a name for a new file to avoid overwriting.
        
    Takes a file name 'my_file'. It returns a related unnocupied 
    file name 'free_file'. If necessary, it makes a new 
    directory to agree with 'my_file' path.
        
    Parameters
    ----------
    my_file : str
        Tentative file name (must contain full path and extension).
    newformat='{}_{}' : str
        Format string that indicates how to make new names.
    
    Returns
    -------
    new_fname : str
        Unoccupied file name (also contains full path and extension).
    
    """
    
    base = os.path.split(my_file)[0]
    extension = os.path.splitext(my_file)[-1]
    
    if not os.path.isdir(base):
        os.makedirs(base)
        free_file = my_file
    
    else:
        sepformat = newformat.split('{}')[-2]
        free_file = my_file
        while os.path.isfile(free_file):
            free_file = os.path.splitext(free_file)[0]
            free_file = free_file.split(sepformat)
            number = free_file[-1]
            free_file = free_file[0]
            try:
                free_file = newformat.format(
                        free_file,
                        str(int(number)+1),
                        )
            except ValueError:
                free_file = newformat.format(
                        os.path.splitext(my_file)[0], 
                        2)
            free_file = os.path.join(base, free_file+extension)
    
    return free_file

#%%
    
def new_name(name, newseparator='_'):

    '''Returns a name of a unique file or directory so as to not overwrite.
    
    If proposed name existed, will return name + newseparator + number.
     
    Parameters:
    -----------
        name : str (path)
            proposed file or directory name influding file extension
        nweseparator : str
            separator between original name and index that gives unique name
            
    Returns:
    --------
        name : str
            unique namefile using input 'name' as template
    '''
    
    #if file is a directory, extension will be empty
    base, extension = os.path.splitext(name)
    i = 2
    while os.path.exists(name):
        name = base + newseparator + str(i) + extension
        i += 1
        
    return name

#%%

def saveplot(file, overwrite=False):
    
    """Saves a plot on an image file.
    
    This function saves the current matplotlib.pyplot plot on a file. 
    If 'overwrite=False', it checks whether 'file' exists or not; if it 
    already exists, it defines a new file in order to not allow 
    overwritting. If overwrite=True, it saves the plot on 'file' even if 
    it already exists.
    
    Variables
    ---------
    file : string
        The name you wish (must include full path and extension)
    overwrite=False : bool
        Indicates whether to overwrite or not.
    
    Returns
    -------
    nothing
    
    Yields
    ------
    an image file
    
    See Also
    --------
    free_file()
    
    """
    
    if not os.path.isdir(os.path.split(file)[0]):
        os.makedirs(os.path.split(file)[0])
    
    if not overwrite:
        file = free_file(file)

    plt.savefig(file, bbox_inches='tight')
    
    print('Archivo guardado en {}'.format(file))
    

#%%

def savetxt(file, datanumpylike, overwrite=False, header='', footer=''):
    
    """Takes some array-like data and saves it on a '.txt' file.
    
    This function takes some data and saves it on a '.txt' file.
    If 'overwrite=False', it checks whether 'file' exists or not; if it 
    already exists, it defines a new file in order to not allow 
    overwritting. If overwrite=True, it saves the plot on 'file' even if 
    it already exists.
    
    Variables
    ---------
    file : string
        The name you wish (must include full path and extension)
    datanumpylike : array, list
        The data to be saved.
    overwrite=False : bool, optional
        Indicates whether to overwrite or not.
    header='' : list, str, optional
        Data's descriptor. Its elements should be str, one per column.
        But header could also be a single string.
    footer='' : dict, str, optional
        Data's specifications. Its elements and keys should be str. 
        But footer could also be a single string. Otherwise, an element 
        could be a tuple containing value and units; i.e.: (100, 'Hz').
    
    Return
    ------
    nothing
    
    Yield
    -----
    '.txt' file
    
    See Also
    --------
    free_file()
    
    """
    
    base = os.path.split(file)[0]
    if not os.path.isdir(base):
        os.makedirs(base)
    
    if header != '':
        if not isinstance(header, str):
            try:
                header = '\t'.join(header)
            except:
                TypeError('Header should be a list or a string')

    if footer != '':
        if not isinstance(footer, str):
            try:
                aux = []
                for key, value in footer.items():
                    if isinstance(value, np.ndarray) or isinstance(value, tuple):
                        value = str(list(value))
                    elif isinstance(value, str):
                        value = '"{}"'.format(value)
                    aux.append('{}={}'.format(key, value) + ', ')
                footer = ''.join(aux)
            except:
                TypeError('Header should be a dict or a string')

    file = os.path.join(
            base,
            (os.path.splitext(os.path.basename(file))[0] + '.txt'),
            )
    
    if not overwrite:
        file = free_file(file)
        
    np.savetxt(file, np.array(datanumpylike), 
               delimiter='\t', newline='\n', header=header, footer=footer)
    
    print('Archivo guardado en {}'.format(file))
    
    return

#%%

def saveanimation(file,
                  animation,
                  frames_per_second=30,
                  overwrite=False):
    
    """Saves a matplotlib.animation object as '.gif' or '.mp4'.
    
    Variables
    ---------
    file : str
        Desired file (must include full path and extension).
    animation : matplotlib.animation object
        Animation to save.
    frames_per_second=30 : int
        Animation's frames per second.
    overwrite=False : bool
        Indicates wheter to overwrite or not.
        
    Returns
    -------
    nothing
    
    Yields
    ------
    video file
    
    Warnings
    --------
    To save '.gif' you must have ImageMagick installed.
    To save '.mp4' you must have FFPG installed.
    
    See Also
    --------
    free_file()
    fwp_plot.animation_2D()
    
    """
    
    if not os.path.isdir(os.path.split(file)[0]):
        os.makedirs(os.path.split(file)[0])
    
    if not overwrite:
        file = free_file(file)
    
    extension = os.path.splitext(file)[-1]
    
    if extension == '.mp4':
        animation.save(file,
                       extra_args=['-vcodec', 'libx264'])
    elif extension == '.gif':
        animation.save(file,
                       dpi=50,
                       writer='imagemagick')    
    
    print('Archivo guardado en {}'.format(file))
    
    return

#%%

def save_slice_generator(sim, filename, datanames, get_slices):

    """Generates a Meep stepfunction to save slices or outputs to HDF5.

    Parameters
    ----------
    sim : mp.Simulation
        Meep simulation instance. Must be initialized so that ground set of 
        data can be saved.
    filename : str
        Filename of the HDF5 file to be created. Must include full path and .h5 
        extension. Beware! If already in existance, old file is replaced.
    datanames : str or list of str
        HDF5 dataset name.
    get_slices : function or list of functions
        Function 'get_slice(sim, state)' that takes 'sim' Meep simulation 
        instance as required argument and may take a second argument 'state', 
        which is a string with value 'step' or 'finish'. Must return the 
        desired array to be saved for each time step that the stepfunction 
        is called on.

    Returns
    -------
    file : h5.File
        HDF5 file instance of module h5py, initialized as 'w'. Will be closed 
        once the simulation is finished. Meanwhile, it must exist as a global 
        variable.
    save_slice_stepfun : function
        Stepfunction that will execute the list of functions 'get_slice' on 
        each step and save the results, appending each of them to the 
        associated 'dataname' dataset inside the 'filename' HDF5 file.
        
    See also
    --------
    mp.Simulation
    h5.File

    """
    
    if type(datanames)!=list:
        datanames = [datanames]
    if type(get_slices)!=list:
        get_slices = [get_slices]
    if len(datanames)!=len(get_slices):
        raise ValueError("Must have as many datanames as step functions")

    file = h5.File(filename, 'w', libver='latest')
    
    shapes = []
    dueshapes = []
    for get_slice, dataname in zip(get_slices, datanames):
        
        data = get_slice(sim) # Data zero
        shape = data.shape
        dueshape  = (1,*shape)
        
        file.create_dataset(dataname, chunks=dueshape, 
                            maxshape=tuple(None for d in dueshape), 
                            data=data.reshape(dueshape))
        
        shapes.append(shape)
        dueshapes.append(dueshape)
    
    file.swmr_mode = True
    
    def save_slice_stepfun(sim, state):
        
        if state=="step":
            
            for i in range(len(datanames)):
            
                data_now = get_slices[i](sim)
                
                dim_before = file[datanames[i]].shape[0]
                shape_now = ( dim_before+1, *shapes[i] )
                file[datanames[i]].resize( shape_now )
                
                file[datanames[i]][dim_before,] = data_now.reshape(dueshapes[i])
                
                file[datanames[i]].flush()
                # Notify the reader process that new data has been written
            
        elif state=="finish":
            file.close()
        
        return
            
    return file, save_slice_stepfun

#%%

def retrieve_footer(file, comment_marker='#'):

    """Retrieves the footer of a .txt file saved with np.savetxt.
    
    Parameters
    ----------
    file : str
        File's root (must include directory and termination).
    comment_marker='#' : str, optional
        Sign that indicates a line is a comment on np.savetxt.
    
    Returns
    -------
    last_line : str, dict
        File's footer
    
    Raises
    ------
    ValueError : "Footer not found. Sorry!"
        When the last line doesn't begin with 'comment_marker'.
        
    See Also
    --------
    fwp_save.savetxt
    
    """
    
    
    with open(file, 'r') as f:
        for line in f:
            last_line = line
    
    if last_line[0] == comment_marker:
        try:
            last_line = last_line.split(comment_marker + ' ')[-1]
            last_line = last_line.split('\n')[0]
            footer = vu.string_to_dict(last_line)
        except:
            footer = last_line
        return footer
        
    else:
        raise ValueError("No footer found. Sorry!")

#%%

def retrieve_header(file, comment_marker='#'):

    """Retrieves the header of a .txt file saved with np.savetxt.
    
    Parameters
    ----------
    file : str
        File's root (must include directory and termination).
    comment_marker='#' : str, optional
        Sign that indicates a line is a comment on np.savetxt.
    
    Returns
    -------
    last_line : str, list
        File's header
    
    Raises
    ------
    ValueError : "Header not found. Sorry!"
        When the first line doesn't begin with 'comment_marker'.
    
    See Also
    --------
    fwp_save.savetxt
    
    """
    
    
    with open(file, 'r') as f:
        for line in f:
            first_line = line
            break
    
    if comment_marker in first_line:
        header = first_line.split(comment_marker + ' ')[-1]
        header = header.split('\n')[0]
        header = header.split('\t')
        if len(header) > 1:
            return header
        else:
            split_header = re.split(r'  ', header[0])
            new_header = []
            for h in split_header:
                new_header.append('')
                for s in h:
                    if s!=" ":
                        new_header[-1] += s
            if len(new_header) > 1:
                return new_header
            else:
                return header[0]
        
    else:
        raise ValueError("No header found. Sorry!")

#%%

while True:
    try:
        current_system = setup_system()
        break
    except:
        init_system()

sysname_definition, home_definition, syshome_definition = current_system
