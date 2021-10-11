"""
Plot of a single spherical NP cell under incidence of a planewave source.

See also
--------
Routines/monoch_field
Routines/np_monoch_field
"""

import matplotlib.pyplot as plt
import numpy as np
import v_meep as vm
import v_plot as vp
import v_utilities as vu

vp.set_style()

#%% PARAMETERS

"""
series = "DTest532"
folder = "Field/NPMonoch/AuSphere/VacWatTest/DefinitiveTest/Vacuum"

with_line = True
with_plane = True
with_box = False
with_nanoparticle = True

english = False
"""

#%%

def plot_np_planewave_cell(params, series, folder, 
                           with_line=False, with_plane=False,
                           with_box=False, with_nanoparticle=False, 
                           english=False):
    
    #%% SETUP
    
    # Saving directories
    sa = vm.SavingAssistant(series, folder)
    
    trs = vu.BilingualManager(english=english)
    
    """
    import h5py as h5
    
    f = h5.File(sa.file("Field-Lines.h5"), "r+")
    params = dict(f["Ez"].attrs)
    """
    
    """
    import v_save as vs
    
    params = vs.retrieve_footer(sa.file("Results.txt"))
    """
    
    #%% DATA EXTRACTION
    
    from_um_factor = params["from_um_factor"]
    try:
        units = params["units"]
    except:
        units = True
    
    cell_width = params["cell_width"]
    pml_width = params["pml_width"]
    
    try:
        r = params["r"]
        material = params["material"]
    except:
        r = 0
        material = "none"
        with_nanoparticle = False
    
    try:
        wlen = params["wlen"]
    except:
        wlen_range = params["wlen_range"]
        wlen_center = np.mean(wlen_range)
        wlen_width = float(np.diff(wlen_range))
    source_center = params["source_center"]
    
    submerged_index = params["submerged_index"]
    surface_index = params["surface_index"]
    try:
        overlap = params["overlap"]
    except:
        overlap = 0

    try:        
        flux_box_size = params["flux_box_size"]
    except:
        flux_box_size = 0
        with_box = False
    
    #%% PLOT
    
    fig, ax = plt.subplots()
    ax.grid(False)
    
    # PML borders
    pml_out_square = plt.Rectangle((-cell_width/2, -cell_width/2), 
                                   cell_width, cell_width,
                                   fill=False, edgecolor="m", linestyle="dashed",
                                   hatch='/', 
                                   zorder=-20,
                                   label=trs.choose("PML borders", "Bordes PML"))
    pml_inn_square = plt.Rectangle((-cell_width/2+pml_width,
                                    -cell_width/2+pml_width), 
                                   cell_width - 2*pml_width, cell_width - 2*pml_width,
                                   facecolor="white", edgecolor="m", 
                                   linestyle="dashed", linewidth=1, zorder=-10)
   
    # Surrounding medium
    if submerged_index != 1:
        surrounding_square = plt.Rectangle((-cell_width/2, -cell_width/2),
                                           cell_width, cell_width,
                                           color="blue", alpha=.1, zorder=-6,
                                           label=trs.choose(fr"Medium $n$={submerged_index}",
                                                            fr"Medio $n$={submerged_index}"))
        
    # Surface medium
    if surface_index != submerged_index:
        surface_square = plt.Rectangle((r - overlap, -cell_width/2),
                                       cell_width/2 - r + overlap, 
                                       cell_width,
                                       edgecolor="navy", hatch=r"\\", 
                                       fill=False, zorder=-3,
                                       label=trs.choose(fr"Surface $n$={surface_index}",
                                                        fr"Superficie $n$={surface_index}"))
    # Nanoparticle
    if with_nanoparticle:
        if material=="Au":
            circle_color = "gold"
        elif material=="Ag":
            circle_color="darkgrey"
        else:
            circle_color="peru"
        circle = plt.Circle((0,0), r, color=circle_color, linewidth=1, alpha=.4, 
                            zorder=0, label=trs.choose(f"{material} Nanoparticle",
                                                          f"Nanopartícula de {material}"))
    
    # Source
    ax.vlines(source_center, -cell_width/2, cell_width/2,
              color="r", linestyle="dashed", zorder=5, 
              label=trs.choose("Planewave Source", "Fuente de ondas plana"))
    
    # Sampling line
    if with_line:
        ax.hlines(0, -cell_width/2, cell_width/2,
                  color="blue", linestyle=":", zorder=7, # limegreen
                  label=trs.choose("Sampling Line", "Línea de muestreo"))
    
    # Sampling plane
    if with_plane:
        ax.vlines(0, -cell_width/2, cell_width/2,
                  color="blue", linestyle="dashed", zorder=7, 
                  label=trs.choose("Sampling Plane", "Plano de muestreo"))
        
    # Flux box
    if with_box:
        flux_square = plt.Rectangle((-flux_box_size/2, -flux_box_size/2), 
                                    flux_box_size, flux_box_size,
                                    linewidth=1, edgecolor="limegreen", linestyle="dashed",
                                    fill=False, zorder=10, 
                                    label=trs.choose("Flux box", "Caja de flujo"))
    
    if with_nanoparticle: ax.add_patch(circle)
    if submerged_index!=1: ax.add_patch(surrounding_square)
    if surface_index!=submerged_index and surface_index!=1: ax.add_patch(surface_square)
    if with_box: ax.add_patch(flux_square)
    ax.add_patch(pml_out_square)
    ax.add_patch(pml_inn_square)
    
    # General configuration
    box = ax.get_position()
    box.x0 = box.x0 - .26 * (box.x1 - box.x0)
    # box.x1 = box.x1 - .05 * (box.x1 - box.x0)
    box.y1 = box.y1 + .10 * (box.y1 - box.y0)
    ax.set_position(box)           
    plt.legend(bbox_to_anchor=trs.choose( (1.53, 0.5), (1.60, 0.5) ), 
               loc="center right", frameon=False)
    
    fig.set_size_inches(7.5, 4.8)
    ax.set_aspect("equal")
    plt.xlim(-cell_width/2, cell_width/2)
    plt.ylim(-cell_width/2, cell_width/2)
    plt.xlabel(trs.choose(r"Position $X$ [MPu]", r"Posición $X$ [uMP]"))
    plt.ylabel(trs.choose(r"Position $Z$ [MPu]", r"Posición $Z$ [uMP]"))
    
    if units:
        plt.annotate(trs.choose(f"1 MPu = {from_um_factor * 1e3:.0f} nm",
                                f"1 uMP = {from_um_factor * 1e3:.0f} nm"),
                     (5, 5), xycoords='figure points')
        try:
            plt.annotate(fr"$\lambda$ = {wlen * from_um_factor * 1e3:.0f} nm",
                         (350, 5), xycoords='figure points', color="r")
        except:
            plt.annotate(fr"$\lambda_0$ = {wlen_center * from_um_factor * 1e3:.0f} nm" + 
                         ", " +
                         fr"$\Delta\lambda$ = {wlen_width * from_um_factor * 1e3:.0f} nm",
                         (345, 5), xycoords='figure points', color="r")
    else:
        plt.annotate(trs.choose(r"1 MPu = $\lambda$",
                                r"1 uMP = $\lambda$"),
                     (5, 5), xycoords='figure points')
        try:
            plt.annotate(r"$\lambda$ = 1 " + trs.choose("MPu", "uMP"),
                         (350, 5), xycoords='figure points', color="r")
        except:
            plt.annotate(r"$\lambda_0$ = 1 " + trs.choose("MPu", "uMP") + 
                         ", " +
                         fr"$\Delta\lambda$ = {wlen_width}" + trs.choose("MPu", "uMP"),
                         (345, 5), xycoords='figure points', color="r")
    
    if with_nanoparticle or material=="none":
        plt.savefig(sa.file("SimBox.png"))
    else:
        plt.savefig(sa.file("SimBoxNorm.png"))
            