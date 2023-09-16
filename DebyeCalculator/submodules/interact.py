import ipywidgets as widgets
from ipywidgets import HBox, VBox
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import clear_output
from glob import glob
import os
import threading
import time
import sys
import base64
from IPython.display import display, HTML
from datetime import datetime

from debye_calculator import DebyeCalculator

def run_interact(
    debye_calc: DebyeCalculator,
    _cont_updates: bool = False
):
    qmin = debye_calc.qmin
    qmax = debye_calc.qmax
    qstep = debye_calc.qstep
    qdamp = debye_calc.qdamp
    rmin = debye_calc.rmin
    rmax = debye_calc.rmax
    rstep = debye_calc.rstep
    rthres = debye_calc.rthres
    biso = debye_calc.biso
    device = debye_calc.device
    batch_size = debye_calc.batch_size
    lorch_mod = debye_calc.lorch_mod
    radiation_type = debye_calc.radiation_type
    profile = False
    
    # Buttons
    radtype_button = widgets.ToggleButtons(options=['xray', 'neutron'],
        value=radiation_type,
        description='Rad. type',
        layout = widgets.Layout(width='900px'),
        button_style='info'
    )
    select_radius = widgets.FloatText(
        min = 0,
        max = 50,
        step=0.01,
        value=5,
        description='Radius (.cif):',
        disabled = True,
    )
    device_button = widgets.ToggleButtons(
        options=['cpu', 'cuda'],
        value=device,
        description='Hardware:',
        button_style='info',
    )
    batch_size_button = widgets.IntText(
        min = 100,
        max = 10000,
        value=batch_size,
        description='Batch Size:',
    )
    qslider = widgets.FloatRangeSlider(
        value=[qmin, qmax],
        min=0.0,
        max=50.0,
        step=0.01,
        description='Qmin/Qmax:',
        continuous_update=_cont_updates,
        orientation='horizontal',
        readout=True,
        style={'font_weight':'bold', 'slider_color': 'white'},
        layout = widgets.Layout(width='900px'),
    )
    rslider = widgets.FloatRangeSlider(
        value=[rmin, rmax],
        min=0,
        max=100.0,
        step=rstep,
        description='rmin/rmax:',
        continuous_update=_cont_updates,
        orientation='horizontal',
        readout=True,
        style={'font_weight':'bold', 'slider_color': 'white'},
        layout = widgets.Layout(width='900px'),
    )
    qdamp_slider = widgets.FloatSlider(
        min=0.00,
        max=0.10,
        value=qdamp, 
        step=0.01,
        description='Qdamp:',
        layout = widgets.Layout(width='900px'),
        continuous_update=_cont_updates,
    )
    biso_slider = widgets.FloatSlider(
        min=0.00,
        max=1.00,
        value=biso,
        step=0.01,
        description='B-iso:',
        continuous_update=_cont_updates,
        layout = widgets.Layout(width='900px'),
    )
    qstep_box = widgets.FloatText(
        min = 0.001,
        max = 1,
        step=0.001,
        value=qstep,
        description='Qstep:',
    )
    rstep_box = widgets.FloatText(
        min = 0.001,
        max = 1,
        step=0.001,
        value=rstep,
        description='rstep:',
    )
    rthres_box = widgets.FloatText(
        min = 0.001,
        max = 1,
        step=0.001,
        value=rthres,
        description='rthres:',
    )
    lorch_mod_button = widgets.Checkbox(
        value=lorch_mod,
        description='Lorch mod.:',
    )
    scale_type_button = widgets.ToggleButtons(
        options=['linear', 'logarithmic'],
        value='linear',
        description='Axes scaling:',
        button_style='info'
    )
    
    # Download options
    def create_download_link(filename_prefix, data, header=None):
    
        # Collect Metadata
        metadata ={
            'qmin': qslider.value[0],
            'qmax': qslider.value[1],
            'qdamp': qdamp_slider.value,
            'qstep': qstep_box.value,
            'rmin': rslider.value[0], 
            'rmax': rslider.value[1],
            'rstep': rstep_box.value, 
            'rthres': rthres_box.value,
            'biso': biso_slider.value,
            'device': device_button.value,
            'batch_size': batch_size_button.value, 
            'lorch_mod': lorch_mod_button.value,
            'radiation_type': radtype_button.value
        }

        # Join content
        output = ''
        content = "\n".join([",".join(map(str, np.around(row,len(str(qstep_box.value))))) for row in data])
        for k,v in metadata.items():
            output += f'{k}:{v}\n'
        output += '\n'
        if header:
            output += header + '\n'
        output += content
    
        # Encode as base64
        b64 = base64.b64encode(output.encode()).decode()
    
        # Add Time
        t = datetime.now()
        year = f'{t.year}'[-2:]
        month = f'{t.month}'.zfill(2)
        day = f'{t.day}'.zfill(2)
        hours = f'{t.hour}'.zfill(2)
        minutes = f'{t.minute}'.zfill(2)
        seconds = f'{t.second}'.zfill(2)
        
        # Make filename
        filename = filename_prefix + '_' + select_file.value.split('/')[-1].split('.')[0] + '_' + month + day + year + '_' + hours + minutes + seconds + '.csv'
    
        # Make href and return
        href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href
    
    def create_structure_download_link(filename_prefix, ase_atoms):
        
        # Get atomic properties
        positions = ase_atoms.get_positions()
        elements = ase_atoms.get_chemical_symbols()
        num_atoms = len(ase_atoms)
    
        # Make header
        header = str(num_atoms) + "\n\n"
    
        # Join content 
        content = header + "\n".join([el + '\t' + "\t".join(map(str,np.around(row, 3))) for row, el in zip(positions, elements)])
        
        # Encode as base64
        b64 = base64.b64encode(content.encode()).decode()
        
        # Add Time
        t = datetime.now()
        year = f'{t.year}'[-2:]
        month = f'{t.month}'.zfill(2)
        day = f'{t.day}'.zfill(2)
        hours = f'{t.hour}'.zfill(2)
        minutes = f'{t.minute}'.zfill(2)
        seconds = f'{t.second}'.zfill(2)
    
        # Make ilename
        filename = filename_prefix + '_' + select_file.value.split('/')[-1].split('.')[0] + str(select_radius.value) + '_' + month + day + year + '_' + hours + minutes + seconds + '.xyz'
    
        # Make href and return
        href = f'<a href="data:text/xyz;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href
    
    # Download buttons
    download_button = widgets.Button(description="Download Data")
    
    @download_button.on_click
    def on_download_button_click(button):
        # Try to compile all the data and create html link to download files
        try:
            # Data
            iq_data = np.column_stack([q, iq])
            sq_data = np.column_stack([q, sq])
            fq_data = np.column_stack([q, fq])
            gr_data = np.column_stack([r, gr])
        
            # Clear warning message
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
    
            # Display download links
            display(HTML(create_download_link('iq', iq_data, "q,I(Q)")))
            display(HTML(create_download_link('sq', sq_data, "q,S(Q)")))
            display(HTML(create_download_link('fq', fq_data, "q,F(Q)")))
            display(HTML(create_download_link('gr', gr_data, "r,G(r)")))
    
            if not select_radius.disabled:
                ase_atoms, _ = DebyeCalculator().generate_nanoparticles(select_file.value, select_radius.value)
                
                display(HTML(create_structure_download_link('structure', ase_atoms[0])))
    
        except Exception as e:
            raise(e)
            print('FAILED: No data has been selected', end="\r")
                  
    # Folder dropdown widget
    folder = widgets.Text(description='Data Dir.:', placeholder='Provide data directory', disabled=False)
    
    # Create a dropdown menu widget for selection of XYZ file and an output area
    DEFAULT_MSG = ''
    select_file = widgets.Dropdown(description='Select File:', options=[DEFAULT_MSG], value=DEFAULT_MSG, disabled=True)
    
    # Define a function to update the scattering patterns based on the selected parameters
    def update_options(change):
        folder = change.new
        paths = sorted(glob(os.path.join(folder, '*.xyz')) + glob(os.path.join(folder, '*.cif')))
        if len(paths):
            select_file.options = ['Select data file'] + paths
            select_file.value = 'Select data file'
            select_file.disabled = False
        else:
            select_file.options = [DEFAULT_MSG]
            select_file.value = DEFAULT_MSG
            select_file.disabled = True
    
    # Link the update function to the dropdown widget's value change event
    folder.observe(update_options, names='value')
    
    def update_options_radius(change):
        #select_radius = change.new
        selected_ext = select_file.value.split('.')[-1]
        if selected_ext == 'xyz':
            select_radius.disabled = True
        elif selected_ext == 'cif':
            select_radius.disabled = False
    
    select_file.observe(update_options_radius, names='value')
    
    plot_button = widgets.Button(
        description='Plot',
    )
    
    def update_figure(r, q, iq, sq, fq, gr, _unity_sq=True):
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=75)
        axs = axs.flatten()
    
        if scale_type_button.value == 'logarithmic':
            axs[0].set_xscale('log')
            axs[0].set_yscale('log')
    
        axs[0].plot(q, iq)
        axs[0].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$I(Q)$ [counts]')
        
        axs[1].axhline(1, alpha=0.5, ls='--', c='g')
        axs[1].plot(q, sq+int(_unity_sq))
        axs[1].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$S(Q)$')
        
        axs[2].axhline(0, alpha=0.5, ls='--', c='g')
        axs[2].plot(q, fq)
        axs[2].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$F(Q)$')
        
        axs[3].plot(r, gr)
        axs[3].set(xlabel='$r$ [$\AA$]', ylabel='$G_r(r)$')
    
        labels = ['Scattering Intensity, I(Q)',
                  'Structure Function, S(Q)',
                  'Reduced Structure Function, F(Q)',
                  'Reduced Pair Distribution Function, G(r)']
    
        for ax, label in zip(axs, labels):
            ax.relim()
            ax.autoscale_view()
            ax.set_title(label)
            ax.grid(alpha=0.2)
    
        fig.suptitle("XYZ file: " + select_file.value.split('/')[-1].split('.')[0])
        fig.tight_layout()
    
    @plot_button.on_click
    def update_parameters(b=None):
    
        clear_output(wait=True)
        display_tabs()
    
        global r, q, iq, sq, fq, gr
    
        try:
            path_ext = select_file.value.split('.')[-1]
        except Exception as e:
            return
    
        if (select_file.value is not None) and select_file.value != DEFAULT_MSG and path_ext in ['xyz', 'cif']:
            try:
                debye_calc = DebyeCalculator(
                    device=device_button.value, 
                    batch_size=batch_size_button.value,
                    radiation_type=radtype_button.value,
                    qmin=qslider.value[0], 
                    qmax=qslider.value[1], 
                    qstep=qstep_box.value, 
                    qdamp=qdamp_slider.value,
                    rmin=rslider.value[0],
                    rmax=rslider.value[1], 
                    rstep=rstep_box.value, 
                    rthres=rthres_box.value, 
                    biso=biso_slider.value,
                    lorch_mod=lorch_mod_button.value
                )
                if not select_radius.disabled and select_radius.value > 8:
                    print('Generating...')
                r, q, iq, sq, fq, gr = debye_calc._get_all(select_file.value, select_radius.value)
                
                clear_output(wait=True)
                display_tabs()
                update_figure(r, q, iq, sq, fq, gr)
                
            except Exception as e:
                raise(e)
                print(f'FAILED: Could not load data file: {path}', end='\r')
                
    # Make File Tab
    file_tab = VBox(children = [
        folder,
        select_file,
    ])

    # Make Generate Tab
    generate_tab = VBox(children = [
        folder,
        select_file,
        select_radius,
    ])
    
    # Make Scattering Tab
    scattering_tab = VBox(children = [
        radtype_button,
        HBox(children=[qslider, qstep_box]),
        HBox(children=[rslider, rstep_box]),
        HBox(children=[qdamp_slider, rthres_box]),
        HBox(children=[biso_slider, lorch_mod_button]),
    ])

    # Make Plotting Tab
    plotting_tab = VBox(children = [
        scale_type_button,
    ])

    # Make Hardware Tab
    hardware_tab = VBox(children = [
        device_button,
        batch_size_button,
    ])

    # Display Tabs
    tabs = widgets.Tab(children=[
        file_tab,
        generate_tab,
        scattering_tab,
        plotting_tab,
        hardware_tab,
    ])

    tabs.set_title(0, 'XYZ File Select')
    tabs.set_title(1, 'CIF Nanoparticle Generation')
    tabs.set_title(2, 'Scattering Parameters')
    tabs.set_title(3, 'Plotting Options')
    tabs.set_title(4, 'Hardware Options')

    def display_tabs():
        display(VBox(children=[tabs, HBox(children=[plot_button, download_button])]))
    display_tabs()
