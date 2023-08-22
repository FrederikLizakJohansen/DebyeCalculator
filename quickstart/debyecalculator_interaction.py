import os
import base64
import ipywidgets as widgets
from IPython.display import display, HTML
from ipywidgets import interact, interact_manual
from DebyeCalculator.debye_calculator import DebyeCalculator
import matplotlib.pyplot as plt
import time
import numpy as np
import traitlets
from glob import glob

"""
Interact with the DebyeCalculator to calculate the scattering intensity I(q) through the Debye scattering equation, 
the Total Scattering Structure Function S(q), the Reduced Total Scattering Function F(q), and the Reduced Atomic Pair Distribution Function G(r).

This script demonstrates an interactive interface using ipywidgets and DebyeCalculator for I(Q), S(Q), F(Q), and G(r) based on user-provided parameters.

Usage:
1. Run the script in a Jupyter Notebook environment.
2. The interactive interface will appear with widgets to adjust various parameters.
3. Select a data folder containing XYZ files to analyze.
4. Choose a specific XYZ file from the dropdown menu.
5. Adjust other parameters such as batch size, hardware (CPU or CUDA), radiation type, etc.
6. The graphs showing I(Q), S(Q), F(Q), and G(r) will be displayed and updated as parameters change.

Note: Make sure to have the necessary libraries installed, including ipywidgets, matplotlib, and the DebyeCalculator package.

"""

def interact_debye(
    _cont_updates = False,
):
    radtype = widgets.ToggleButtons(
        options=['Xray', 'Neutron'],
        value='Xray',
        description='Rad. Type:',
        disabled=False,
        layout = widgets.Layout(width='900px'),
        button_style='info'
    )

    path = widgets.Text(
        value='',
        placeholder="some/path/to/data/..",
        description='Data Folder:',
        disabled=False
    )

    batch_size = widgets.IntText(
        min = 100,
        max = 10000,
        value=100,
        description='Batch Size:',
    )

    device = widgets.ToggleButtons(
        options=['CPU', 'CUDA'],
        value='CPU',
        description='Hardware:',
        disabled=False,
        button_style='info',
    )

    lorch_mod = widgets.Checkbox(
        value=False,
        description='Lorch mod.',
        disabled=False
    )

    qslider = widgets.FloatRangeSlider(
        value=[0.8, 25],
        min=0.0,
        max=50.0,
        step=0.01,
        description='Qmin/Qmax:',
        disabled=False,
        continuous_update=_cont_updates,
        orientation='horizontal',
        readout=True,
        #readout_format='.{1}f',
        style={'font_weight':'bold', 'slider_color': 'white'},
        layout = widgets.Layout(width='900px'),
        slider_color = 'white',
    )

    qdamp_slider = widgets.FloatSlider(
        min=0.00,
        max=0.10,
        value=0.04, 
        step=0.01,
        description='Qdamp:',
        layout = widgets.Layout(width='900px'),
        continuous_update=_cont_updates,
    )

    rslider = widgets.FloatRangeSlider(
        value=[0.0, 25],
        min=0,
        max=100.0,
        step=0.1,
        description='rmin/rmax:',
        disabled=False,
        continuous_update=_cont_updates,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style={'font_weight':'bold', 'slider_color': 'white'},
        layout = widgets.Layout(width='900px'),
        slider_color = 'white'
    )

    biso_slider = widgets.FloatSlider(
        min=0.00,
        max=1.00,
        value=0.30,
        step=0.01,
        description='B-iso:',
        layout = widgets.Layout(width='900px'),
        continuous_update=_cont_updates,
    )

    scale_type = widgets.ToggleButtons(
        options=['Normal', 'SAS'],
        value='Normal',
        description='Axes scale:',
        button_style='info'
    )

    # Create a color dropdown widget
    folder = widgets.Text(description='Data Folder:', placeholder='path/to/data/folder')

    # Create a dropdown menu widget for selection of XYZ file and an output area
    standard_msg = 'provide valid folder'
    select_file = widgets.Dropdown(description='Select File:', options=[standard_msg], value=standard_msg, disabled=True)

    # Define a function to update the scattering patterns based on the selected parameters
    def update_options(change):
        folder = change.new
        paths = sorted(glob(os.path.join(folder, '*.xyz')))
        if len(paths):
            select_file.options = paths
            select_file.disabled = False
        else:
            select_file.options = [standard_msg]
            select_file.value = standard_msg
            select_file.disabled = True

    # Link the update function to the dropdown widget's value change event
    folder.observe(update_options, names='value')

    def save_datasets():
        # Save each dataset to a CSV file using numpy
        np.savetxt("iq_data.csv", np.column_stack([q, iq_values]), delimiter=",", header="q,I(Q)", comments='')
        np.savetxt("sq_data.csv", np.column_stack([q, sq_values]), delimiter=",", header="q,S(Q)", comments='')
        np.savetxt("fq_data.csv", np.column_stack([q, fq_values]), delimiter=",", header="q,F(Q)", comments='')
        np.savetxt("gr_data.csv", np.column_stack([r, gr_values]), delimiter=",", header="r,G(r)", comments='')

    def create_download_link(filename, data, header=None):
        content = "\n".join([",".join(map(str, row)) for row in data])
        if header:
            content = header + "\n" + content
        b64 = base64.b64encode(content.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        return href

    def on_download_button_click(button):
        save_datasets()
        
        iq_data = np.column_stack([q, iq_values])
        sq_data = np.column_stack([q, sq_values])
        fq_data = np.column_stack([q, fq_values])
        gr_data = np.column_stack([r, gr_values])

        display(HTML(create_download_link("iq_data.csv", iq_data, "q,I(Q)")))
        display(HTML(create_download_link("sq_data.csv", sq_data, "q,S(Q)")))
        display(HTML(create_download_link("fq_data.csv", fq_data, "q,F(Q)")))
        display(HTML(create_download_link("gr_data.csv", gr_data, "r,G(r)")))

    # Create a function to update the output area
    def update_output(folder, file, batch_size, device, radtype, lorch_mod, qminmax, qdamp, rminmax, biso, scale_type):
        global q, r, iq_values, sq_values, fq_values, gr_values  # Declare these variables as global
        if (file is not None) and file != standard_msg:

            calculator = DebyeCalculator(device=device.lower(), batch_size=batch_size, radiation_type=radtype,
                                         qmin=qminmax[0], qmax=qminmax[1], qstep=0.01, qdamp=qdamp,
                                         rmin=rminmax[0], rmax=rminmax[1], rstep=0.01, rthres=1, biso=biso,
                                         lorch_mod=lorch_mod)

            r, q, iq_values, sq_values, fq_values, gr_values = calculator._return_all(file)

            fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=75)
            axs = axs.flatten()

            if scale_type == 'SAS':
                axs[0].set_xscale('log')
                axs[0].set_yscale('log')

            axs[0].plot(q, iq_values, lw=None)
            axs[0].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$I(Q)$ [counts]')
            axs[1].axhline(1, alpha=0.5, ls='--', c='g')
            axs[1].plot(q, sq_values+1, lw=None)
            axs[1].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$S(Q)$')
            axs[2].axhline(0, alpha=0.5, ls='--', c='g')
            axs[2].plot(q, fq_values, lw=None)
            axs[2].set(xlabel='$Q$ [$\AA^{-1}$]', ylabel='$F(Q)$')
            axs[3].plot(r, gr_values, lw=None)
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

            fig.suptitle("XYZ file: " + file.split('/')[-1].split('.')[0])
            fig.tight_layout()

            download_button = widgets.Button(description="Download Data")
            download_button.on_click(on_download_button_click)
            display(download_button)

    # Create an interactive function that triggers when the user-defined parameters changes
    interact(
        update_output, 
        folder=folder,
        file=select_file,
        batch_size = batch_size,
        device = device,
        radtype = radtype,
        lorch_mod = lorch_mod,
        qminmax = qslider,
        qdamp = qdamp_slider,
        rminmax = rslider,
        biso = biso_slider,
        scale_type=scale_type
    );

    