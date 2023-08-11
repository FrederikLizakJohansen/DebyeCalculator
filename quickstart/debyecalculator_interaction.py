import os
os.chdir('../packages')

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interact_manual
from debye_calculator import DebyeCalculator
import matplotlib.pyplot as plt
import time
import numpy as np
import traitlets
from glob import glob

"""
Interact with the DebyeCalculator to visualize scattering intensity and distribution functions.

This script demonstrates an interactive interface using ipywidgets and DebyeCalculator for visualizing
scattering intensity and distribution functions (S(Q), F(Q), G(r)) based on user-provided parameters.

Usage:
1. Run the script in a Jupyter Notebook environment.
2. The interactive interface will appear with widgets to adjust various parameters.
3. Select a data folder containing XYZ files to analyze.
4. Choose a specific XYZ file from the dropdown menu.
5. Adjust other parameters such as batch size, hardware (CPU or CUDA), radiation type, etc.
6. The graphs showing scattering intensity and distribution functions will be displayed and updated as parameters change.

Note: Make sure to have the necessary libraries installed, including ipywidgets, matplotlib, and the DebyeCalculator package.

"""

def interact_debye(
    _cont_updates = False,
    _step_size = 0.1
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
        step=_step_size,
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
        min=0.0,
        max=1.0,
        value=0.0, 
        step=_step_size,
        description='Qdamp:',
        layout = widgets.Layout(width='900px'),
        continuous_update=_cont_updates,
    )

    rslider = widgets.FloatRangeSlider(
        value=[0.0, 25],
        min=0,
        max=100.0,
        step=_step_size,
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
        min=0.0,
        max=10.0,
        value=0.1,
        step=_step_size,
        description='B-iso:',
        layout = widgets.Layout(width='900px'),
        continuous_update=_cont_updates,
    )

    # Create a color dropdown widget
    folder = widgets.Text(description='Data Folder:', placeholder='path/to/data/folder')

    # Create a dropdown menu widget for fruits and an output area
    standard_msg = 'provide valid folder'
    select_file = widgets.Dropdown(description='Select File:', options=[standard_msg], value=standard_msg, disabled=True)

    # Define a function to update the fruit options based on the selected color
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

    # Link the update function to the color_dropdown widget's value change event
    folder.observe(update_options, names='value')

    # Create a function to update the output area
    def update_output(folder, file, batch_size, device, radtype, lorch_mod, qminmax, qdamp, rminmax, biso):
        if (file is not None) and file != standard_msg:

            calculator = DebyeCalculator(device=device.lower(), batch_size=batch_size, radiation_type=radtype,
                                         qmin=qminmax[0], qmax=qminmax[1], qstep=0.01, qdamp=qdamp,
                                         rmin=rminmax[0], rmax=rminmax[1], rstep=0.01, rthres=1, biso=biso,
                                         lorch_mod=lorch_mod)

            r, q, iq_values, sq_values, fq_values, gr_values = calculator._return_all(file)

            fig, axs = plt.subplots(2, 2, figsize=(12, 8), dpi=75)
            axs = axs.flatten()

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

            fig.suptitle(file.split('/')[-1].split('.')[0])
            fig.tight_layout()


    # Create an interactive function that triggers when the fruit selection changes
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
    );