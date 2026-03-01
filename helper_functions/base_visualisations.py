#plot_firing_rates(**locals())
import matplotlib.pyplot as plt
import numpy as np



def visualise_inputs(seq = None, day_timesteps = None, nstep = None, **kwargs):
    #Visualising the input sequence
    time_points = np.arange(0, nstep, 1)
    tmp_sig = np.zeros(nstep)
    for i in range(len(seq)-1):
        if i % 2 == 0:
            tmp_sig[seq[i]:seq[i+1]] = 1
    plt.figure(figsize=(15, 4))
    plt.plot(time_points, tmp_sig, color='black')
    plt.vlines(day_timesteps, ymin=0, ymax=1, color='red', linestyle='--', label='Day Boundaries')
    plt.title("Visualisation of Input Sequence")
    plt.xlabel("Time (ms)")
    plt.ylabel("Input Strength")
    plt.show()

def visualise_excitability(Emat = None, exc0 = None, test= None, **kwargs):
    plt.figure()
    plt.imshow(Emat, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Excitability Matrix Over Time')
    plt.show()
    plt.plot(exc0, 'r--', label='Baseline Excitability')
    plt.title('Baseline Excitability per Neuron')
    plt.show()

def visualise_firing_rates(nstep = None, r = None, N = None, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    ax.set_ylabel('Neurons')
    ax.set_xlabel('Time (ms)')
    im = ax.imshow(np.maximum(0,r),aspect = nstep/N*2/9,cmap = 'cividis')
    cbar = fig.colorbar(im, ax=ax, location='left', pad=0.02)
    cbar.set_label('Firing rate (a.u)')
    plt.show()

def visualise_transient_firing_rates(r_1, r_2, nstep = None, r = None, N = None, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')

    ax.set_ylabel('Firing rate (a.u)')
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(0, nstep)

    for n in range(N):
        ax.plot(r[n,:],'k',linewidth = 1)
    for n in range(r_1,r_2):
        ax.plot(r[n,:],'r',linewidth = 1)
    #ax.plot(r[19,:],'k',linewidth = 1)

    plt.show()

def visualise_weight_matrix(Nevent = None, day_timesteps = None, y = None, index = None, N = None, **kwargs):
    fig, ax = plt.subplots(1, Nevent, figsize=(Nevent * 3.5, 3.5), sharey=True, layout='constrained')
    fig.suptitle('Network weights at the end of each day', fontsize=16)

    # ax.set_ylabel('Firing rate (a.u)')
    # ax.set_xlabel('Time (ms)')
    # ax.set_xlim(0, nstep)

    if Nevent == 1:
            ax = [ax]
    for i, ax in enumerate(ax):
            ax.set_title('Day ' + str(i+1) + ' (' + str(day_timesteps[i+1]) + ' ms)')
            im = ax.imshow(y[index[1][:],day_timesteps[i+1]].reshape((N,N)), vmin = 0, vmax = 1, cmap= 'binary')

    cbar = fig.colorbar(im, ax=ax, location='right', pad=0.05)
    plt.show()

def dynamic_visualiser(nstep = None, seq = None, y = None, index = None, Emat = None, N = None, **kwargs):
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import numpy as np
    # --- 1. Pre-calculate Data traces ---
    time_points = np.arange(0, nstep, 1)
    input_trace = np.zeros(nstep)
    for i in range(0, len(seq)-1, 2):
        if seq[i] < nstep and seq[i+1] <= nstep:
            input_trace[seq[i]:seq[i+1]] = 1.0

    # Extract Firing Rate (r)
    r_data = y[index[0], :] 

    # --- 2. Setup Widgets ---
    time_slider = widgets.IntSlider(
        value=0, 
        min=0, 
        max=y.shape[1]-1, 
        step=100, 
        description='Time (ms):',
        layout=widgets.Layout(width='800px')
    )

    plot_output = widgets.Output()

    display(time_slider, plot_output)

    # --- 3. The Update Logic ---
    def on_value_change(change):
        t = change['new']
        
        with plot_output:
            clear_output(wait=True)
            plt.close('all')
            
            # --- LAYOUT SETUP ---
            fig = plt.figure(figsize=(16, 8), constrained_layout=True)
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.5])
            
            # Sub-grids
            gs_left = gs[0].subgridspec(2, 1)
            gs_right = gs[1].subgridspec(3, 1, height_ratios=[1, 3, 3], hspace=0.05)
            
            # ================= LEFT COLUMN =================
            
            # 1. Synaptic Weights (Top Left)
            ax_w = fig.add_subplot(gs_left[0])
            W_t = y[index[1][:], t].reshape((N, N))
            im_w = ax_w.imshow(W_t, vmin=0, vmax=1, cmap='binary')
            ax_w.set_title(f'Synaptic Weights at t = {t} ms')
            ax_w.set_ylabel('Pre-synaptic')
            
            # 2. Current Excitability (Bottom Left) - THE NEW PLOT
            ax_exc_now = fig.add_subplot(gs_left[1])
            
            # Get current excitability vector (Size N)
            current_E = Emat[:, t]
            
            # Tile it to make a square matrix (Size N x N)
            # This creates vertical stripes corresponding to neuron indices
            E_visual = np.tile(current_E, (N, 1))
            
            # Plot
            im_exc_now = ax_exc_now.imshow(E_visual, vmin=0, vmax=np.max(Emat), cmap='Purples')
            ax_exc_now.set_title('Current Excitability Bias')
            ax_exc_now.set_ylabel('(Repeated for Visual)')
            ax_exc_now.set_xlabel('Neuron Index')
            
            # Add colorbar
            cbar_en = fig.colorbar(im_exc_now, ax=ax_exc_now, location='right', pad=0.05)
            cbar_en.set_label('Excitability', rotation=270, labelpad=15)

            # ================= RIGHT COLUMN =================
            
            # 3. Input Sequence (Top Right)
            ax_in = fig.add_subplot(gs_right[0])
            ax_in.plot(time_points, input_trace, color='black', lw=1.5)
            ax_in.axvline(x=t, color='red', linewidth=2, linestyle='-', alpha=0.8)
            ax_in.set_title('Network Dynamics over Time', fontsize=12)
            ax_in.set_ylabel('Input')
            ax_in.set_xlim(0, nstep)
            ax_in.set_yticks([])
            ax_in.set_xticklabels([])
            
            # 4. Firing Rates (Middle Right)
            ax_fire = fig.add_subplot(gs_right[1], sharex=ax_in)
            im_fire = ax_fire.imshow(np.maximum(0, r_data), aspect='auto', cmap='cividis', 
                                    interpolation='nearest', origin='upper')
            ax_fire.axvline(x=t, color='red', linewidth=2, linestyle='-', alpha=0.8)
            ax_fire.set_ylabel('Neuron ID')
            plt.setp(ax_fire.get_xticklabels(), visible=False)
            
            # 5. Excitability History (Bottom Right)
            ax_exc = fig.add_subplot(gs_right[2], sharex=ax_in)
            im_exc = ax_exc.imshow(Emat, aspect='auto', cmap='viridis', 
                                interpolation='nearest', origin='upper')
            ax_exc.axvline(x=t, color='red', linewidth=2, linestyle='-', alpha=0.8)
            ax_exc.set_ylabel('Neuron ID')
            ax_exc.set_xlabel('Time (ms)')

            plt.show()

    # --- 4. Connect & Run ---
    time_slider.observe(on_value_change, names='value')
    on_value_change({'new': time_slider.value})

def visualise_correlation_change_over_day(x_days_apart, threshold, N=None, Nevent=None, day_timesteps=None, y=None, index=None, **kwargs):
    total_days = Nevent
    all_delta_c = []
    
    # Iterate through all possible session pairs separated by x_days_apart [cite: 620, 621]
    for start_day in range(0, (total_days - x_days_apart)):
        # Extract intra-day firing rates based on session windows [cite: 127]
        rates_day_1 = y[index[0], day_timesteps[start_day]:day_timesteps[start_day+1]]
        rates_day_2 = y[index[0], day_timesteps[start_day + x_days_apart]:day_timesteps[start_day + x_days_apart + 1]]

        # Calculate intra-day Pearson correlation matrices [cite: 619]
        correlation_day_1 = np.corrcoef(rates_day_1)
        correlation_day_2 = np.corrcoef(rates_day_2)

        # Extract unique pairs (upper triangle) to avoid self-correlation and duplicates [cite: 181]
        upper_tri = np.triu_indices(N, k=1)
        c1_values = correlation_day_1[upper_tri]
        c2_values = correlation_day_2[upper_tri]

        # Apply C_min threshold to identify stable/active neuron pairs [cite: 317, 624]
        mask = c2_values >= threshold
        
        if np.any(mask):
            c1_filtered = c1_values[mask]
            c2_filtered = c2_values[mask]

            # Calculate Scaled Correlation Change (Delta C) [cite: 316]
            delta_c = (c2_filtered - c1_filtered) / c2_filtered
            all_delta_c.append(delta_c)
    
    # 1. FIX: Concatenate list of arrays into a single dataset [cite: 486, 623]
    if len(all_delta_c) == 0:
        print("No pairs met the threshold.")
        return
    aggregated_data = np.concatenate(all_delta_c)

    plt.figure(figsize=(8, 5))
    
    # 2. Use density=True for probability density and raw strings for LaTeX titles [cite: 608, 750]
    plt.hist(aggregated_data, bins=300, density=True, color='skyblue', edgecolor='black', alpha=0.7)

    # 3. Highlight the origin to separate noise from nonstationary drift [cite: 321, 475]
    plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Stable/Noise Peak')

    # Formatting using raw strings (r'') to avoid SyntaxWarnings with backslashes
    plt.title(r'Distribution of Scaled Correlation Changes ($\Delta C$)' + f'\n{x_days_apart} Days Apart')
    plt.xlabel(r'Scaled Correlation Change ($\Delta C$)')
    plt.ylabel('Probability Density')
    plt.xlim(-0.5, 1.5) # Standard range for drift characterization [cite: 381, 386]
    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    #plt.show()
    return aggregated_data

        
    
    