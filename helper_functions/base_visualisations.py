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

        
    
def readout_visualiser(nstep=None, y=None, index=None, N=None, **kwargs):
    
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import numpy as np
    # --- 1. Pre-calculate Data traces ---
    time_points = np.arange(0, nstep, 1)
    r_data = y[index[0], :]           # RNN Firing Rates [cite: 338]
    w_rec_data = y[index[1], :]       # Recurrent Weights [cite: 323]
    w_out_data = y[index[4], :]       # Readout Weights [cite: 391]
    
    # Calculate Readout Firing Rate (y = sum(W_out * r)) [cite: 399]
    # We apply the threshold logic here for the visualization [cite: 342]
    readout_theta = kwargs.get('readout_theta', 0.0) 
    readout_rate_trace = np.maximum(0, np.sum(w_out_data * r_data, axis=0) - readout_theta)

    # --- 2. Setup Widgets ---
    time_slider = widgets.IntSlider(
        value=0, min=0, max=nstep-1, step=100, 
        description='Time (ms):', layout=widgets.Layout(width='800px')
    )
    plot_output = widgets.Output()
    display(time_slider, plot_output)

    # --- 3. The Update Logic ---
    def on_value_change(change):
        t = change['new']
        with plot_output:
            clear_output(wait=True)
            plt.close('all')
            
            fig = plt.figure(figsize=(16, 8), constrained_layout=True)
            gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2])
            
            # ================= LEFT: DETAILED NETWORK SCHEMATIC =================
            ax_net = fig.add_subplot(gs[0])
            ax_net.set_title(f"Network & Readout State (t={t})", fontsize=14)
            
            # RNN Layout (Circular)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            rnn_x, rnn_y = np.cos(angles), np.sin(angles)
            read_x, read_y = 2.8, 0 # Position of readout neuron
            
            # Current states
            curr_r = r_data[:, t]
            curr_w_rec = w_rec_data[:, t].reshape((N, N))
            curr_w_out = w_out_data[:, t]
            curr_readout_r = readout_rate_trace[t]

            # A. Draw Recurrent Weights (RNN -> RNN) [cite: 323, 324]
            # We use a power-law alpha to make high weights pop and low weights vanish
            max_w_rec = np.max(w_rec_data) if np.max(w_rec_data) > 0 else 1
            visibility_threshold = 0.8  # Ignore weights below 20% of max 
            
            for i in range(N):
                for j in range(N):
                    if i != j:
                        weight_ratio = curr_w_rec[i, j] / max_w_rec
                        if weight_ratio > visibility_threshold:
                            # Use weight_ratio^2 or ^3 to aggressively hide lower values
                            alpha = np.clip(weight_ratio**2, 0, 0.4) 
                            ax_net.plot([rnn_x[j], rnn_x[i]], [rnn_y[j], rnn_y[i]], 
                                        color='gray', alpha=alpha, lw=0.8, zorder=1)

            # B. Draw Readout Weights (RNN -> Readout) [cite: 391, 399]
            max_w_out = np.max(w_out_data) if np.max(w_out_data) > 0 else 1
            for i in range(N):
                alpha = np.clip(curr_w_out[i] / max_w_out, 0, 1)
                ax_net.plot([rnn_x[i], read_x], [rnn_y[i], read_y], 
                            color='red', alpha=alpha, lw=1.5, zorder=2)

            # C. Plot RNN Neurons [cite: 65, 89]
            sc_rnn = ax_net.scatter(rnn_x, rnn_y, c=curr_r, s=150, cmap='cividis', 
                                    edgecolors='black', zorder=4, vmin=0, vmax=10)
            
            # D. Plot Readout Neuron [cite: 263]
            # Color is tied to its own firing rate
            sc_read = ax_net.scatter([read_x], [read_y], c=[curr_readout_r], s=400, 
                                     cmap='Reds', edgecolors='black', marker='D', 
                                     zorder=5, vmin=0, vmax=np.max(readout_rate_trace)+1)
            
            ax_net.set_xlim(-1.5, 3.5); ax_net.set_ylim(-1.5, 1.5)
            ax_net.axis('off')
            
            # Colorbars
            cb1 = plt.colorbar(sc_rnn, ax=ax_net, label='RNN Rate', location='bottom', pad=0.05, shrink=0.5)
            cb2 = plt.colorbar(sc_read, ax=ax_net, label='Readout Rate', location='right', pad=0.05, shrink=0.5)

            # ================= RIGHT: TIME SERIES =================
            gs_right = gs[1].subgridspec(2, 1, hspace=0.1)
            
            # 1. Readout Neuron Activity Trace [cite: 222, 264]
            ax_y = fig.add_subplot(gs_right[0])
            ax_y.plot(time_points, readout_rate_trace, color='red', lw=2)
            ax_y.fill_between(time_points, readout_rate_trace, color='red', alpha=0.1)
            ax_y.axvline(x=t, color='black', linestyle='--')
            ax_y.set_ylabel('Readout Rate ($y$)')
            ax_y.set_xlim(0, nstep)

            # 2. Output Weight Evolution [cite: 238, 266]
            ax_w_heat = fig.add_subplot(gs_right[1], sharex=ax_y)
            im_w = ax_w_heat.imshow(w_out_data, aspect='auto', cmap='binary', origin='upper')
            ax_w_heat.axvline(x=t, color='red', linewidth=2)
            ax_w_heat.set_ylabel('RNN Neuron Index')
            ax_w_heat.set_xlabel('Time (ms)')

            plt.show()

    time_slider.observe(on_value_change, names='value')
    on_value_change({'new': time_slider.value})

def visualise_input_sequence(nstep = None, INPUT=None, N = None, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np

    time_points = np.arange(0, nstep, 1)
    # Traces to store the amplitude for each context
    trace_a = np.zeros(nstep)
    trace_b = np.zeros(nstep)

    # Pre-calculate the traces using your INPUT function logic
    for t in range(nstep):
        inp = INPUT(t)
        # Check if any neuron in the first half is active
        if np.any(inp[0:int(N/2)] > 0):
            trace_a[t] = np.mean(inp[0:int(N/2)])
        # Check if any neuron in the second half is active
        if np.any(inp[int(N/2):N] > 0):
            trace_b[t] = np.mean(inp[int(N/2):N])

    plt.figure(figsize=(15, 4))
    
    # Fill areas to make it look like the eLife paper's stimulus bars
    plt.fill_between(time_points, trace_a, color='skyblue', alpha=0.8, label='Context A (Neurons 0-24)')
    plt.fill_between(time_points, trace_b, color='salmon', alpha=0.8, label='Context B (Neurons 25-49)')
    
    plt.title("Sequential Memory Stimulation Protocol", fontsize=14)
    plt.xlabel("Time (ms)", fontsize=12)
    plt.ylabel("Input Amplitude (IN)", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show() 

def tworeadout_visualiser(INPUT=None, nstep=None, y=None, index=None, N=None, **kwargs):
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import numpy as np

    # --- FIX: Prevent multiple widget instances ---
    if 'readout_box' in globals():
        globals()['readout_box'].close()

    # --- 1. Pre-calculate Data traces ---
    time_points = np.arange(0, nstep, 1)
    r_data = y[index[0], :]           # RNN Firing Rates
    w_rec_data = y[index[1], :]       # Recurrent Weights
    
    # Extract weights for both readouts
    # Assuming index[4] is Readout 1 and index[5] is Readout 2
    w_out1_data = y[index[4], :]      
    w_out2_data = y[index[5], :]      
    
    # Calculate Firing Rates for both
    readout_theta = kwargs.get('readout_theta', 0.1) 
    y1_trace = np.maximum(0, np.sum(w_out1_data * r_data, axis=0) - readout_theta)
    y2_trace = np.maximum(0, np.sum(w_out2_data * r_data, axis=0) - readout_theta)

    # --- 2. Setup Widgets ---
    time_slider = widgets.IntSlider(
        value=0, min=0, max=nstep-1, step=100, 
        description='Time (ms):', layout=widgets.Layout(width='800px')
    )
    plot_output = widgets.Output()
    
    global readout_box
    readout_box = widgets.VBox([time_slider, plot_output])
    display(readout_box)

    # --- 3. The Update Logic ---
    def on_value_change(change):
        t = change['new']
        with plot_output:
            clear_output(wait=True)
            plt.close('all')
            
            fig = plt.figure(figsize=(16, 10), constrained_layout=True)
            gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 2])
            
            # ================= LEFT: NETWORK SCHEMATIC =================
            ax_net = fig.add_subplot(gs[0])
            ax_net.set_title(f"Dual Readout State (t={t})", fontsize=14)
            
            # Layout
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            rnn_x, rnn_y = np.cos(angles), np.sin(angles)
            
            # Positions for two readouts (Top-Right and Bottom-Right)
            read1_pos = (2.8, 0.6)
            read2_pos = (2.8, -0.6)
            
            # Current states
            curr_r = r_data[:, t]
            curr_w_rec = w_rec_data[:, t].reshape((N, N))
            curr_w1 = w_out1_data[:, t]
            curr_w2 = w_out2_data[:, t]

            # A. Draw Recurrent Weights (with noise filtering)
            max_w_rec = np.max(w_rec_data) if np.max(w_rec_data) > 0 else 1
            for i in range(N):
                for j in range(N):
                    if i != j and curr_w_rec[i, j] / max_w_rec > 0.3:
                        alpha = np.clip((curr_w_rec[i, j]/max_w_rec)**3, 0, 0.3)
                        ax_net.plot([rnn_x[j], rnn_x[i]], [rnn_y[j], rnn_y[i]], 
                                    color='gray', alpha=alpha, lw=0.5, zorder=1)

            # B. Draw Readout Weights (Blue for R1, Red for R2)
            for i in range(N):
                # Weights to Readout 1
                if curr_w1[i] > 0.01:
                    ax_net.plot([rnn_x[i], read1_pos[0]], [rnn_y[i], read1_pos[1]], 
                                color='skyblue', alpha=np.clip(curr_w1[i]*5,0,0.6), lw=1, zorder=2)
                # Weights to Readout 2
                if curr_w2[i] > 0.01:
                    ax_net.plot([rnn_x[i], read2_pos[0]], [rnn_y[i], read2_pos[1]], 
                                color='salmon', alpha=np.clip(curr_w2[i]*5,0,0.6), lw=1, zorder=2)

            # C. Plot RNN & Readout Neurons
            ax_net.scatter(rnn_x, rnn_y, c=curr_r, s=150, cmap='cividis', edgecolors='black', zorder=4)
            
            # Readout 1 (Memory A)
            ax_net.scatter([read1_pos[0]], [read1_pos[1]], c=[y1_trace[t]], s=400, 
                           cmap='Blues', edgecolors='black', marker='D', vmin=0, vmax=2)
            # Readout 2 (Memory B)
            ax_net.scatter([read2_pos[0]], [read2_pos[1]], c=[y2_trace[t]], s=400, 
                           cmap='Reds', edgecolors='black', marker='D', vmin=0, vmax=2)

            ax_net.set_xlim(-1.5, 3.5); ax_net.set_ylim(-1.5, 1.5); ax_net.axis('off')

           # ================= RIGHT: TIME SERIES =================
            gs_right = gs[1].subgridspec(2, 1, hspace=0.2)
            
            # 1. Comparison of Readout Activities + Stimulus Overlay
            ax_y = fig.add_subplot(gs_right[0])
            
            # --- OVERLAY STIMULUS ---
            # We sample the INPUT function for the whole time range to get the 'ground truth'
            # Note: For performance, you can pre-calculate these traces in Section 1
            trace_a_in = np.zeros(nstep)
            trace_b_in = np.zeros(nstep)
            for ts in range(0, nstep, 10): # Sample every 10ms for speed
                inp = INPUT(ts)
                trace_a_in[ts:ts+10] = np.mean(inp[0:int(N/2)])
                trace_b_in[ts:ts+10] = np.mean(inp[int(N/2):N])

            # Shade the regions where stimuli are active
            ax_y.fill_between(time_points, 0, np.max(y1_trace)*1.2, where=(trace_a_in > 0), 
                              color='skyblue', alpha=0.2, label='Stimulus A')
            ax_y.fill_between(time_points, 0, np.max(y2_trace)*1.2, where=(trace_b_in > 0), 
                              color='salmon', alpha=0.2, label='Stimulus B')

            # Plot the actual firing rates
            ax_y.plot(time_points, y1_trace, color='blue', lw=2, label='Recall A')
            ax_y.plot(time_points, y2_trace, color='red', lw=2, label='Recall B')
            
            ax_y.axvline(x=t, color='black', linestyle='--')
            ax_y.set_ylabel('Firing Rate')
            ax_y.set_title("Readout Response vs. External Stimulus")
            ax_y.legend(loc='upper right', fontsize='small', ncol=2)
            ax_y.set_ylim(0, max(np.max(y1_trace), np.max(y2_trace)) * 1.3)

            # 2. Weight Distribution (Heatmap for both)
            ax_w = fig.add_subplot(gs_right[1], sharex=ax_y)
            # Concatenate weights to show them together
            combined_w = np.vstack([w_out1_data, np.zeros((5, nstep)), w_out2_data])
            ax_w.imshow(combined_w, aspect='auto', cmap='magma')
            ax_w.set_ylabel('R1 Index (Top) | R2 Index (Bottom)')
            ax_w.set_xlabel('Time (ms)')

            plt.show()

    time_slider.observe(on_value_change, names='value')
    on_value_change({'new': time_slider.value})