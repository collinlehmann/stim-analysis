import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def view_waveforms(
    waveforms: pd.DataFrame,
    save_to: str = None,
    inline_display: bool = True
) -> plt.figure.Figure:
    """Plot waveforms extracted 
    
    Parameters
    ----------
    waveforms : pd.DataFrame
        Pandas dataframe consisting a waveform in each row, with snippet frames in the columns, as returned by process_waveforms
    
    save_to : str, optional
        Complete file path for figure export location.
    
    inline_display : bool, optional
        Boolean deciding whether to show figure inline.
    
    Returns
    -------
    plt.figure.Figure
        Matplotlib figure object for waveform visualization
    """
    waveforms_for_plot = (
        waveforms
        .reset_index()
        .melt(
            id_vars=["snippet_id","x","y","is_spike","channel"],
            value_name='Amplitude (μV)', 
            var_name='Time (ms)'
        ) 
    )
    
    fig, axes = plt.subplots(8, 18, figsize=(18, 9))
    for row in range(1,9):
        for column in range(1,19):
            subplot_data = (
                waveforms_for_plot.loc[(waveforms_for_plot['x']==column) 
                & (waveforms_for_plot['y']==row),:].reset_index()
            )
            if not subplot_data.empty:
                sns.lineplot(
                    subplot_data, 
                    x='Time (ms)', 
                    y='Amplitude (μV)', 
                    hue='is_spike', 
                    units='snippet_id', 
                    estimator=None, 
                    ax=axes[row-1, column-1], 
                    legend=False, 
                    lw=0.25
                )
                axes[row-1,column-1].text(
                    x=0.5, 
                    y=0, 
                    s=subplot_data.loc[0,'channel'], 
                    transform=axes[row-1,column-1].transAxes, 
                    va='bottom', 
                    ha='center', 
                    fontsize=8
                )
            axes[row-1,column-1].set(
                xlabel=None, 
                ylabel=None, 
                xticklabels=[], 
                yticklabels=[],
                ylim=(-1000,1000)
            )
    if save_to:
        plt.savefig(save_to)
    if inline_display:
        plt.show()
    return fig