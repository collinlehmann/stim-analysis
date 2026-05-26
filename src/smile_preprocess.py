import logging
import numpy as np
import pandas as pd
import torch as th
from miso_analysis.process_ripple import (
    get_neural_and_stim_entities, 
    label_channels, 
    read_map,
)
from nasnet import load_pretrained
from pyns.nsentity import EntityType, EventEntity
from pyns.nsfile import NSFile
logger = logging.getLogger(__name__)

def remove_cross_channel_artifacts(
        neural_data: pd.DataFrame, 
        channel_fraction_limit: float = 0.5
) -> pd.DataFrame:
    """Remove all spikes that are coincident greater than a set fraction of all channels
    
    Parameters
    ----------
    neural_data : pd.DataFrame
        Pandas dataframe containing neural data with one channel/unit (if units sorted) per column and time in the index, as returned by smile_extract.bin_spikes(). Bin size should be small enough that no more than one spike per unit occurs in each bin (e.g. 1ms).

    channel_fraction_limit : float, optional
        Maximum fraction of channels spike can be detected without being flagged for deletion
    
    Returns
    -------
    pd.DataFrame
        Copy of neural_data with coincident spikes removed (binned spike count set to zero)
    """
    cross_channel_coinc_rate = neural_data.sum(axis=1)/neural_data.shape[1]
    neural_data.loc[cross_channel_coinc_rate > channel_fraction_limit, :] = 0
    return neural_data

def get_coincidence_matrix(
        neural_data: pd.DataFrame
) -> pd.DataFrame:
    """Returns cross-channel coincidence matrix for binned neural data
    
    Parameters
    ----------
    neural_data : pd.DataFrame
        Pandas dataframe containing neural data with one channel/unit (if units sorted) per column and time in the index, as returned by smile_extract.bin_spikes(). Bin size should be small enough that no more than one spike per unit occurs in each bin (e.g. 1ms).
    
    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing coincidence matrix for neural data where row is the fraction of spikes from a single channel coincident with spikes recorded from the channel reported: sum(row AND column)/sum(row). Diagonal is 1 by definition, unless no spikes are recorded for a channel, in which case its row is undefined, and the rest of its column is 0. 
    """
    coincidence_matrix = (
        neural_data
        .pipe(lambda x: x.T @ x)
        .pipe(lambda x: x.div(np.diag(x), axis=1))
    )
    return coincidence_matrix

def channel_names_from_pins(
        pins: pd.Series
) -> pd.Series:
    """Take series of channel pins from Sulley's array and return series of channel names
    
    Parameters
    ----------
    pins : pd.Series
        Pandas series containing strings with pin names from Sulley's array
    
    Returns
    -------
    pd.Series
        Pandas series containing strings with channel names corresponding to pin inputs
    """    
    pin_parts = pd.DataFrame({
        'X':pins.str[5:6],
        'Y':pins.str[7:8].astype(int),
        'Z':pins.str[9:].astype(int)
        })
    pin_parts['channel_names'] = (
        (
            pin_parts.Z + 
            + 32*(pin_parts.Y-1)
            + 96*(pin_parts.X == 'B')
        )
        .astype(str)
        .str.pad(width=3, side='left', fillchar='0')
    )
    pin_parts.loc[pin_parts.Y==1, 'channel_names'] = (
        'M1.chan'
        + pin_parts.loc[pin_parts.Y==1, 'channel_names']
    )
    pin_parts.loc[pin_parts.Y!=1, 'channel_names'] = (
        'PMd.chan'
        + pin_parts.loc[pin_parts.Y!=1, 'channel_names']
    ) 
    pin_names = pin_parts['channel_names']
    return pin_names

def get_impedance_data(
    filepath: str
) -> pd.DataFrame:
    """Read impedance data from impedance test file.
    
    Parameters
    ----------
    filepath : str
        Complete path to impedance test file.
    
    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing an 'impedance' column recording impedance magnitude, indexed by channel name.
    """
    impedance_table = pd.read_table(
        filepath, 
        skiprows=10, 
        header=None, 
        sep=r'\s+', 
        engine='python'
    )
    impedance_labels = channel_names_from_pins(impedance_table.iloc[:,2])
    impedance_data = pd.DataFrame(
        {
            'impedance':impedance_table.iloc[:,7], 
            'recorded channel':impedance_labels
        }
    ).set_index('recorded channel')
    return impedance_data

def process_waveforms_fast(
    nsfile: NSFile,
    monkey: str = 'Sulley'
) -> pd.DataFrame:
    """Extract waveforms from NSFile - runs nearly 2x faster than original. 
    
    Parameters
    ----------
    nsfile : NSFile
        NSFile spe

    monkey : str, optional
        String specifying monkey name.
    
    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing individual waveform data. Each row is a waveform with columns of snippet frames. Multi-index includes snippet id, channel, unit.
    """
    neural_entities, _ = get_neural_and_stim_entities(nsfile)
    
    if len(neural_entities) == 0:
        # Return empty DataFrame with correct structure
        logger.warning("No neural entities recorded in NSFile.")
        return pd.DataFrame(
            columns=pd.RangeIndex(0, 0, name='snippet frame'),
            index=pd.RangeIndex(0, 0, name='snippet_id'),
        )
    
    #count the waveforms associated with each neural entity
    wf_counts = [entity.item_count for entity in neural_entities]
    total_wfs = sum(wf_counts)

    if total_wfs == 0:
        # Return empty DataFrame with correct structure
        logger.warning("No waveforms extracted from NSFile.")
        return pd.DataFrame(
                columns=pd.RangeIndex(0, 0, name='snippet frame'),
                index=pd.RangeIndex(0, 0, name='snippet_id'),
        )
    
    #Get number of samples in each waveform
    num_samples = len(neural_entities[0].get_segment_data(0)[1])

    #Create series of channel names repeated to match the number of waveforms in each channel
    channel_series = (
        pd.Series([
            label_channels(entity.label, monkey) 
            for entity in neural_entities
        ])
        .repeat(wf_counts)
        .reset_index(drop=True)
    )  
    
    #preallocate sized arrays for speed
    wf_array = np.empty((total_wfs,num_samples))
    unit_array = np.empty((total_wfs))

    #working one channel at a time, fill each row with data for 1 waveform
    output_index = 0
    for entity in neural_entities:
        for index in range(entity.item_count):
            entity_data = entity.get_segment_data(index)
            wf_array[output_index,:] = entity_data[1]
            unit_array[output_index] = entity_data[2]
            output_index+=1

    #format pandas dataframe output
    df = (
            pd.DataFrame(
                wf_array,
                columns=pd.RangeIndex(start=0, stop=num_samples, name='snippet frame'),
                index=pd.RangeIndex(start=0, stop=total_wfs, name='snippet_id'),
            )
            .assign(**{
                'channel': channel_series,
                'unit': unit_array,
            })
            .set_index(['channel', 'unit'], append=True)
        )
    return df

def apply_nasnet(
    waveforms: pd.DataFrame,
    model_name: str = "UberNet_N50_L1",
    gamma: float = 0.2
) -> pd.DataFrame:
    """Apply loaded ML model from nasnset package to classify waveforms as spike or not-a-spike. See Issar, D., et al. (2020). "A neural network for online spike classification that improves decoding accuracy." J Neurophysiol 123(4): 1472-1485.
    
    Parameters
    ----------
    waveforms : pd.DataFrame
        Pandas dataframe consisting a waveform in each row, with snippet frames in the columns, as returned by process_waveforms
    
    model_name : str, optional
        String specifying pretrained model to load.
        
    gamma : float, optional
        Float specifying gamma parameter. 
    
    Returns
    -------
    pd.DataFrame
        Pandas dataframe waveforms with boolean is_spike index added
    """
    model = load_pretrained(model_name)
    classified_waveforms = (
        waveforms
        .assign(
            is_spike = lambda x: 
            model.classify(th.tensor(x.values, dtype=th.float32), gamma=gamma)
        )
        .set_index('is_spike', append=True)
    )
    return classified_waveforms

def apply_channel_map(
    waveforms: pd.DataFrame,
    mapfile: str,
    channel_col: str = "channel"
) -> type:
    """Extract x and y coordinates of channels from mapfile and assign coordinates to data in waveform dataframe
    
    Parameters
    ----------
    waveforms : pd.DataFrame
        Pandas dataframe consisting a waveform in each row, with snippet frames in the columns, and index containing "channel" level, as returned by process_waveforms.
    
    mapfile : str
        String specifying complete path to channel mapfile

    channel_col : str, optional
        String specifying index name with channel identifier
    
    Returns
    -------
    pd.DataFrame
        Pandas dataframe waveforms with x and y channel position indeces added
    """
    channel_map = read_map(mapfile).drop(columns='hw_address')
    all_other_levels = waveforms.index.names.difference([channel_col])
    waveforms_with_map = (
            waveforms
            .reset_index(level=all_other_levels)
            .pipe(pd.merge, right=channel_map, on='channel', how='left')
            .set_index(all_other_levels + ['x','y'], append=True)    
    )
    return waveforms_with_map