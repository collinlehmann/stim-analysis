import pandas as pd

def multi_index_resample(
    df: pd.DataFrame, 
    timecol: str = "time", 
    freq: pd.Timedelta = pd.Timedelta(milliseconds=10)
) -> pd.DataFrame:
    """Perform time resampling on multi-index pandas dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe containing neural data with one channel/unit (if units sorted) per column and time in the index, as returned by smile_extract.bin_spikes().

    timecol : str, optional
        String specifying name of time index.

    freq : pd.Timedelta, optional
        pd.Timedelta specifying new sampling frequency.
    
    Returns
    -------
    pd.DataFrame
        Original dataframe resampled at new sampling frequency.
    """
    assert isinstance(df.index, pd.MultiIndex), "Dataframe is not multi-index."
    all_other_levels = df.index.names.difference([timecol])
    groupers = (
        [pd.Grouper(level=level) for level in all_other_levels]
        + [pd.Grouper(level=timecol,freq=freq)]
    )
    resampled_data = df.groupby(groupers)
    return resampled_data