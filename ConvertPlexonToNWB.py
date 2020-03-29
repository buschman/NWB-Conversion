from qtpy.QtWidgets import QFileDialog, QApplication
from os import getcwd, path
import warnings
import sys
from datetime import datetime
from dateutil.tz import tzlocal
import numpy as np

from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.ecephys import SpikeEventSeries, EventWaveform

import neo  # this has code for reading Plexon files


def AddPlexonSpikeDataToNWB(plx_file_name, nwb_file_name='', nwb_file=None, elec_ids=None, verbose=False, add_units=False, unit_info=None):
    """
        Copies filtered LFP data from the specified Blackrock file to Neurodata Without Borders (NWB) file.
        User must specify the file with the LFP data (at desired sampling rate and with desired filtering).
        Multiple calls can be used to load multiple LFP data from multiple files.

        Typically, the NWB file will already be initialized by calling InitializeNWBFromBlackrock.
        Multiple electrodes can be added at once or with separate calls to this function.

        :param plx_file_name: {str} full path of Plexon file to convert to NWB. If empty, will open dialog.
        :param nwb_file_name: [optional] {str} full path of NWB file to export to. Default is to change blackrock extension to 'nwb'
        :param elec_ids: [optional] {list} List of electrode IDs to copy over. If empty, all are copied
        :param verbose: [optional] {bool} whether to print updates while converting. Default is false.
        :param add_units: [optional] {bool} whether to add waveform labels as units in NWB file. Default is false.
        :param unit_info: [optional] {list} extra information about units
        :return: {str} filename of NWB file (empty if error)
        """
    # Check to see if user specified a Blackrock filename
    if not plx_file_name:  # no file name passed
        # Ask user to specify a file
        if 'app' not in locals():
            app = QApplication([])
        plx_file_name = QFileDialog.getOpenFileName(QFileDialog(), "Select File", getcwd(), "Plexon Data File (*.plx)")
        plx_file_name = plx_file_name[0]

    # Check to see if an open NWB file was passed, if so use that
    nwb_io = None
    if nwb_file is None:
        # Check to see if valid nwb_file_name is passed
        if not nwb_file_name:
            nwb_file_name = path.splitext(plx_file_name)[0] + '.nwb'
        if verbose: print("Writing to NWB data file %s" % (nwb_file_name))

        # Initialize the NWB file
        try:
            if not path.isfile(nwb_file_name):
                # Initialize NWB file
                if verbose: print("NWB file doesn't exist. Creating new one: %s..." % (nwb_file_name))
                raise ValueError("NWB file doesn't exist.")
                # InitializeNWBFromPlexon(plx_file_name, nwb_file_name, verbose=verbose) # TODO: write other functions for Plexon

            # Append to existing file
            if verbose: print("Opening NWB file %s..." % (nwb_file_name), end='')
            nwb_file_append = True
            nwb_io = NWBHDF5IO(nwb_file_name, mode='a')
            nwb_file = nwb_io.read()
            if verbose: print("done.")
        except:  # catch *all* exceptions
            e = sys.exc_info()[0]
            if nwb_io: nwb_io.close()
            raise FileExistsError("Couldn't open NWB file. Error: %s" % e)

    # Read in data from Plexon file
    try:
        # Requires the raw data to be imported
        if verbose: print("Reading data from PLX file: %s" % (plx_file_name))
        plx_file = neo.io.PlexonIO(plx_file_name)
    except:  # catch *all* exceptions
        e = sys.exc_info()[0]
        warnings.warn("Couldn't open Plexon file. Error: %s" % e, UserWarning)
        return ""


    # Validate the elec_ids list
    if not elec_ids:
        # Grab all of the data
        elec_ids = []
        for i, head in enumerate(plx_file.header['signal_channels']):
            elec_ids.append(head[1])

    # Create processing module for saving ephys data
    if 'ecephys' not in nwb_file.processing.keys():
        if verbose: print("Processing module for extracellular electrophysiology (ecephys) does not exist. Creating.")
        ecephys_module = ProcessingModule(name='ecephys',
                                      description="Processing module for recordings from %s." % (path.split(plx_file_name)[1]))
        nwb_file.add_processing_module(ecephys_module)

    # Create data interface for the spike waveforms
    event_waveform_name = "Spike waveforms from %s" % (path.split(plx_file_name)[1])
    if event_waveform_name not in nwb_file.processing['ecephys'].data_interfaces.keys():
        cur_eventwaveform = EventWaveform(name="Spike waveforms from %s" % (path.split(plx_file_name)[1]))
        if verbose: print("Created EventWaveform data interface (%s)" % (cur_eventwaveform.name))
    else:
        cur_eventwaveform = nwb_file.processing['ecephys'].data_interfaces[event_waveform_name]

    # Loop through and save spiking data for each electrode --  we are doing this separately for each electrode so that
    # we can save information about the threshold used for each channel in the description.
    if verbose: print("Writing spike waveforms data to NWB.")
    for cur_elec_ind, cur_elec in enumerate(elec_ids):
        # Find indices in the extended header information
        try:
            cur_chan_ind = plx_file.channel_id_to_index([cur_elec])
        except:  # catch *all* exceptions
            print("Couldn't find specified electrode %d in NSx file. Skipping." % (cur_elec))
            continue
        cur_chan_ind = cur_chan_ind[0]
        if verbose: print("\tWriting spike waveforms for electrode %d." % (cur_elec))

        # Create electrode table
        cur_elec_table_ind = 0
        while (cur_elec_table_ind < len(nwb_file.electrodes)) and (nwb_file.electrodes[cur_elec_table_ind, 0] != cur_elec):
            cur_elec_table_ind = cur_elec_table_ind + 1
        if cur_elec_table_ind >= len(nwb_file.electrodes):
            raise ValueError("Couldn't find electrode %d in NWB file list." % (cur_elec))
        electrode_table_region = nwb_file.create_electrode_table_region([cur_elec_table_ind], "Electrode %s" % (cur_elec))

        # Find units from this electrode
        unit_index_list = []
        for i, cur_unit_channel in enumerate(plx_file.header['unit_channels']):
            if cur_unit_channel[0] == plx_file.header['signal_channels'][cur_chan_ind][0]:
                unit_index_list.append(i)
        if not unit_index_list:
            print("\tCan't find any units associated with electrode %d." % (cur_elec))
            continue

        # Grab all of the waveforms
        wf = np.array(plx_file.get_spike_raw_waveforms(unit_index=unit_index_list[0]))
        ts = np.array(plx_file.get_spike_timestamps(unit_index=unit_index_list[0]))
        # Check the length of these waveforms against what already exists
        spk_wf_len = wf.shape[1]
        for i in np.arange(1, len(unit_index_list)):
            ts = np.concatenate([ts, np.array(plx_file.get_spike_timestamps(unit_index=unit_index_list[i]))], axis=0)
            cur_wf = np.array(plx_file.get_spike_raw_waveforms(unit_index=unit_index_list[i]))
            # Check the length of the waveforms against the limit for this interface
            if cur_wf.shape[1] > spk_wf_len:
                # Clip outer edges of waveform
                if verbose:
                    warnings.warn("Unit index %d on electrode %d has too many waveform points (%d). Clipping to %d." % (
                        i, cur_elec, wf.shape[1], spk_wf_len))
                wf_clip = int(np.round((cur_wf.shape[1] - spk_wf_len) / 2) - 1)
                cur_wf = cur_wf[:, wf_clip:(wf_clip + spk_wf_len)]
            elif cur_wf.shape[1] < spk_wf_len:
                # Pad with zeros to get to correct size
                if verbose:
                    warnings.warn("Unit index %d on electrode %d has too many waveform points (%d). Clipping to %d." % (
                        i, cur_elec, wf.shape[1], spk_wf_len))
                temp_wf = np.zeros((cur_wf.shape[0], spk_wf_len))
                wf_pad = int(np.round((spk_wf_len - cur_wf.shape[1]) / 2) - 1)
                temp_wf[:, wf_pad:(wf_pad + wf.shape[1])] = cur_wf
                cur_wf = temp_wf
            # Add current waveforms to overall matrix
            wf = np.concatenate([wf, cur_wf], axis=0)
        wf = wf.squeeze()

        # Write raw data for the electrode
        if verbose: print("\tAdding waveforms (%d by %d) to NWB..." % (wf.shape), end='')
        # Create spike event series
        spike_wf = SpikeEventSeries('Spike waveforms for Channel %s' % (cur_elec),
                                    wf,
                                    ts,
                                    electrode_table_region,
                                    conversion=plx_file.header['signal_channels'][cur_chan_ind][5], # pulled from header
                                    description="Spikes waveforms from channel %d in PLX file." % (cur_elec))
        # Add to our event waveform module
        cur_eventwaveform.add_spike_event_series(spike_wf)
        if verbose: print("done")

    # Add spike waveform container to NWB file
    if verbose: print("\tAdding spike waveform data to processing module.")
    nwb_file.processing['ecephys'].add(cur_eventwaveform)

    if add_units:
        # Check to see if we have any units already
        if (nwb_file.units is not None) and ('waveform_mean' in nwb_file.units.colnames):
            unit_wf_len = nwb_file.units['waveform_mean'].data[0].shape[-1]
        else:
            unit_wf_len = np.NaN

        # Check to see if user rating column already exists
        if (not nwb_file.units) or ('UserRating' not in nwb_file.units.colnames):
            nwb_file.add_unit_column('UserRating', 'Quality of isolation, as rated by user.')
        if (not nwb_file.units) or ('UserComment' not in nwb_file.units.colnames):
            nwb_file.add_unit_column('UserComment', 'Comments on spike isolation from user.')

        # Add units to the NWB file
        if verbose: print("Adding units to NWB file.")
        for cur_elec_ind, cur_elec in enumerate(elec_ids):
            # Determine the electrode index in NWB file for this electrode id
            cur_elec_table_ind = 0
            while (cur_elec_table_ind < len(nwb_file.electrodes)) and (
                    nwb_file.electrodes[cur_elec_table_ind, 0] != cur_elec):
                cur_elec_table_ind = cur_elec_table_ind + 1
            if cur_elec_table_ind >= len(nwb_file.electrodes):
                raise ValueError("Couldn't find electrode %d in NWB file list." % (cur_elec))

            # Get list of units for this electrode
            unit_index_list = []
            for i, cur_unit_channel in enumerate(plx_file.header['unit_channels']):
                if cur_unit_channel[0] == plx_file.header['signal_channels'][cur_chan_ind][0]:
                    unit_num = int(cur_unit_channel[1].split('#')[-1])
                    unit_index_list.append(unit_num)
            if not unit_index_list:
                print("\tCan't find any units associated with electrode %d." % (cur_elec))
                continue
            if verbose: print("\tElectrode %d, found %d unique units (%s)." % (cur_elec, len(unit_index_list), unit_index_list))

            # Loop through units
            for cur_unit_index, cur_unit in enumerate(unit_index_list):
                #Get wf and ts for this unit
                wf = plx_file.get_spike_raw_waveforms(unit_index=cur_unit_index)
                wf = wf.squeeze()
                ts = plx_file.get_spike_timestamps(unit_index=cur_unit_index)

                # Check the length of the waveforms against the limit for the file
                if np.isnan(unit_wf_len):
                    # Units haven't been written yet, we get to define the length of waveforms
                    unit_wf_len = wf.shape[1]
                if wf.shape[1] > unit_wf_len:
                    # Clip outer edges of waveform
                    if verbose:
                        warnings.warn("Unit %d on electrode %d has too many waveform points (%d). Clipping to %d." % (
                            cur_unit, cur_elec, wf.shape[1], unit_wf_len))
                    wf_clip = int(np.round((wf.shape[1] - unit_wf_len) / 2) - 1)
                    wf = wf[:, wf_clip:(wf_clip + unit_wf_len)]
                elif wf.shape[1] < unit_wf_len:
                    # Pad with zeros to get to correct size
                    if verbose:
                        warnings.warn("Unit %d on electrode %d has too few waveform points (%d). Clipping to %d." % (
                            cur_unit, cur_elec, wf.shape[1], unit_wf_len))
                    temp_wf = np.zeros((wf.shape[0], unit_wf_len))
                    wf_pad = int(np.round((unit_wf_len - wf.shape[1]) / 2) - 1)
                    temp_wf[:, wf_pad:(wf_pad + wf.shape[1])] = wf
                    wf = temp_wf

                # Interval over which this unit was observed
                obs_intervals = np.zeros((1,2))
                obs_intervals[0,0] = np.min(ts)
                obs_intervals[0,1] = np.max(ts)

                # Find the index in the unit_info list
                if not unit_info:
                    cur_unit_info = {'UserRating': -1, 'UserComment': ''}
                else:
                    unit_dict = {'electrode': cur_elec, 'unit': cur_unit}
                    i = 0
                    while (i < len(unit_info)) and not all(item in unit_info[i].items() for item in unit_dict.items()):
                        i = i + 1
                    if i >= len(unit_info):
                        cur_unit_info = {'UserRating': -1, 'UserComment': ''}
                    else:
                        cur_unit_info = unit_info[i]
                #Make sure UserComment is a string
                if not isinstance(cur_unit_info['UserComment'], str):
                    warnings.warn('UserComment for Unit %d on Electrode %d was not valid. Setting to empty.' % (cur_unit, cur_elec))
                    cur_unit_info['UserComment'] = ''
                # Make sure UserComment is a string
                if not isinstance(cur_unit_info['UserRating'], int):
                    warnings.warn('UserRating for Unit %d on Electrode %d was not valid. Setting to empty.' % (cur_unit, cur_elec))
                    cur_unit_info['UserRating'] = -1

                # Add to NWB file
                nwb_file.add_unit(spike_times=ts,
                                  obs_intervals=obs_intervals,
                                  electrodes=[cur_elec_table_ind],
                                  waveform_mean=np.nanmean(wf, axis=0),
                                  waveform_sd=np.nanstd(wf, axis=0),
                                  UserRating=cur_unit_info['UserRating'],
                                  UserComment=cur_unit_info['UserComment'])
                if verbose: print("\t\tAdded class %s." % (cur_unit))

    # Did we open a file?  Then close it...
    if nwb_io != None:
        # Write the file
        if verbose: print("\tWriting NWB file and closing.")
        nwb_io.write(nwb_file)
        nwb_io.close()
        return nwb_file_name
    else:
        return nwb_file