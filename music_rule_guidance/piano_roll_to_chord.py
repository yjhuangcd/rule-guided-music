from music21 import *
import math
import numpy as np
import torch
from typing import Callable
import os

# Can either go into pretty-midi-modified folder to install locally,
# or directly use the below two lines with the path to pretty-midi-modified
import sys

# sys.path.append('./pretty-midi-modified/')  # doesn't need this if do 'pip install -e .' under the modified folder
import pretty_midi

KEY_DICT = {"D major": 0, "g minor": 1, "B- major": 2, "G major": 3, "d minor": 4, "c# minor": 5, "F major": 6,
            "E- major": 7, "e minor": 8, "f# minor": 9, "C major": 10, "F# major": 11, "g# minor": 12, "A major": 13,
            "a minor": 14, "B major": 15, "A- major": 16, "b- minor": 17, "E major": 18, "c minor": 19, "b minor": 20,
            "e- minor": 21, "f minor": 22, "C# major": 23, "no key": 24}

IND2KEY = {v: k for k, v in KEY_DICT.items()}

MIN_PIANO, MAX_PIANO = 21, 108


def get_chord_progression(stream, window_size, k_str=None, total_time=25.6, show_chords=False):
    """
    Given a filename of a midi file, return a sequence of Roman Numeral Analysis.
    The RESULT is too long.
    :param filename: midi filename
    :return: there are two conditions: chunk 1 and chunk 2
    If chunk 1: a list of roman numerals of chords
    If chunk 2: a nested list of roman numerals of chords.
             Each list inside bigger list is a bar of chords corresponding with the index.
             E.g. [[bar 0 chords], [bar 1 chords], [bar 2 chords], ...]
    """

    # mf = midi.MidiFile()
    # mf.open(filename)
    # mf.read()
    # mf.close()
    # stream = midi.translate.midiFileToStream(mf)
    # stream = stream.makeMeasures()
    sChords = stream.chordify()
    if show_chords:
        sChords.show('text', addEndTimes=True)

    # k_str, _, _ = classify_keys(filename)
    k_str = k_str.split(' ')[0]

    end_time = duration_to_seconds(sChords.highestTime, 120)
    end_time = min(end_time, total_time)
    # print("stream highest time: ", stream.highestTime)
    # print("How many quarternotes: ", float(sChords.highestTime))
    # print("end time: ", float(end_time))
    tempo = 120

    sChords = sChords.flatten()

    list_chords = []
    for c in sChords.recurse().getElementsByClass(chord.Chord):
        rn = roman.romanNumeralFromChord(c, key.Key(k_str))
        # print(duration_to_seconds(c.offset, tempo))
        list_chords.append([float(c.seconds), float(duration_to_seconds(c.offset, tempo)), str(rn.figure)])

    chord_progressions = get_longest_chords(list_chords, end_time, window_size=window_size, total_time=total_time)
    return chord_progressions


def get_longest_chords(chords, end_time, window_size=1.6, total_time=10.24):
    result = []
    #     window_size=window_size
    # Convert chords to a NumPy array
    chords_array = np.array(chords)
    starts = chords_array[:, 1].astype(float)
    ends = starts + chords_array[:, 0].astype(float)

    # Define a function to get overlapping duration of a chord with a window
    def get_overlap_duration(chord, start, end):
        chord_end = chord[1] + chord[0]
        # print(chord[1], start)
        overlap_start = max(chord[1], start)
        overlap_end = min(chord_end, end)
        return max(0, overlap_end - overlap_start)

    current_time = 0
    while current_time < end_time:
        window_start = current_time
        window_end = current_time + window_size

        # Filter chords that fall into the current window using boolean indexing
        overlapping_indices = np.where((starts < window_end) & (ends > window_start))[0]

        # Use indices to fetch overlapping chords from the original chords list
        # overlapping_chords = [chords[i] for i in overlapping_indices]
        overlapping_chords = chords_array[overlapping_indices]

        overlapping_chords = [[float(c[0]), float(c[1]), str(c[2])] for c in overlapping_chords]

        #     # If there are overlapping chords, select the one with the longest overlap duration
        if len(overlapping_chords) > 0:

            chord_durations = [get_overlap_duration(chord=chord_i, start=window_start, end=window_end) for chord_i in
                               overlapping_chords]

            max_id = np.argmax(chord_durations)

            max_chord = overlapping_chords[max_id]

            result.append(max_chord[2])

        else:
            result.append('null')

        current_time += window_size

    # need to have results with fixed length, pad null for empty music
    while len(result) < int(total_time / window_size):
        result.append('null')

    return result


def duration_to_seconds(d, tempo):
    """
    Convert a music21 duration to seconds based on a given tempo.

    Parameters:
    - duration: a music21 duration object or a float representing the duration in quarter notes.
    - tempo: tempo in beats per minute (BPM).

    Returns:
    - duration in seconds.
    """
    if isinstance(d, duration.Duration):
        duration_quarter_notes = d.quarterLength
    else:
        duration_quarter_notes = d  # assuming it's already in quarter notes

    return (duration_quarter_notes / tempo) * 60


def piano_roll_to_music21_stream(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a Music21 Stream object
   with a single instrument.
  Parameters
  ----------
  piano_roll : np.ndarray, shape=(128,frames), dtype=int
      Piano roll of one instrument
  fs : int
      Sampling frequency of the columns, i.e. each column is spaced apart
      by ``1./fs`` seconds.
  program : int
      The program number of the instrument.
  Returns
  -------
  music21_object : music21.stream
      A music21 stream class instance describing
      the piano roll.
  '''
    pm = piano_roll_to_pretty_midi(piano_roll, fs=fs)
    midi_data = pm.get_midi_data()
    pm_stream = midi.translate.midiStringToStream(midi_data)

    return pm_stream


def piano_roll_to_pretty_midi(full_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    full_roll : np.ndarray, shape=(128,frames), dtype=int, within range [0,127]
        Piano roll of one instrument, or with shape (2, 128, frames) when with pedal
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    is_onset = False
    if len(full_roll.shape) == 3:
        piano_roll = full_roll[0]
        if full_roll.shape[0] == 2:
            pedal_roll = full_roll[1]
        else:
            onset_roll = full_roll[1]
            onset_roll[onset_roll < 64] = 0  # set 64 to be the threshold
            pedal_roll = full_roll[2]
            is_onset = True
        pedal_roll[pedal_roll < 4] = 0  # background should be 0
        # only need 1D information, only take 88 range
        pedal_roll = pedal_roll[MIN_PIANO:MAX_PIANO + 1].mean(axis=0).astype(np.intc)
        is_pedal = not math.isclose(pedal_roll.max(), 0)
    else:
        piano_roll = full_roll
        is_pedal = False
    notes, frames = piano_roll.shape
    background = piano_roll[:MIN_PIANO, :].max()  # background should be 0
    piano_roll[piano_roll <= background] = 0
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # smooth uneven notes for noisy generation
    piano_roll_binary = piano_roll.copy()
    piano_roll_binary[piano_roll_binary != 0] = 1

    # use changes in velocities to find note on / note off events
    diff_piano_roll = np.diff(piano_roll_binary).T
    velocity_changes = np.nonzero(diff_piano_roll)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            if is_onset:
                start_ind = round(note_on_time[note] * fs)
                end_ind = round(time * fs)
                onsets_note = onset_roll[note, start_ind:end_ind + 1]
                onset_times = np.nonzero(onsets_note)[0]
                # only put note if there are onsets under it
                if len(onset_times) > 0:
                    # follow onset time, if want to follow pr time, use note_on_time[note] as the first start_time
                    start_times = (onset_times + start_ind) / fs
                    end_times = np.concatenate((start_times[1:], np.array([time])), axis=0)
                    for i in range(len(onset_times)):
                        pm_note = pretty_midi.Note(
                            velocity=prev_velocities[note],
                            pitch=note,
                            start=start_times[i],
                            end=end_times[i])
                        instrument.notes.append(pm_note)
            else:
                pm_note = pretty_midi.Note(
                    velocity=prev_velocities[note],
                    pitch=note,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
            prev_velocities[note] = 0

    # write pedal information
    if is_pedal:
        pedal_changes = np.nonzero(pedal_roll)
        for time, in zip(*pedal_changes):
            pedal_val = pedal_roll[time]
            if pedal_val < 16:
                pedal_val = 0  # because we used quantization with num_bin=8, 1-16 should be 0.
            if pedal_val > 112:
                pedal_val = 127
            time = time / fs
            pm_pedal = pretty_midi.ControlChange(
                number=64,  # sustain pedal
                value=pedal_val,
                time=time
            )
            instrument.control_changes.append(pm_pedal)
    pm.instruments.append(instrument)
    return pm


def chord_tag_num(chord):
    """
    Given a string of chord, return a int by chord's Roman Numerals.
    :param chord: a string, e.g. iii+64
    :return: an int.
    """
    if "VII" in chord or "vii" in chord:
        return 7
    elif "VI" in chord or "vi" in chord:
        return 6
    elif "IV" in chord or "iv" in chord:
        return 4
    elif "V" in chord or "v" in chord:
        return 5
    elif "III" in chord or "iii" in chord:
        return 3
    elif "II" in chord or "ii" in chord:
        return 2
    elif "I" in chord or "i" in chord:
        return 1
    else:
        return 0


def prepare_sequence_chord_tag_num(seq, tag_func):
    idxs = [tag_func(w) for w in seq]
    return torch.LongTensor(idxs)


def piano_roll_to_chords(piano_roll_excerpt: np.matrix,
                         given_key: str = None,
                         return_key: bool = False,
                         tagging_func: Callable = chord_tag_num,
                         fs: float = 100.,
                         window_size: float = 1.28):
    """
    Given one piano roll with shape (128, 1024), return chords, key and the correlated coefficient of string.
    This function saves a temporate midi file from the piano roll, then apply music21 package to analyze the midi file.
    This function separated this music excerpt into small non-overlap segments with window_size.
    For each segment, use the chord of longest duration as the tag of such segment.
    :param piano_roll_excerpt: npy matrix, the piano roll of music
    :param tagging_func: the tagging function to tag the chords into pytorch LongTensor
    :param fs: float
    :param window_size: float,
    :return:
        chord: a list of LongTensor.
            e.g. tensor([3, 3, 3, 3, 3, 3, 5, 3]) from ['iii+64', 'iii+64', '#iii6b42', '#iiib42', 'iii+#53', 'iii+#53', 'V6', 'iii+64']
        key_str: a string. e.g. A- major
        correlationCoefficient: a float, the correlation coefficient of the key produced by music21.
            When correlationCoefficient < 0.85, this chord tag is not recommended to use.

    """
    time_dim = piano_roll_excerpt.shape[-1]  # time dimension of piano roll
    total_time_seconds = time_dim / fs  # total time in seconds

    music21_stream = piano_roll_to_music21_stream(piano_roll=piano_roll_excerpt, fs=fs)

    # Get the chord progression, here, chords -> a list of strings
    if given_key is not None and not return_key:
        chords = get_chord_progression(stream=music21_stream, window_size=window_size, k_str=given_key,
                                       total_time=total_time_seconds, show_chords=False)
        chords = prepare_sequence_chord_tag_num(chords, tagging_func)
        out_dict = {"chords": chords}
    else:
        key_str, correlationCoefficient, _ = classify_keys_from_stream(stream=music21_stream)
        # In some cases, the key of some music excerpt may not be identified by music21
        if key_str is None:
            # print(f"Key: {key_str}, the key of piano roll can not be identified by music21.")
            null_chords = torch.zeros(int(total_time_seconds / window_size), dtype=torch.long)
            null_key = KEY_DICT["no key"]
            return {"chords": null_chords, "key": null_key, "correlationCoefficient": 0.}
        if given_key is not None:
            key_used = given_key
        else:
            key_used = key_str

        chords = get_chord_progression(stream=music21_stream, window_size=window_size, k_str=key_used,
                                       total_time=total_time_seconds, show_chords=False)
        chords = prepare_sequence_chord_tag_num(chords, tagging_func)
        out_dict = {"chords": chords, "key": KEY_DICT[key_str], "correlationCoefficient": correlationCoefficient}

    return out_dict


def piano_roll_to_chords_save_midi(piano_roll_excerpt: np.matrix,
                                   midi_dir: str = "loggings/debug",
                                   midi_savename: str = "sample.midi",
                                   tagging_func: Callable = chord_tag_num,
                                   fs: int = 100,
                                   time_dim: int = 1024,
                                   window_size: float = 1.28):
    """
    Given one piano roll with shape (128, 1024), return chords, key and the correlated coefficient of string.
    This function saves a temporate midi file from the piano roll, then apply music21 package to analyze the midi file.
    This function separated this music excerpt into small non-overlap segments with window_size.
    For each segment, use the chord of longest duration as the tag of such segment.
    :param piano_roll_excerpt: npy matrix, the piano roll of music
    :param midi_dir: str, the ABSOLUTE folder to save temporate midi file of the piano roll.
    :param midi_savename: str, the savename of such midi file.
    :param tagging_func: the tagging function to tag the chords into pytorch LongTensor
    :param fs: int
    :param time_dim: int, time dimension of piano roll
    :param window_size: int,
    :return:
        chord: a list of LongTensor.
            e.g. tensor([3, 3, 3, 3, 3, 3, 5, 3]) from ['iii+64', 'iii+64', '#iii6b42', '#iiib42', 'iii+#53', 'iii+#53', 'V6', 'iii+64']
        key_str: a string. e.g. A- major
        correlationCoefficient: a float, the correlation coefficient of the key produced by music21.
            When correlationCoefficient < 0.85, this chord tag is not recommended to use.

    """
    midi_path = midi_dir
    total_time_seconds = time_dim / fs  # total time in secondsd
    midi_savepath = os.path.join(midi_path, midi_savename)

    piano_roll_pretty_midi = piano_roll_to_pretty_midi(piano_roll_excerpt, fs=fs)

    piano_roll_pretty_midi.write(midi_savepath)

    mf = midi.MidiFile()
    mf.open(midi_savepath)
    mf.read()
    mf.close()

    music21_stream = midi.translate.midiFileToStream(mf)
    key_str, correlationCoefficient, _ = classify_keys_from_stream(stream=music21_stream)

    # In some cases, the key of some music excerpt may not be identified by music21
    if key_str == None:
        # print(f"Key: {key_str}, the key of piano roll can not be identified by music21.")
        null_chords = torch.zeros(int(total_time_seconds / window_size), dtype=torch.long)
        null_key = KEY_DICT["no key"]
        return {"chords": null_chords, "key": null_key, "correlationCoefficient": 0.}

    # Get the chord progression, here, chords -> a list of strings
    chords = get_chord_progression(stream=music21_stream, window_size=window_size, k_str=key_str,
                                   total_time=total_time_seconds, show_chords=False)

    chords = prepare_sequence_chord_tag_num(chords, tagging_func)

    out_dict = {"chords": chords, "key": KEY_DICT[key_str], "correlationCoefficient": correlationCoefficient}

    return out_dict


def classify_keys_from_stream(stream: object):
    """
    Given a filename of a midi file, classify which key the midi file is.
    The key finding algorithm that music21 uses: http://rnhart.net/articles/key-finding/
    :param filename: midi filename
    :return: key: string, the key
            correlation: integer,
    """
    try:
        fis = stream.analyze('key')
    except Exception as e:
        print(e)
        return None, None, None

    correlationCoefficient = fis.correlationCoefficient
    tonalCertainty = fis.tonalCertainty()

    return str(fis), correlationCoefficient, tonalCertainty

# if __name__ == '__main__':
#
#     # Below is the example of usage
#
#     midi_dir = "midi_files" # the absolute folder to save midi files produced by piano_roll_to_chords
#
#     os.makedirs(midi_dir, exist_ok=True)
#
#     # load piano roll
#     piano_roll = np.load('train-npy/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_02_Track02_wav_0.npy')
#     #print(piano_roll.shape)
#     out_dict = piano_roll_to_chords(piano_roll_excerpt=piano_roll, tagging_func=chord_tag_num,
#                                     fs=100, time_dim=1024, window_size = 1.28)
#     print("chords: ", out_dict["chords"])
#     print("key: ", out_dict["key"])
#     print("correlation coefficient: ", out_dict["correlationCoefficient"])
