from midi2audio import FluidSynth
import os
import sys


# This program converts a folder of .midi files to a folder of .wav files
# Need to download FluidSynth and midi2audio packages
#
# Usage:
# python convert_to_wav.py midi_dir wav_dir
# More info about FluidSynth: https://github.com/FluidSynth/fluidsynth
# More info about midi2audio: https://github.com/bzamecnik/midi2audio
# Need Sound fonts to run this program: https://sites.google.com/site/soundfonts4u/
# The sound font used in this program: https://drive.google.com/file/d/1nvTy62-wHGnZ6CKYuPNAiGlKLtWg9Ir9/view?usp=sharing

def convert_midi_to_audio(input_dir, output_dir, fs):
    # sound_font_path = os.path.join(os.getcwd(), "Dore Mark's NY S&S Model B-v5.2.sf2")
    # fs = FluidSynth(sound_font_path)
    os.chdir(input_dir)
    filenames = os.listdir(input_dir)
    for midi_file in filenames:
        filename = midi_file[:-5]
        filename = filename + ".wav"
        output_file = os.path.join(output_dir, filename)
        fs.midi_to_audio(midi_file, output_file)

    return


if __name__ == '__main__':
    sound_font_path = os.path.join(os.getcwd(), "Dore Mark's NY S&S Model B-v5.2.sf2")
    fs = FluidSynth(sound_font_path)
    # fs.midi_to_audio('MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_01_Track01_wav_0.midi', 'output.wav')

    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    current_path = os.getcwd()
    output_dir = os.path.join(current_path, output_dir)

    input_dir = sys.argv[1]
    input_dir = os.path.join(current_path, input_dir)
    convert_midi_to_audio(input_dir, output_dir, fs)
