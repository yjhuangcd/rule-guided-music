import os
import pretty_midi
import argparse
import random

def select_midi(input_path, output_dir, select_length=10.24):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for midi_file_name in os.listdir(input_path):
        if not (midi_file_name.endswith('.midi') or midi_file_name.endswith('.mid')):
            continue  # Skip non-midi files

        full_path = os.path.join(input_path, midi_file_name)
        try:
            midi_data = pretty_midi.PrettyMIDI(full_path)
        except Exception as e:
            print(f"Error processing {midi_file_name}: {e}")
            continue  # Skip to the next file if an error occurs

        end_time = midi_data.get_end_time()  # Get end time directly with pretty_midi

        if select_length > end_time:
            print("Segment length is longer than the MIDI file duration.")
            continue

        start_time = random.uniform(0, end_time - select_length)
        segment_end_time = start_time + select_length

        # Create a new MIDI object for each chunk
        chunk_midi_data = pretty_midi.PrettyMIDI()

        # Merge non-drum instruments into a single instrument
        merged_instrument = pretty_midi.Instrument(program=0, is_drum=False)

        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    if start_time <= note.start < segment_end_time:
                        # Shift the note start and end times to start at 0
                        new_note = pretty_midi.Note(
                            velocity=note.velocity,
                            pitch=note.pitch,
                            start=note.start - start_time,
                            end=note.end - start_time
                        )
                        merged_instrument.notes.append(new_note)
            else:
                # If it's a drum instrument, just adjust the note times and append it
                new_drum_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=True,
                                                             name=instrument.name)
                new_drum_instrument.notes = [note for note in instrument.notes if
                                             start_time <= note.start < segment_end_time]
                for note in new_drum_instrument.notes:
                    note.start -= start_time
                    note.end -= start_time
                chunk_midi_data.instruments.append(new_drum_instrument)

        # Add the merged instrument to the MIDI object
        chunk_midi_data.instruments.append(merged_instrument)

        # Save the chunk with the same name as the original file
        chunk_midi_data.write(os.path.join(output_dir, midi_file_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk MIDI files into specified lengths.")
    parser.add_argument("--input_path", type=str, help="Path to the directory containing the MIDI files to chunk.")
    parser.add_argument("--output_dir", type=str, help="Path to the directory where the chunked MIDI files will be saved.")
    parser.add_argument("--select_length", type=float, default=10.24, help="length to chunk the midi file to (s).")
    args = parser.parse_args()

    select_midi(args.input_path, args.output_dir, select_length=args.select_length)
