# Music Rules Doc
Use music21 package to classify the key and detect the chord progression out of a sample.
## Package
numpy, music21
## Usage

```python
python music_rules.py
```
Need to have target file ready, either one midi or npy file

## Functions

### classify_keys(filename):
Given a filename of a midi file, classify which key the midi file is. The key finding algorithm that music21 uses: http://rnhart.net/articles/key-finding/

**Return: key (string), the key correlation (int), the tonal certainty (int)**

### classify_keys_from_stream(s):
Given a music21 stream, classify which key this stream is.
    The key finding algorithm that music21 uses: http://rnhart.net/articles/key-finding/

**Return: key (string), the key correlation (int), the tonal certainty (int)**

### piano_roll_to_music21_stream(piano_roll, fs=10, program=0):

Convert a Piano Roll array into a Music21 Stream object with a single instrument.
**TODO:** this function can not deal with overlap between two notes.

**Return: A music21 stream class instance describing the piano roll.**

### get_chord_progression(filename, window_size=0.5):

Given a filename of a midi file, return a sequence of Roman Numeral Analysis.
Separate the music into little pieces with the length of window_size seconds (0.5s for each pieces by default)
For each window with window_size, get the chord lasts longest in such window.

**Return: a list of roman numerals of chords, a list of strings**

### get_chord_progression_from_stream(s, window_size=0.5):

Given a music21 stream, return a sequence of Roman Numeral Analysis.
Separate the music into little pieces with the length of window_size seconds (0.5s for each pieces by default)
For each window with window_size, get the chord lasts longest in such window.

**Return: a list of roman numerals of chords. A list of strings**

### get_longest_chords(chords, end_time, window_size=0.5):

 Given a list of chords, split the list of chords into small time periods with the length of window_size
 Then, given each window_size, compute the chord with the longest duration (which chord lasts longest in the time of window_size)

    :params chords: a nested list, where each element is a list containing three entries, this list will also in start time increasing manner
                    the first entry is how long this chord lasts
                    the second entry when this chord starts
                    the third entry is the name of the chord
                    e.g. [[0.1, 0.03, "C major"], [0.2, 0.05, "C minor"], [0.1, 0.08, "A major"]]
            end_time: an int, the end time of such chord list (or such music piece), in seconds
            window_size: the size of the time length that we check the chord
**Return: return: a list of chord names (roman numerals).**

### duration_to_seconds(d, tempo):
Convert a music21 duration to seconds based on a given tempo.

**Return: duration in seconds.**

### read_piano_roll(filename):
Read a piano roll

**Return: a piano roll.**