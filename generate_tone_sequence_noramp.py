from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, core
import numpy as np

def generate_tone_sequence(coherence, frequency, frequency_range, tone_duration=0.025, sequence_duration=0.5):
    num_tones = int(sequence_duration / tone_duration)
    num_coherent_tones = int(num_tones * coherence)

    # Generate the coherent tone sequence
    coherent_tones = [sound.Sound(value=frequency, secs=tone_duration) for _ in range(num_coherent_tones)]

    # Generate the incoherent tone sequence with octave-based spacing
    incoherent_tones = []
    for _ in range(num_tones - num_coherent_tones):
        random_octave_shift = np.random.uniform(-1, 1)
        random_frequency = frequency * 2 ** (random_octave_shift * frequency_range)
        incoherent_tones.append(sound.Sound(value=random_frequency, secs=tone_duration, hamming=True))

    # Combine the coherent and incoherent tone sequences
    tone_sequence = coherent_tones + incoherent_tones
    np.random.shuffle(tone_sequence)

    # Play the tone sequence
    for tone in tone_sequence:
        tone.play()
        core.wait(tone_duration)

# Example usage:
#generate_tone_sequence(coherence=0.9, frequency=4000, frequency_range=1)

generate_tone_sequence(coherence=0.1, frequency=4000, frequency_range=2)