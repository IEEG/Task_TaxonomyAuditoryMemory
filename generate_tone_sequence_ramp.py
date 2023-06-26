from psychopy import sound, core
import numpy as np

def generate_tone_sequence(coherence, frequency, octave_range, tone_duration=0.025, ramp_duration=0.005, sequence_duration=0.5):
    num_tones = int(sequence_duration / tone_duration)
    num_coherent_tones = int(num_tones * coherence)

    # Generate the coherent tone sequence
    coherent_tones = [sound.Sound(value=frequency, secs=tone_duration) for _ in range(num_coherent_tones)]

    # Generate the incoherent tone sequence
    incoherent_tones = []
    for _ in range(num_tones - num_coherent_tones):
        random_octave_shift = np.random.uniform(-octave_range, octave_range)
        random_frequency = frequency * 2 ** random_octave_shift
        incoherent_tones.append(sound.Sound(value=random_frequency, secs=tone_duration))

    # Combine the coherent and incoherent tone sequences
    tone_sequence = coherent_tones + incoherent_tones
    np.random.shuffle(tone_sequence)

    # Play the tone sequence with amplitude ramps
    for tone in tone_sequence:
        tone.setRamp(ramp_duration, ramp_duration)
        tone.play()
        core.wait(tone_duration)

# Example usage:
generate_tone_sequence(coherence=0.5, frequency=440, octave_range=0.5)
