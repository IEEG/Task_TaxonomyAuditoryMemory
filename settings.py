SETTINGS = {
    "iti": (0,1),                   # Range of intertrial interval in seconds
    "nTrials": 1,                   # Number of repetitions of each trial definied in blocks
    "pre_time": 0.624*1,            # Time before onset of Cue in seconds
    "trial_time": 0.624*8,          # Time between when cue and choice sounds in seconds
    "wait_time": 0.624*8,           # duration to allow for choice before proceeding to next trial in seconds
    "click_soa": 0.624,             # Time between clicks in seconds
    "click_dur": 0.01,              # Duration of a single click in seconds
    "response_fixation_time": 0.5,  # Duration required for fixation in seconds
    "sequence_duration": 0.5, 	    # Duration of cue and target sounds in seconds
    "feedback_duration": 1, 	    # Duration of feedback in seconds
    "click_scale": 0.25,            # Scale of intensity of click stream
    "frequency": 400,               # Center frequency of cue and choice stimulus In Hz
    "frequency_range": 0.1, 	    # Range of deviation around center frequency in Ocatves
    "coherence": 0.5, 		    # Coherence of 'soundcloud'	
    "cue_seeds": (42, 56),          # Random seeds for cue sounds, size of tuple defines number of distinct sounds
    "target_seeds": (33, 102),	    # Random seeds for target sounds, size of tuple defines number of distinct sounds
    "flash_size": [200, 200]	    # Size of flashing box	
}
