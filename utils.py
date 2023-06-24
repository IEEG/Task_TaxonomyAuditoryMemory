"""
Useful functions for tasks typically required in an experiment

Noah Markowitz
Chase Mackey
North Shore University Hospital and Nathan Kline Institute
Human Brain Mapping Laboratory
June 2023
"""

import glob
import json
import os

import numpy as np
import psychopy
import soundfile as sf
from psychopy import visual, monitors, event, gui, core, logging, sound
from psychopy.constants import NOT_STARTED, STARTED, FINISHED
from psychopy.tools.monitorunittools import cm2pix
from psychopy.tools.filetools import fromFile, toFile
from scipy.signal import resample
from collections import OrderedDict
import json


def openingDlg():
    """The opening dialogue for AV40"""

    # TTL options
    ttlOpts = ['None', 'USB_TTL', 'ParallelPort']

    # Possible photodiode options
    diodeOpts = ['None','UpperLeft','UpperRight','LowerLeft','LowerRight']

    # Eyetracker options
    eyetrackerOpts = ['None','300','600','1200']
    
    # Response options
    responseOpts = ["saccade", "mouse", "keyboard"]

    # Retrieve info and files
    _thisDir = os.path.dirname(os.path.abspath(__file__))
    mons_all = glob.glob(_thisDir + os.sep + u'monitors' + os.sep + '*.json')
    monitorOpts = [os.path.basename(x) for x in mons_all]

    # Try loading .last_settings.json
    lastSettingsFile = _thisDir + os.sep + u'.prev_dlg.json'
    try:

        prevDlg = fromFile(lastSettingsFile)

        monIdx = monitorOpts.index(prevDlg['monitor'])
        monitorOpts.insert(0,monitorOpts.pop(monIdx))
        runId = prevDlg['runid']
        #dist = prevDlg['dist']
        ttlIdx = ttlOpts.index(prevDlg['ttl'])
        ttlOpts.insert(0,ttlOpts.pop(ttlIdx))
        responseIdx = responseOpts.index(prevDlg['responseType'])
        responseOpts.insert(0, responseOpts.pop(responseIdx))
        #diodeIdx = diodeOpts.index(prevDlg['photodiode'])
        #diodeOpts.insert(0,diodeOpts.pop(diodeIdx))
        etIdx = eyetrackerOpts.index(prevDlg['eyetracker'])
        eyetrackerOpts.insert(0,eyetrackerOpts.pop(etIdx))

    except:
        runId = ''
        #dist = 60

    # Construct dialogue common to all
    dlgTitle = 'Please enter information below'
    runDlg = gui.Dlg(title=dlgTitle)
    runDlg.addField('RunID',runId)
    #runDlg.addField('Distance from screen (cm)', dist)
    runDlg.addField('TTL',choices=ttlOpts)
    runDlg.addField('Monitor',choices=monitorOpts)
    runDlg.addField('Reponse Type', choices=responseOpts)
    #runDlg.addField('Photodiode',choices=diodeOpts)
    runDlg.addField('EyeTracker',choices=eyetrackerOpts)
    #runDlg.addText(
    #    'Distance is only necessary if visual angle ("deg") is being used to calculate\nstimulus size or if you would like to calculate visual angle post-hoc')
    #fieldnames = ['runid','dist','ttl','monitor','photodiode','eyetracker']
    runDlg.addText("If response should be pressing a touchscreen, select 'mouse' as response type")
    # fieldnames = ['runid','dist','ttl','monitor','response','eyetracker']
    fieldnames = ['runid','ttl','monitor','responseType','eyetracker']

    # If it doesn't exist, create logs folder
    logsFolder = _thisDir + os.sep + u'logs'
    if not os.path.isdir(logsFolder):
        os.makedirs(logsFolder)

    # Keep showing the dialogue until acceptable answers are passed
    nonacceptable = True
    while nonacceptable:
        runOrder = runDlg.show()
        if runDlg.OK == False:
            core.quit()

        # Check if the RunID already exists
        expInfo = dict(zip(fieldnames,runOrder))
        outputDir = logsFolder + os.sep + '%s' % (expInfo['runid'])
        if os.path.isdir(outputDir):
            runDlg.setWindowTitle('runID already exists')
            nonacceptable = True
        else:
            nonacceptable = False

    # Save accepted dialogue for next run
    #toFile(lastSettingsFile, expInfo)
    with open(lastSettingsFile,'w') as f:
        f.write( json.dumps( expInfo ) )

    # Add task settings
    #taskFile = _thisDir + os.sep + 'settings.json'
    #task_settings = fromFile(taskFile)
    #expInfo.update(task_settings)

    # Read monitor file
    monFile = _thisDir + os.sep + u'monitors' + os.sep + expInfo['monitor']
    monitor_settings = fromFile(monFile)
    expInfo['monitor'] = expInfo['monitor'].split('.json')[0]

    # If there are variables to update task settings then add them
    if 'task_settings' in monitor_settings.keys():
        custom_task_settings = monitor_settings.pop('task_settings')
        expInfo.update(custom_task_settings)

    expInfo.update(monitor_settings)

    # Set TTL settings
    if expInfo['ttl'] == 'ParallelPort':
        expInfo['ttl_port'] = expInfo['pport_port_code']
    elif expInfo['ttl'] == 'USB':
        expInfo['usb_port'] = expInfo['usb_port_code']
    else:
        expInfo['ttl_port'] = 'NaN'

    #expInfo['logfilename'] = outputFile

    expInfo['outputDir'] = outputDir
    expInfo['date'] = psychopy.data.getDateStr()
    expInfo['psychopy_version'] = psychopy.__version__

    return expInfo

def setScreen(screen_res,scrWidth,fullScr,monName,dist=60, color="black"):
    """Creates the window and monitor objects for the experiment

    Args:
        screen_res: The resolution of the screen being used
        scrWidth: The width of the screen to be used (in cm)
        fullScr: Whether to use the full screen (1 or 0)
        monName: The name of the monitor

    Returns:
        win: The window object to be used in the experiment
        mon: The monitor object to be used in the experiment

    """

    # Set monitor parameters. If it doesn't exist, create it
    mon = monitors.Monitor(monName, width=scrWidth, distance=dist)
    mon.setSizePix((screen_res))
    mon.setWidth(scrWidth)
    mon.setDistance(dist)

    # If the monitor doesn't exist yet, then save it
    all_mons = monitors.getAllMonitors()
    if monName not in all_mons:
        mon.saveMon()

    # Create the window that will draw all the stimuli
    win = visual.Window(
        size=screen_res, fullscr=fullScr, screen=0,
        allowGUI=False, allowStencil=False,
        monitor=mon, color=color, colorSpace='rgb255',
        blendMode='avg', useFBO=True, units='deg')

    return win, mon

def createAudioStream(arr, soa, samplingRate, reps, blanks=[],prepare=True):
    """

    Args:
        arr: The audio stream to play
        soa: Sound Onset Asynchrony. Time between sound onsets for each repetition
        samplingRate: Auditory samplingrate
        reps: Number of times audio stream should be repeated
        missing: which repetitions should be blank

    Returns:
        Audiostream to play

    """

    # The zero padding between each click
    dur = len(arr) / samplingRate
    offset2onset_time = soa - dur
    offset2onset_padding = np.zeros(round(samplingRate * offset2onset_time))

    # The missing/absent repetitions
    blankPeriod = np.zeros(arr.size)
    if not isinstance(blanks,list):
        blanks = [blanks]
    blankReps = np.array(blanks) - 1

    # Create the whole audio stream
    audioStream = np.array([])
    for ii in range(reps):
        if ii in blankReps:
            audioStream = np.concatenate((audioStream, blankPeriod, offset2onset_padding))
        else:
            audioStream = np.concatenate((audioStream, arr, offset2onset_padding))
    
    if prepare:
        audioStream = np.vstack((audioStream,audioStream)).T.astype('float32')

    return audioStream

def set_ttl(trigger, address):
    """This is used to create an anonymous function that sends out TTL pulses
    or does nothing but act as a standin and displays when TTL pulses would be sent

    Args:
        trigger: Type of hardware that will be used to send TTL pulses. Options are ['None','MMB','ParallelPort']
        address: Port address for the hardware

    Returns:
        Two function handles
        send_ttl: Accepts a numeric argument that is the code wishing to be sent
        close_ttl: Used to close the ttl port

    """
    if trigger == 'None':
        #send_ttl = lambda code: print(code)
        #close_ttl = lambda: print('Pseudo close TTL')

        def send_ttl(code):
            None

        def close_ttl():
            None

    elif trigger == 'USB':
        try:
            import serial
            ser = serial.Serial()
            ser.port = address # Must be something like 'COM4'
            ser.timeout = 0.01
            ser.baudrate = 128000
            ser.open()

            def send_ttl(code):
                ser.write(bytes([code]))

            def close_ttl():
                ser.close()
        except:
            dlg = gui.Dlg(title="No USB TTL Found!", pos=(200, 400))
            dlg.addText('Subject Info', color='Red')
            dlg.show()
            core.quit()
    
    elif trigger == 'MMB':
        try:
            import serial
            ser = serial.Serial()
            ser.port = address # Must be something like 'COM4'
            ser.timeout = 0
            ser.baudrate = 9600
            ser.open()

            def send_ttl(code):
                ser.write(bytes([code]))

            def close_ttl():
                ser.close()
        except:
            dlg = gui.Dlg(title="No MMB Trigger Box Found!", pos=(200, 400))
            dlg.addText('Subject Info', color='Red')
            dlg.show()
            core.quit()
    
    # Direct parallel port
    elif trigger == 'ParallelPort':
        from psychopy import parallel
        p = address # Must be something like 'DFF8'
        parallel.setPortAddress(int(p,16))
        parallel.setData(0)
        #send_ttl = lambda code: parallel.setData(code)

        def send_ttl(code):
            from psychopy import parallel
            parallel.setData(code)
            core.wait(0.001)
            parallel.setData(0)

        def close_ttl():
            None

    return send_ttl, close_ttl

def read_wav(filename, new_fs=48000, dual=True):
    """Read a wav file and adjust its sampling rate to desired rate

    Args:
        filename: wav filename
        new_fs: the desired sampling rate

    Returns: numpy array of audio file resampled to new_fs

    """

    soundArray, orig_fs = sf.read(filename, dtype='float32')
    
    if soundArray.ndim == 1 and dual:
        soundArray = np.vstack((soundArray, soundArray)).T

    if new_fs not in [orig_fs, None]:
        audTime = soundArray.shape[0] / orig_fs
        newNumSamples = round(audTime * new_fs)
        audStream = resample(soundArray, newNumSamples)
        return audStream
    else:
        return soundArray
        

def createToneReps(value="A",tone_dur=0.05, blank_dur=0.05, reps=2, sampleRate=44100):
    tmp = sound.Sound(value=value, secs=tone_dur, sampleRate=sampleRate,stereo=True, autoLog=False)
    tone = tmp.sndArr
    blank = np.zeros(( round(blank_dur*sampleRate), 2))
    for ii in range(reps):
        if ii == 0:
            arr = tone
        else:
            arr = np.vstack((arr, tone))
        arr = np.vstack((arr, blank))
    return arr
    
def pauseAndReadText(win,TxtToWrite,mouse=None,txtColor = [0,0,0],keys=['escape'],wait=2):
    """Displays a message to the screen. Waits until users presses the mouse button
    or presses 'escape' on the keyboard

    Args:
        win: The window object for experiment
        TxtToWrite: The message to be displayed in this instance
        mouse: The mouse object
        txtColor: The color the text should be

    Returns:
        Boolean value (True or False) for whether the task should continue.
        Based on if the user clicked the mouse or clicked "escape" on the keyboard.

    """
    continueRoutine = True
    pauseText = visual.TextStim(win = win, units = 'norm', height = 0.1,
                pos = (0,0), text = TxtToWrite, alignHoriz = 'center',
                alignVert = 'center', color = txtColor, wrapWidth=1.5, autoLog=False)
    pauseText.setAutoDraw(True)
    # If mouse given as argument
    useMouse = False
    from psychopy.event import Mouse
    if isinstance(mouse,Mouse):
        useMouse = True
        mouseObj = mouse
        mouseObj.status = NOT_STARTED
        mouseObj.status = STARTED
        prevButtonState = mouseObj.getPressed(0)  # if button is down already this ISN'T a new click

    pauseText.setAutoDraw(True)
    win.flip()
    core.wait(wait)
    
    # Continue until a button is pressed
    while continueRoutine:

        if useMouse:
            buttons = mouseObj.getPressed(0)
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if buttons:
                    clickedBttn = 'mouse'
                    continueRoutine = False


        # check for a key press
        keyPressed = event.getKeys(keyList=keys)
        if any(keyPressed):
            clickedBttn = keyPressed[0]
            continueRoutine = False

        if not continueRoutine:
            pauseText.setAutoDraw(False)
            pauseText.status = FINISHED
            if useMouse:
                mouseObj.status = FINISHED

        win.flip()

    win.flip()
    return clickedBttn
