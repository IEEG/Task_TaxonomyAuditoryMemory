"""
Auditory Memory Task

Noah Markowitz
Human Brain Mapping Laboratory
North Shore University Hospital
March 2022
"""

from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from datetime import datetime
import numpy as np
from numpy.random import randint
import os
import os.path as op
import pandas as pd
import sys
from psychtoolbox import GetSecs
from psychopy import core, event, visual, sound, logging, gui
from psychopy.hardware import keyboard
from psychopy.data import ExperimentHandler, TrialHandler2
from psychopy.constants import NOT_STARTED, STARTED, FINISHED

# Ensure that relative paths start from the same directory as this script
_thisDir = op.dirname(op.abspath(__file__))
logDir = op.join(_thisDir, 'logs')
stimDir = op.join(_thisDir, 'stimuli')
from utils import openingDlg, set_ttl, create_audioStream, setScreen, read_wav
from settings import SETTINGS

def run():

    ###############################################################
    # Setup
    ###############################################################
    expInfo = openingDlg()
    run_dir = expInfo['outputDir']

    globalFs = expInfo['sound_fs']
    ttl_code = expInfo['ttl_code']

    # Set TTL pulse
    if expInfo['ttl'] == 'ParallelPort':
        address = expInfo['pport_port_code']
    elif expInfo['ttl'] == 'MMB':
        address = expInfo['mmb_port_code']
    else:
        address = 'NaN'

    # Set how to send and then close the TTL port
    send_ttl, close_ttl = set_ttl(expInfo['ttl'], address)

    # Setup the Window, Keyboard, Mouse
    win, mon = setScreen(
        expInfo['screen_resolution'], expInfo['monitor_width'], expInfo['full_screen'], expInfo['monitor'],
        color='black', dist=expInfo['dist'])
    kb = keyboard.Keyboard()
    myMouse = event.Mouse()
    myMouse.setVisible(1)

    # Display a cross in the middle of the screen
    crossFixation = visual.ShapeStim(
        win=win, name='crossFixation', vertices='cross',
        size=(0.1,0.1), ori=0, pos=(0, 0),
        lineWidth=0, lineColor='white', lineColorSpace='rgb',
        fillColor='white', fillColorSpace='rgb', opacity=1, interpolate=True,
        units='norm')
    crossFixation.setAutoDraw(True)

    filename = expInfo['outputDir'] + os.sep + expInfo['runid']
    
    # Experiment Handler
    thisExp = ExperimentHandler(
        name=expInfo['runid'], version='0.0.1',
        extraInfo=expInfo, dataFileName=filename, autoLog=True,
        saveWideText=True, savePickle=True)

    # TrialHandler
    trials = TrialHandler2(
        op.join(_thisDir,'soundslist.csv'),
        nReps=1,
        method='sequential',
        originPath=__file__,
        extraInfo=expInfo)
    thisExp.addLoop(trials)

    # Trial feedback text
    correctResponseText = visual.TextStim(
        win=win, text='Correct', font='Arial', units='norm', pos=(0, 0),
        height=0.1, ori=0,color='white', colorSpace='rgb', opacity=1, 
        name='correctResponseText')
    incorrectResponseText = visual.TextStim(
        win=win, text='Incorrect', font='Arial', units='norm', pos=(0, 0),
        height=0.1, ori=0,color='white', colorSpace='rgb', opacity=1, 
        name='incorrectResponseText')
    noResponseText = visual.TextStim(
        win=win, text='No response timeout', font='Arial', units='norm', pos=(0, 0),
        height=0.1, ori=0,color='white', colorSpace='rgb', opacity=1, 
        name='noResponseText')


    # The response buttons
    sameBttn = visual.TextBox2(
        win,
        "Same",
        pos=(-0.5,0),
        letterHeight=0.05,
        fillColor='green',
        color='black',
        colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='sameBttn',
        size=(0.3,0.3),
        units='norm')

    diffBttn = visual.TextBox2(
        win,
        "Different",
        pos=(0.5,0),
        letterHeight=0.05,
        fillColor='red',
        color='black',
        colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='sameBttn',
        size=(0.3,0.3),
        units='norm')

    # Initiate audio
    stream = sound.Sound(name='initializer', sampleRate=globalFs, stereo=True)

    # Create audio clicks
    clickDur = SETTINGS["click_dur"]
    clickSOA = SETTINGS["click_soa"]
    trlDur = 60
    clickReps = np.floor(trlDur/clickSOA).astype(int)
    singleClick = np.ones( ( round(clickDur*globalFs),) )
    clickStream = create_audioStream(singleClick,clickSOA,globalFs,clickReps)
    choice2cue_clickStream = create_audioStream(singleClick,clickSOA,globalFs,5)

    win.flip()
    
    # Start trials
    for thisTrial in trials:

        # Load sound files
        cuesoundFile = thisTrial['cuesound']
        cueSound = read_wav(op.join(stimDir, cuesoundFile), new_fs=globalFs)
        choicesoundFile = thisTrial['choicesound']
        choiceSound = read_wav(op.join(stimDir, choicesoundFile), new_fs=globalFs)

        # Check what the correct response to this trial should be
        if cuesoundFile == choicesoundFile:
            correctResponse = "same"
        else:
            correctResponse = "different"
        
        # Create audio stream. Embed choiceSound within the click train
        audStream = np.concatenate((cueSound, choice2cue_clickStream, choiceSound, clickStream))

        # When response can start to be made
        responseStartTime = np.concatenate((cueSound, choice2cue_clickStream, choiceSound)).shape[0]/globalFs

        # Intertrial interval wait time
        thisTrialITI = randint(SETTINGS['iti'][0]*1000, high=SETTINGS['iti'][1]*1000)/1000

        # Play the audio after ITI
        stream = sound.Sound(audStream, name='trial_audio', sampleRate=globalFs, stereo=True)
        stream.status == NOT_STARTED
        stream.play(when=GetSecs()+thisTrialITI)
        
        # When audio starts to play send a TTL
        while not stream.isPlaying:
            win.flip()
        soundOnset = core.getTime()
        send_ttl(ttl_code)
        
        # Set button statuses
        tAllowResponse = soundOnset + responseStartTime
        sameBttn.status = NOT_STARTED
        sameBttn.tStart = tAllowResponse
        sameBttn.tStop = 0
        diffBttn.status = NOT_STARTED
        diffBttn.tStart = tAllowResponse
        diffBttn.tStop = 0
        
        response = 'NA'
        continueTrial = True
        
        while continueTrial:

            # Current time
            tNow = core.getTime()
            tNextFlip = win.getFutureFlipTime()
            
            # Present two images/shape representing the choices when time is right
            if tNow >= sameBttn.tStart and sameBttn.status == NOT_STARTED:
                sameBttn.setAutoDraw(True)
                diffBttn.setAutoDraw(True)

            # If too much time has passed then end the trial
            if not stream.isPlaying: #stream.status == FINISHED:
                response = "NA"
                responseTime = 0
                continueTrial = False
                
            # Keep checking for if a button is pressed
            if sameBttn.status == STARTED:

                if myMouse.isPressedIn(sameBttn):
                    response = "same"
                    responseTime = tNow
                    continueTrial = False
                elif myMouse.isPressedIn(diffBttn):
                    response = "different"
                    responseTime = tNow
                    continueTrial = False
                
            # Wait for response. If <esc> pressed then exit task. If click train ends then end the trial
            keysPressed = kb.getKeys(keyList=["escape"])
            if keysPressed == ["escape"]:
                close_ttl()
                win.close()
                thisExp.saveAsWideText(filename+'.csv', delim='auto')
                thisExp.saveAsPickle(filename)
                logging.flush()
                thisExp.abort()
                core.quit()
                
            win.flip()


        stream.stop()
        
        # Display feedback
        if response == correctResponse:
            txtObj = correctResponseText
        elif response != correctResponseText and responseTime != 0:
            txtObj = incorrectResponseText
        else:
            txtObj = noResponseText

        txtObj.setAutoDraw(True)
        crossFixation.setAutoDraw(False)
        sameBttn.setAutoDraw(False)
        diffBttn.setAutoDraw(False)

        tEnd = tNow + 3
        while tNow < tEnd:
            tNow = core.getTime()
            win.flip()

        txtObj.setAutoDraw(False)
        crossFixation.setAutoDraw(True)

        # Append trial info
        trials.addData('audio_onset', soundOnset)
        trials.addData('response_time', responseTime)
        trials.addData('response', response)
        thisExp.nextEntry()
    
    # Task over. Close everything
    close_ttl()
    win.close()
    thisExp.saveAsWideText(filename+'.csv', delim='auto')
    thisExp.saveAsPickle(filename)
    logging.flush()
    thisExp.abort()
    core.quit()

if __name__ == '__main__':
    run()
    
