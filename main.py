"""
TAMy
Auditory Memory Task

Noah Markowitz
Human Brain Mapping Laboratory
North Shore University Hospital
June 2023
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
from psychtoolbox import GetSecs, WaitSecs
from psychopy import core, event, visual, sound, logging, gui
from psychopy.hardware import keyboard
#from psychopy.tools import environmenttools
from psychopy.data import ExperimentHandler, TrialHandler2
from psychopy.constants import NOT_STARTED, STARTED, FINISHED
import tobii_research as tobii


# Ensure that relative paths start from the same directory as this script
_thisDir = op.dirname(op.abspath(__file__))
logDir = op.join(_thisDir, 'logs')
stimDir = op.join(_thisDir, 'stimuli')
from utils import openingDlg, set_ttl, createAudioStream, setScreen, read_wav, createToneReps, pauseAndReadText, generate_tone_sequence
from settings import SETTINGS

# Callback function for tobii eyetracker
def gaze_callback(gazedata):
    cdata = np.empty((1,33))
    #cdata[0,0] = gazedata._GazeData__device_time_stamp / 1000000
    #cdata[0,1] = gazedata._GazeData__system_time_stamp / 1000000
    cdata[0,0] = core.getTime()
    cdata[0,1] = gazedata._GazeData__device_time_stamp
    cdata[0,2] = gazedata._GazeData__system_time_stamp
    cdata[0,3] = gazedata._GazeData__left._EyeData__gaze_origin._GazeOrigin__position_in_track_box_coordinates[0]
    cdata[0,4] = gazedata._GazeData__left._EyeData__gaze_origin._GazeOrigin__position_in_track_box_coordinates[1]
    cdata[0,5] = gazedata._GazeData__left._EyeData__gaze_origin._GazeOrigin__position_in_track_box_coordinates[2]
    cdata[0,6] = gazedata._GazeData__left._EyeData__gaze_origin._GazeOrigin__position_in_user_coordinates[0]
    cdata[0,7] = gazedata._GazeData__left._EyeData__gaze_origin._GazeOrigin__position_in_user_coordinates[1]
    cdata[0,8] = gazedata._GazeData__left._EyeData__gaze_origin._GazeOrigin__position_in_user_coordinates[2]
    cdata[0,9] = gazedata._GazeData__left._EyeData__gaze_origin._GazeOrigin__validity
    cdata[0,10] = gazedata._GazeData__left._EyeData__gaze_point._GazePoint__position_in_user_coordinates[0]
    cdata[0,11] = gazedata._GazeData__left._EyeData__gaze_point._GazePoint__position_in_user_coordinates[1]
    cdata[0,12] = gazedata._GazeData__left._EyeData__gaze_point._GazePoint__position_in_user_coordinates[2]
    cdata[0,13] = gazedata._GazeData__left._EyeData__gaze_point._GazePoint__position_on_display_area[0]
    cdata[0,14] = gazedata._GazeData__left._EyeData__gaze_point._GazePoint__position_on_display_area[1]
    cdata[0,15] = gazedata._GazeData__left._EyeData__gaze_point._GazePoint__validity
    cdata[0,16] = gazedata._GazeData__left._EyeData__pupil_data._PupilData__diameter
    cdata[0,17] = gazedata._GazeData__left._EyeData__pupil_data._PupilData__validity
    cdata[0,18] = gazedata._GazeData__right._EyeData__gaze_origin._GazeOrigin__position_in_track_box_coordinates[0]
    cdata[0,19] = gazedata._GazeData__right._EyeData__gaze_origin._GazeOrigin__position_in_track_box_coordinates[1]
    cdata[0,20] = gazedata._GazeData__right._EyeData__gaze_origin._GazeOrigin__position_in_track_box_coordinates[2]
    cdata[0,21] = gazedata._GazeData__right._EyeData__gaze_origin._GazeOrigin__position_in_user_coordinates[0]
    cdata[0,22] = gazedata._GazeData__right._EyeData__gaze_origin._GazeOrigin__position_in_user_coordinates[1]
    cdata[0,23] = gazedata._GazeData__right._EyeData__gaze_origin._GazeOrigin__position_in_user_coordinates[2]
    cdata[0,24] = gazedata._GazeData__right._EyeData__gaze_origin._GazeOrigin__validity
    cdata[0,25] = gazedata._GazeData__right._EyeData__gaze_point._GazePoint__position_in_user_coordinates[0]
    cdata[0,26] = gazedata._GazeData__right._EyeData__gaze_point._GazePoint__position_in_user_coordinates[1]
    cdata[0,27] = gazedata._GazeData__right._EyeData__gaze_point._GazePoint__position_in_user_coordinates[2]
    cdata[0,28] = gazedata._GazeData__right._EyeData__gaze_point._GazePoint__position_on_display_area[0]
    cdata[0,29] = gazedata._GazeData__right._EyeData__gaze_point._GazePoint__position_on_display_area[1]
    cdata[0,30] = gazedata._GazeData__right._EyeData__gaze_point._GazePoint__validity
    cdata[0,31] = gazedata._GazeData__right._EyeData__pupil_data._PupilData__diameter
    cdata[0,32] = gazedata._GazeData__right._EyeData__pupil_data._PupilData__validity
    global ETdata
    ETdata = np.append(ETdata,cdata, axis=0)

def ETvalidation(win,eyetracker,etFrequency):
    # just give it the window and eyetracker objects created in the main script as well as the frequency that the eyetracker is set to
    # get an output variable from this which will tell you if you should continue the experiment after validation
    # TO DO: still needs to capture the time stamps of the validation points first being shown and return them
    
    continueValidation = True
    continueExp = True
    validationPoints = np.array([[-.25,-.25],[.25,-.25],[-.25,.25],[.25,.25]])*win.size
    
    point = visual.Circle(
        win=win, name='point',
#        win.size[0]*0.00003, win.size[0]*0.00003
        size=(15,15), pos=(0, 0), lineWidth=1.0, colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True, units='pix')
    circle = visual.Circle(
        win=win, name='point',
        size=(0, 0), pos=(0, 0), lineWidth=5.0, colorSpace='rgb', lineColor='white', fillColor=None,
        opacity=None, depth=0.0, interpolate=True, units='pix')
    
    eyedotsLeft = np.empty((0,2))
    eyedotsRight = np.empty((0,2))
    times = np.empty((4,2))
    eyetracker.subscribe_to(tobii.EYETRACKER_GAZE_DATA,gaze_callback)
    
    Inst = visual.TextStim(win = win, units = 'norm', height = 0.1,
                pos = (0,0), text = 'Look at the dots!', alignHoriz = 'center',
                alignVert = 'center', color = 'white', wrapWidth=1.5, autoLog=False)
    Inst.draw()
    win.flip()
    core.wait(2)
    
    for i in range(validationPoints.shape[0]):
        point.setPos(validationPoints[i,:])
        circle.setPos(validationPoints[i,:])
        tStart = core.getTime()
        tStartET = ETdata[-1,0]
        tNow = core.getTime()
        point.setAutoDraw(True)
        circle.setAutoDraw(True)
        while tNow <= tStart + 2:
            size = 100*(1-(tNow-tStart)*0.5)
            if size < 0:
                size = 0
            circle.setSize([size,size])
            win.flip()
            tNow = core.getTime()
        eyedotsLeft = np.append(eyedotsLeft,ETdata[ETdata.shape[0]-int(1.5*etFrequency):ETdata.shape[0],13:15],axis=0)
        eyedotsRight = np.append(eyedotsRight,ETdata[ETdata.shape[0]-int(1.5*etFrequency):ETdata.shape[0],28:30],axis=0)
        times[i,0] = tStart
        times[i,1] = tStartET
        core.wait(0.4)
    point.setAutoDraw(False)
    circle.setAutoDraw(False)
    eyetracker.unsubscribe_from(tobii.EYETRACKER_GAZE_DATA)
    
    # average samples
    avgLeft = np.empty((0,2))
    avgRight = np.empty((0,2))
    for i in range(int(eyedotsLeft.shape[0]/50)):
        sample = eyedotsLeft[i*50:(i+1)*50]
        avgLeft = np.append(avgLeft,np.reshape(np.nanmean(sample,axis=0),(1,2)),axis=0)
        sample = eyedotsRight[i*50:(i+1)*50]
        avgRight = np.append(avgRight,np.reshape(np.nanmean(sample,axis=0),(1,2)),axis=0)
    
    # show validation results
    dotsRight = visual.ElementArrayStim(win=win,
                        name="dotsRight",
                        nElements=avgRight.shape[0],
                        colors='blue',
                        sizes=[10,10],
                        xys=(avgRight-0.5)*win.size,
                        elementMask='circle',
                        elementTex=None,
                        units='pix')
    dotsLeft = visual.ElementArrayStim(win=win,
                        name="dotsLeft",
                        nElements=avgLeft.shape[0],
                        colors='green',
                        sizes=[10,10],
                        xys=(avgLeft-0.5)*win.size,
                        elementMask='circle',
                        elementTex=None,
                        units='pix')
    allValidationPoints = visual.ElementArrayStim(win=win,
                        name="validationPoints",
                        nElements=4,
                        colors='white',
                        sizes=[10,10],
                        xys=validationPoints,
                        elementMask='circle',
                        elementTex=None,
                        units='pix')

    validationText = visual.TextStim(win = win, units = 'norm', height = 0.05,
                    pos = (0,0), text = 'press spacebar to accept or esc to abort', alignHoriz = 'center',
                    alignVert = 'center', color = 'white', wrapWidth=1.5, autoLog=False)
    validationText.setAutoDraw(True)
    keepwaiting = True
    while keepwaiting:
        dotsLeft.draw()
        dotsRight.draw()
        allValidationPoints.draw()
        keyPressed = event.getKeys(keyList=['space', 'escape'])
        if any(keyPressed):
            if keyPressed[0]=='space':
                keepwaiting = False
                validationText.setAutoDraw(False)
            if keyPressed[0]=='escape':
                keepwaiting = False
                validationText.setAutoDraw(False)
                continueExp=False
        win.flip()
    win.flip()
    return continueExp, times
    
# Main function to run experiment
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
    
    # Calibrate eyetracker
    if expInfo['eyetracker'] != 'None':
        eyetracker = tobii.find_all_eyetrackers()[0]
        eyetracker.set_gaze_output_frequency(expInfo['eyetracker'])
        tracker_manager_path = 'C:/Users/HBML/AppData/Local/Programs/TobiiProEyeTrackerManager/'
        serial_number = 'TPSP1-010202818635'
        mode = 'usercalibration'
        os.system('{}TobiiProEyeTrackerManager.exe --device-sn={} --mode={}'.format(tracker_manager_path, serial_number, mode))
        
    # Setup the Window, Keyboard, Mouse
    win, mon = setScreen(
        expInfo['screen_resolution'], expInfo['monitor_width'], expInfo['full_screen'], expInfo['monitor'],
        color='black')
    kb = keyboard.Keyboard()
    mouse = event.Mouse(win=win)
    mouse.setVisible(1)

    # Display a cross in the middle of the screen
    crossFixation = visual.ShapeStim(
        win=win, name='crossFixation', vertices='cross',
        size=(100,100),#(0.1,0.1), 
        units='pix',#'norm',
        ori=0, pos=(0, 0),
        lineWidth=0, lineColor='white', lineColorSpace='rgb',
        fillColor='white', fillColorSpace='rgb', opacity=1, interpolate=True)
    
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
        method='random',
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

    # Textboxes
    sameBox = visual.Rect(
        win=win, name='sameBox',
        units='norm', 
        width=(0.3, 0.3)[0], 
        height=(0.3, 0.3)[1],
#        units='pix',
#        width=400,
#        height=400,
        ori=0.0, 
        pos=(-0.5, 0), 
        # pos=(-864,558),
        #anchor='center',
        lineWidth=1.0, colorSpace='rgb',  lineColor='green', fillColor='green',
        opacity=None, depth=0.0, interpolate=True)
    diffBox = visual.Rect(
        win=win, name='diffBox',
        units='norm', 
        width=(0.3, 0.3)[0], 
        height=(0.3, 0.3)[1],
#        units='pix',
#        width=400,
#        height=400,
        ori=0.0, 
        pos=(0.5, 0), 
        #pos=(864,558),
        #anchor='center',
        lineWidth=1.0, colorSpace='rgb',  lineColor='red', fillColor='red',
        opacity=None, depth=-1.0, interpolate=True)
    sameText = visual.TextStim(win=win, name='sameText',
        text='Same',
        font='Open Sans',
        units='norm', pos=(-0.5, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0)
    diffText = visual.TextStim(win=win, name='diffText',
        text='Different',
        font='Open Sans',
        units='norm', pos=(0.5, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0)
        
    # get limits of the boxes in tobii coordinates
    sameBoxLims = np.empty((4,1))
    sameBoxLims[0] = (((sameBox.pos[0]-sameBox.width/2)/2)+0.5)
    sameBoxLims[1] = ((sameBox.pos[0]+sameBox.width/2)/2)+0.5
    sameBoxLims[2] = ((sameBox.pos[1]+sameBox.height/2)/-2)+0.5
    sameBoxLims[3] = ((sameBox.pos[1]-sameBox.height/2)/-2)+0.5
    diffBoxLims = np.empty((4,1))
    diffBoxLims[0] = ((diffBox.pos[0]-diffBox.width/2)/2)+0.5
    diffBoxLims[1] = ((diffBox.pos[0]+diffBox.width/2)/2)+0.5
    diffBoxLims[2] = ((diffBox.pos[1]+diffBox.height/2)/-2)+0.5
    diffBoxLims[3] = ((diffBox.pos[1]-diffBox.height/2)/-2)+0.5


    # Initiate audio
    stream = sound.Sound(name='trial_audio', sampleRate=globalFs, stereo=True, syncToWin=win)

    #mouse.status = STARTED

    # Create audio clicks
    clickDur = SETTINGS["click_dur"]
    clickSOA = SETTINGS["click_soa"]
    trlDur = 60
    clickReps = np.floor(trlDur/clickSOA).astype(int)
    singleClick = np.ones( ( round(clickDur*globalFs),) )
    clickStream = createAudioStream(singleClick,clickSOA,globalFs,clickReps)
    choice2cue_clickStream = createAudioStream(singleClick,clickSOA,globalFs,5)
    
    # Initiate Eyetracker
    if expInfo['eyetracker'] != 'None':
        ETdataFilePath = filename + '_et.csv'
        # initiate data frame for eyetracker data
        # ETcolumns = 'deviceTimeStampInSec,systemTimeStampInSec,xLeftGazeOriginInTrackboxCoords,yLeftGazeOriginInTrackboxCoords,zLeftGazeOriginInTrackboxCoords,xLeftGazeOriginInUserCoords,yLeftGazeOriginInUserCoords,zLeftGazeOriginInUserCoords,leftGazeOriginValidity,xLeftGazePositionInUserCoords,yLeftGazePositionInUserCoords,zLeftGazePositionInUserCoords,xLeftGazePositionOnDisplay,yLeftGazePositionOnDisplay,leftGazePointValidity,leftPupilDiameter,leftPupilValidity,xRightGazeOriginInTrackboxCoords,yRightGazeOriginInTrackboxCoords,zRightGazeOriginInTrackboxCoords,xRightGazeOriginInUserCoords,yRightGazeOriginInUserCoords,zRightGazeOriginInUserCoords,rightGazeOriginValidity,xRightGazePositionInUserCoords,yRightGazePositionInUserCoords,zRightGazePositionInUserCoords,xRightGazePositionOnDisplay,yRightGazePositionOnDisplay,rightGazePointValidity,rightPupilDiameter,rightPupilValidity'
        ETcolumns = 'expTime,deviceTimeStamp,systemTimeStamp,xLeftGazeOriginInTrackboxCoords,yLeftGazeOriginInTrackboxCoords,zLeftGazeOriginInTrackboxCoords,xLeftGazeOriginInUserCoords,yLeftGazeOriginInUserCoords,zLeftGazeOriginInUserCoords,leftGazeOriginValidity,xLeftGazePositionInUserCoords,yLeftGazePositionInUserCoords,zLeftGazePositionInUserCoords,xLeftGazePositionOnDisplay,yLeftGazePositionOnDisplay,leftGazePointValidity,leftPupilDiameter,leftPupilValidity,xRightGazeOriginInTrackboxCoords,yRightGazeOriginInTrackboxCoords,zRightGazeOriginInTrackboxCoords,xRightGazeOriginInUserCoords,yRightGazeOriginInUserCoords,zRightGazeOriginInUserCoords,rightGazeOriginValidity,xRightGazePositionInUserCoords,yRightGazePositionInUserCoords,zRightGazePositionInUserCoords,xRightGazePositionOnDisplay,yRightGazePositionOnDisplay,rightGazePointValidity,rightPupilDiameter,rightPupilValidity'        
        global ETdata
        ETdata = np.empty((0,33))
        with open(ETdataFilePath, 'ab') as csvfile:
            np.savetxt(csvfile,ETdata,delimiter=',',header=ETcolumns)

    win.flip()
    
    # Do eyetracker validation
    if expInfo['eyetracker'] != 'None':
        continueExperiment, validationTimes = ETvalidation(win,eyetracker,int(expInfo['eyetracker']))
        ETvalidationFilePath = filename + '_etValidationTimes.csv'
        with open(ETvalidationFilePath, 'w') as csvfile:
            np.savetxt(csvfile,validationTimes,delimiter=',',header='PsychoPyTime,ETtime')
        if not continueExperiment:
            thisExp.abort()
    
    # Show instructions
    instructions = "Two tones will play. After the second tone, decide whether the two tones are the same or different. If the two tones are the same <fill1>. If the sounds are different then <fill2>.\nPress <space> to start"
    if expInfo['responseType'] == 'eyetracker':
        instructions = instructions.replace("<fill1>", "look at the 'Same' box on the screen") 
        instructions = instructions.replace("<fill2>", "look at the 'Different' box on the screen") 
    elif expInfo['responseType'] == 'keyboard':
        instructions = instructions.replace("<fill1>", "press the 'C' key on the keyboard") 
        instructions = instructions.replace("<fill2>", "press the 'M' key on the keyboard") 
    elif expInfo['responseType'] == 'mouse':
        instructions = instructions.replace("<fill1>", "click the 'Same' box on the screen") 
        instructions = instructions.replace("<fill2>", "clickthe 'Different' box on the screen") 
    
    key = pauseAndReadText(win, instructions, mouse=None, txtColor=[1, 1, 1], keys=['space', 'escape'], wait=0)
    if key == 'escape':
        close_ttl()
        win.close()
        if expInfo['eyetracker']!='None':
            eyetracker.unsubscribe_from(tobii.EYETRACKER_GAZE_DATA)
        thisExp.abort()
        core.quit()
    
# Start trials
    for thisTrial in trials:

        crossFixation.setAutoDraw(True)
        
        # Trial variables
        response = 'NA'
        continueTrial = True
        responseStarted = False
        prevButtonState = mouse.getPressed()
    
        # Load sound files (when they're wav files)
        #cuesoundFile = thisTrial['cuesound']
        #cueSound = read_wav(op.join(stimDir, cuesoundFile), new_fs=globalFs)
        #choicesoundFile = thisTrial['choicesound']
        #choiceSound = read_wav(op.join(stimDir, choicesoundFile), new_fs=globalFs)
        
        # Create sounds (tones)
        #cuesound = thisTrial["cuesound"]
        #cueSound = createToneReps(value=cuesound, tone_dur=SETTINGS['tone_dur'], blank_dur=SETTINGS['tone_blank_dur'], reps=SETTINGS['tone_reps'], sampleRate=globalFs)
        #choicesound = thisTrial["choicesound"]
        #choiceSound = createToneReps(value=choicesound, tone_dur=SETTINGS['tone_dur'], blank_dur=SETTINGS['tone_blank_dur'], reps=SETTINGS['tone_reps'], sampleRate=globalFs)
        
        # Create sounds (frequencies)
        cuesound_id = str(thisTrial["cue_frequency"]) + "_" + str(thisTrial["cue_frequency_range"])
        cueSound = generate_tone_sequence(
            coherence=0.9, 
            frequency=thisTrial["cue_frequency"],
            frequency_range=thisTrial["cue_frequency_range"],
            sampleRate=globalFs)
        choicesound_id = str(thisTrial["choice_frequency"]) + "_" + str(thisTrial["choice_frequency_range"])
        choiceSound = generate_tone_sequence(
            coherence=0.9, 
            frequency=thisTrial["choice_frequency"],
            frequency_range=thisTrial["choice_frequency_range"],
            sampleRate=globalFs)
        #arr = generate_tone_sequence(coherence=0.9, frequency=4000, frequency_range=1, sampleRate=44100)
        
        # Check what the correct response to this trial should be
        if cuesound_id == choicesound_id: #cuesound == choicesound:
            correctResponse = "same"
        else:
            correctResponse = "diff"
        
        # Create audio stream. Embed choiceSound within the click train
        audStream = np.concatenate((cueSound, choice2cue_clickStream, choiceSound, clickStream))

        # When response can start to be made
        responseStartTime = np.concatenate((cueSound, choice2cue_clickStream, choiceSound)).shape[0]/globalFs

        # Intertrial interval wait time
        thisTrialITI = randint(SETTINGS['iti'][0]*1000, high=SETTINGS['iti'][1]*1000)/1000

        # Set auditory stimulus
        stream.setSound(audStream)

        # keep track of which components have finished
        trialComponents = [sameBox, diffBox, sameText, diffText, stream]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

        # Start eyetracker
        if expInfo['eyetracker']!='None':
            WaitSecs(1)
            thisTrialITI -= 1
            eyetracker.subscribe_to(tobii.EYETRACKER_GAZE_DATA,gaze_callback)
            
        # Start playing auditory stimulus
        #stream.play(when=GetSecs()+thisTrialITI)
        tNow = core.getTime()
        #tStartAudio = GetSecs()+thisTrialITI
        tStartAudio = tNow+thisTrialITI
        
        # When audio starts to play send a TTL
        keep_going = True
        while keep_going:
            tNow = core.getTime()
            tNextFlip = win.getFutureFlipTime(clock=None)
            
            if stream.status == STARTED:
                keep_going = False
            
            if tNextFlip >= tStartAudio and stream.status == NOT_STARTED:
                stream.play(when=win)
                win.callOnFlip(send_ttl, ttl_code)
                win.timeOnFlip(stream, 'tStartRefresh')
                
            if keep_going:
                win.flip()
            
#        while stream.status == NOT_STARTED:
#            #win.flip()
#            pass
#        soundOnset = GetSecs() #core.getTime()
#        send_ttl(ttl_code)

        # When to allow responses to start to appear
        soundOnset = stream.tStartRefresh
        tAllowResponse = soundOnset + responseStartTime
        
        while continueTrial:

            # Current time
            tNow = core.getTime()
            tNextFlip = win.getFutureFlipTime(clock=None)
            
            # Check if it's time to start audio
            if tNextFlip >= tStartAudio and stream.status == NOT_STARTED:
                stream.play(when=win)
                win.callOnFlip(send_ttl, ttl_code)
                win.timeOnFlip(stream, 'tStartRefresh')
            
            # Present two images/shape representing the choices when time is right
            if tNextFlip >= tAllowResponse and not responseStarted:
                sameBox.setAutoDraw(True)
                sameText.setAutoDraw(True)
                diffBox.setAutoDraw(True)
                diffText.setAutoDraw(True)
                responseStarted = True
                win.timeOnFlip(sameBox, 'tStartRefresh')

            # If too much time has passed then end the trial
            if stream.status == FINISHED:
                response = "NA"
                responseTime = 0
                continueTrial = False
                stream.tStopRefresh = tNow
                
            # Check for pressed keys on keyboard
            keysPressed = kb.getKeys(keyList=["escape","c","m"])
            
            if expInfo['responseType'] == 'saccade':
                # check for target fixation
                fix = SETTINGS['response_fixation_time']
                if expInfo['eyetracker'] != 'None' and sameBox.status == STARTED:
                    if ETdata.shape[0]>(fix*int(expInfo['eyetracker'])):
                        # get the last gaze positions
                        index = ETdata.shape[0]-1
                        xleft = ETdata[(index-int(fix*int(expInfo['eyetracker']))):index,13]
                        yleft = ETdata[index-int(fix*int(expInfo['eyetracker'])):index,14]
                        xright = ETdata[index-int(fix*int(expInfo['eyetracker'])):index,28]
                        yright = ETdata[index-int(fix*int(expInfo['eyetracker'])):index,29]

                        # Check if eyes are in the box indicating "same"
                        isInSameBox = np.nanmean(xleft)>=sameBoxLims[0]\
                        and np.nanmean(xleft)<=sameBoxLims[1]\
                        and np.nanmean(xright)>=sameBoxLims[0]\
                        and np.nanmean(xright)<=sameBoxLims[1]\
                        and np.nanmean(yleft)>=sameBoxLims[2]\
                        and np.nanmean(yleft)<=sameBoxLims[3]\
                        and np.nanmean(yright)>=sameBoxLims[2]\
                        and np.nanmean(yright)<=sameBoxLims[3]
                        
                        # Check if eyes are in the box indicating "different"
                        isInDiffBox = np.nanmean(xleft)>=diffBoxLims[0]\
                        and np.nanmean(xleft)<=diffBoxLims[1]\
                        and np.nanmean(xright)>=diffBoxLims[0]\
                        and np.nanmean(xright)<=diffBoxLims[1]\
                        and np.nanmean(yleft)>=diffBoxLims[2]\
                        and np.nanmean(yleft)<=diffBoxLims[3]\
                        and np.nanmean(yright)>=diffBoxLims[2]\
                        and np.nanmean(yright)<=diffBoxLims[3]
                        
                        if isInSameBox:
                            responseTime = tNow - fix
                            response = 'same'
                            continueTrial = False
                            win.callOnFlip(stream.stop)
                            win.timeOnFlip(stream, 'tStopRefresh')
                        elif isInDiffBox:
                            responseTime = tNow - fix
                            response = 'diff'
                            continueTrial = False
                            win.callOnFlip(stream.stop)
                            win.timeOnFlip(stream, 'tStopRefresh')
            
            # Look for mouse button press
            if expInfo['responseType'] == 'mouse' and sameBox.status == STARTED:
                # Keep checking for if a button is pressed
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        #clickableList = environmenttools.getFromNames([sameBox, diffBox, sameText, diffText], namespace=locals
                        clickableList = [sameBox, diffBox, sameText, diffText]
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse):
                                gotValidClick = True
                                objPressed = obj.name
                                response = objPressed[0:4]
                                responseTime = tNow
                        if gotValidClick:
                            continueTrial = False
                            win.callOnFlip(stream.stop)
                            win.timeOnFlip(stream, 'tStopRefresh')
            
            # If keyboard if used for response
            elif expInfo['responseType'] == 'keyboard' and sameBox.status == STARTED:
                if keysPressed == ["c"]:
                    response = "same"
                    responseTime = tNow
                    continueTrial = False
                    win.callOnFlip(stream.stop)
                    win.timeOnFlip(stream, 'tStopRefresh')
                elif keysPressed == ["m"]:
                    response = "diff"
                    responseTime = tNow
                    continueTrial = False
                    win.callOnFlip(stream.stop)
                    win.timeOnFlip(stream, 'tStopRefresh')
                    
            # If <esc> pressed then exit task. If click train ends then end the trial
            if keysPressed == ["escape"]:
                close_ttl()
                win.close()
                thisExp.saveAsWideText(filename+'.csv', delim='auto')
                thisExp.saveAsPickle(filename)
                logging.flush()
                if expInfo['eyetracker']!='None':
                    with open(ETdataFilePath, 'ab') as csvfile:
                        np.savetxt(csvfile,ETdata,delimiter=',')
                    eyetracker.unsubscribe_from(tobii.EYETRACKER_GAZE_DATA)
                thisExp.abort()
                core.quit()
                        
            if continueTrial:
                win.flip()


        #stream.stop()
        
        # Display feedback
        if response == correctResponse:
            txtObj = correctResponseText
        elif response != correctResponseText and responseTime != 0:
            txtObj = incorrectResponseText
        else:
            txtObj = noResponseText

        # Draw feedback to screen and nothing else
        txtObj.setAutoDraw(True)
        txtObj.tStartRefresh = None
        crossFixation.setAutoDraw(False)
        sameBox.setAutoDraw(False)
        sameText.setAutoDraw(False)
        diffBox.setAutoDraw(False)
        diffText.setAutoDraw(False)
        tEnd = tNow + 3
        continueFeedback = True
        win.timeOnFlip(txtObj, 'tStartRefresh')
        while continueFeedback:
            tNow = core.getTime()
            if tEnd <= tNow:
                continueFeedback = False
            else:
                win.flip()
            
        # unsubscribe from eyetracker
        if expInfo['eyetracker']!='None':
            eyetracker.unsubscribe_from(tobii.EYETRACKER_GAZE_DATA)
            with open(ETdataFilePath, 'ab') as csvfile:
                np.savetxt(csvfile,ETdata,delimiter=',')
            # re-initiate data frame for eyetracker data
            ETdata = np.empty((0,33))
        
        # Stop showing feedbadk and bring back the cross
        txtObj.setAutoDraw(False)
        crossFixation.setAutoDraw(True)

        # Append trial info
        thisExp.addData('audio_onset', stream.tStartRefresh)
        thisExp.addData('audio_offset', stream.tStopRefresh)
        thisExp.addData('display_feedback', txtObj.tStartRefresh)
        thisExp.addData('response_time', responseTime)
        thisExp.addData('response', response)
        thisExp.nextEntry()
    
    # Task over. Close everything
    close_ttl()
    win.close()
    thisExp.saveAsWideText(filename+'.csv', delim='auto')
    thisExp.saveAsPickle(filename)
    logging.flush()
    if expInfo['eyetracker']!='None':
        with open(ETdataFilePath, 'ab') as csvfile:
            np.savetxt(csvfile,ETdata,delimiter=',')
        eyetracker.unsubscribe_from(tobii.EYETRACKER_GAZE_DATA)
    thisExp.abort()
    core.quit()

if __name__ == '__main__':
    run()
    
