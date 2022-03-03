import cv2 as cv
from pathlib import Path
import argparse
import pandas as pd


video_annotations = pd.read_csv(f"C:/annotations.csv", delimiter=';')

parser = argparse.ArgumentParser(description="Choose Video.")
parser.add_argument('video', type=int)
parser.add_argument('--phase', type=int)

#args = parser.parse_args()

video_file = str(Path("C:/Users/Jörn/Documents/FH/BA/cataract-101/videos/case_{}.mp4".format(270)))
capture = cv.VideoCapture(video_file)

video_annotations = video_annotations.loc[video_annotations['VideoID'] == 270]

phases = ["Incision(1)", "ViscousAgentInjection(2)", "Rhexis(3)", "Hydrodissection(4)", "Phacoemulsification(5)",
          "IrrigationAndAspiration(6)", "CapsulePolishing(7)", "LensImplantSettingUp(8)", "ViscousAgentRemoval(9)",
          "TonifyingAndAntibiotics(10)"]


def grab_all_frames():
    print(video_annotations)
    ret = True
    count =  0
    i = video_annotations.first_valid_index()
    f = video_annotations.last_valid_index()
    first_start_frame = video_annotations.loc[i].at['FrameNo']
    last_start_frame = video_annotations.loc[f].at['FrameNo']
    print("::: [INFO] Surgery begins... :::")
    while ret == True:
        ret, frame = capture.read()
        if count >= last_start_frame:
            if (count%200)==0:
                print(":::[INFO] Last Frames in Video of Phase %d... :::" % video_annotations.loc[i].at['Phase'])
            try:
                cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[video_annotations.loc[i].at['Phase']-1]) % (270,count), frame)
            except cv.error:
                continue
        while count < last_start_frame:
            if count >= first_start_frame:
                if count == video_annotations.loc[i+1].at['FrameNo']:
                    i += 1
                ret, frame = capture.read()
                if (count%200) == 0:
                    print("::: [INFO] Writing Frame of Phase %d :::" % video_annotations.loc[i].at['Phase'])
                cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[video_annotations.loc[i].at['Phase']-1]) % (270,count), frame)
            count += 1
        count += 1 
    

def grab_phase_frames(p, count, ret):
    phase_list = video_annotations.loc[video_annotations['Phase'] == p]
    print(phase_list)
    indices = phase_list.index.tolist()
    print(indices)
    print("::: [INFO] Surgery begins... :::")
    for i in range(len(indices)):
        try:
            while ret == True and count < video_annotations.loc[indices[i]+1].at['FrameNo']:
                print("(Phase, Count): (%d,%d)" % (p,count))
                ret, frame = capture.read()
                if ret == False:
                    print("read() failed, count: ", count)
                    input()
                if count >= video_annotations.loc[indices[i]].at['FrameNo']:
                    if(count%200) == 0:
                        print("::: [INFO] Writing Frames of Phase %d :::" % p)
                    cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[p-1]) % (270,count), frame)
                count += 1
        except KeyError:
            print("::: [INFO] Caught KeyError, last Frames of Video... :::")
            while ret == True:
                ret, frame = capture.read()
                if(count%200) == 0:
                    print("::: [INFO] Writing Frames of Phase %d :::" % p)
                    print("Index: ", indices[i])
                try:
                    cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[p-1]) % (270,count), frame)
                except cv.error:
                    count += 1
                    continue
                count += 1
    return video_annotations.loc[indices[0]+1].at['FrameNo']
    

    
            
                
if True:
    req_phases = [1,2,3,4,5,7,8,9]
    count = 0
    ret = True
    for i in range(len(req_phases)):
        temp = grab_phase_frames(req_phases[i], count, ret)
        count = temp
    input("Done, press Enter to leave...")
    exit()

grab_all_frames()
input("Done, press Enter to leave...")    
            