import cv2 as cv
from pathlib import Path
import argparse
import pandas as pd


video_annotations = pd.read_csv(f"C:/annotations.csv", delimiter=';')

parser = argparse.ArgumentParser(description="Choose Video.")
parser.add_argument('video', type=int)
parser.add_argument('--phase', type=int)

args = parser.parse_args()

video_file = str(Path("C:/Users/Jörn/Documents/FH/BA/cataract-101/videos/case_{}.mp4".format(args.video)))
capture = cv.VideoCapture(video_file)

video_annotations = video_annotations.loc[video_annotations['VideoID'] == args.video]
print(video_annotations)

phases = ["Incision(1)", "ViscousAgentInjection(2)", "Rhexis(3)", "Hydrodissection(4)", "Phacoemulsification(5)",
          "IrrigationAndAspiration(6)", "CapsulePolishing(7)", "LensImplantSettingUp(8)", "ViscousAgentRemoval(9)",
          "TonifyingAndAntibiotics(10)"]


def grab_all_frames():
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
            cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[video_annotations.loc[i].at['Phase']-1]) % (args.video,count), frame)
        while count < last_start_frame:
            if count >= first_start_frame:
                if count == video_annotations.loc[i+1].at['FrameNo']:
                    i += 1
                ret, frame = capture.read()
                if (count%200) == 0:
                    print("::: [INFO] Writing Frame of Phase %d :::" % video_annotations.loc[i].at['Phase'])
                cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[video_annotations.loc[i].at['Phase']-1]) % (args.video,count), frame)
            count += 1
        count += 1 
    

def grab_phase_frames():
    count = 0
    ret = True
    phase_list = video_annotations.loc[video_annotations['Phase'] == args.phase]
    indices = phase_list.index.tolist()
    print("::: [INFO] Surgery begins... :::")
    for i in range(len(indices)):
        while ret == True and count < video_annotations.loc[indices[i]+1].at['FrameNo']:
            ret, frame = capture.read()
            if count >= video_annotations.loc[indices[i]].at['FrameNo']:
                if(count%200) == 0:
                    print("::: [INFO] Writing Frames of Phase %d :::" % args.phase)
                cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[args.phase-1]) % (args.video,count), frame)
            count += 1
    
    
if args.phase:
    grab_phase_frames()
    input("Done, press Enter to leave...")
    exit()

grab_all_frames()
input("Done, press Enter to leave...")    
            