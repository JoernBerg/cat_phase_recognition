from cmath import phase
import cv2 as cv
from pathlib import Path
import argparse
import pandas as pd


video_annotations = pd.read_csv(f"C:/annotations.csv", delimiter=';')

#parser = argparse.ArgumentParser(description="Choose Video.")
#parser.add_argument('video', type=int)
#parser.add_argument('--phase', type=int)

#args = parser.parse_args()

#video_file = str(Path("C:/Users/Jörn/Documents/FH/BA/cataract-101/videos/case_{}.mp4".format(args.video)))
#capture = cv.VideoCapture(video_file)
#frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

#video_annotations = video_annotations.loc[video_annotations['VideoID'] == args.video]

phases = ["1_Incision", "2_ViscousAgentInjection", "3_Rhexis", "4_Hydrodissection", "5_Phacoemulsification",
          "6_IrrigationAndAspiration", "7_CapsulePolishing", "8_LensImplantSettingUp", "9_ViscousAgentRemoval",
          "X_TonifyingAndAntibiotics"]


def grab_all_frames(video_annotations, frame_count, video):
    index = video_annotations.first_valid_index()
    first_start_frame = video_annotations.loc[index].at['FrameNo']
    last_start_frame = video_annotations.loc[video_annotations.last_valid_index()].at['FrameNo']
    count = first_start_frame
    capture.set(cv.CAP_PROP_POS_FRAMES, count)
    ret = True
    while ret == True and count <= frame_count:
        ret, frame = capture.read()
        while count < last_start_frame:
            if count == video_annotations.loc[index+1].at['FrameNo']:
                index += 1
            ret, frame = capture.read()
            if (count%200) == 0:
                print("::: [INFO] Writing Frames of Phase %d :::" % video_annotations.loc[index].at['Phase'])
            cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[video_annotations.loc[index].at['Phase']-1]) % (video,count), frame)
            count += 1
        if (count%100)==0:
            print(":::[INFO] Last Frames in Video of Phase %d... :::" % video_annotations.loc[index].at['Phase'])
        try:
            cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[video_annotations.loc[index].at['Phase']-1]) % (video,count), frame)
        except cv.error:
            break
        count += 1 
    

def grab_phase_frames(p):
    phase_list = video_annotations.loc[video_annotations['Phase'] == p]
    indices = phase_list.index.tolist()
    ret = True
    for i in range(len(indices)):
        count = phase_list.loc[indices[i]].at['FrameNo']
        capture.set(cv.CAP_PROP_POS_FRAMES, count)
        try:
            while ret == True and count < video_annotations.loc[indices[i]+1].at['FrameNo']:
                ret, frame = capture.read()
                if (count%200) == 0:
                    print("::: [INFO] Writing Frames of Phase %d... :::" % p)
                cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[p-1]) % (args.video,count), frame)
                count += 1
        except KeyError:
            while ret == True:
                ret, frame = capture.read()
                if (count%200) == 0:
                    print("::: [INFO] Writing Frames of Phase %d... :::" % p)
                try:
                    cv.imwrite("C:/labeledFrames/{}/%d_frame%d.jpg".format(phases[p-1]) % (args.video,count), frame)
                    count += 1
                except cv.error:
                    break
                
            
#videos = [269, 270, 271, 934, 928, 796, 354]

videos = [750, 760, 784, 808, 841]

for i in videos:
    print("next video")
    video_file = str(Path("C:/Users/Jörn/Documents/FH/BA/cataract-101/videos/case_{}.mp4".format(i)))
    capture = cv.VideoCapture(video_file)
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    video_annotations_i = video_annotations.loc[video_annotations['VideoID'] == i]
    grab_all_frames(video_annotations_i, frame_count, i)

input("Done, press Enter to leave...")    
            