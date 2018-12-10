import os
import glob
import pandas as pd
import numpy as np
from shutil import copyfile
from numpy import linalg as LA
from transforms3d import euler as trans
from transforms3d import taitbryan as trans2
import time

import maya.cmds as mc


def resetGesture(jointList, dofList):
    # reset gesture, the first one should be root node
    mc.setAttr(jointList[0] + '.' + dofList[0], 0)
    mc.setAttr(jointList[0] + '.' + dofList[1], 0)
    mc.setAttr(jointList[0] + '.' + dofList[2], 0)
    mc.setAttr(jointList[0] + '.' + dofList[3], 0)
    mc.setAttr(jointList[0] + '.' + dofList[4], 0)
    mc.setAttr(jointList[0] + '.' + dofList[5], 0)
    for j in jointList:
        mc.setAttr(j + '.' + dofList[0], 0)
        mc.setAttr(j + '.' + dofList[1], 0)
        mc.setAttr(j + '.' + dofList[2], 0)

def getPlane(p1, p2, p3):
    try:
        M = [[p1[1], p1[2], 1], [p2[1], p2[2], 1], [p3[1], p3[2], 1]]
        b = [-p1[0], -p2[0], -p3[0]]
        parameters = np.matmul(LA.inv(M), b);
        
        A = 1
        B = parameters[0];
        C = parameters[1];
        D = parameters[2];
        
    except:
        M = [[p1[0], p1[2], 1], [p2[0], p2[2], 1], [p3[0], p3[2], 1]]
        b = [-p1[1], -p2[1], -p3[1]]
        parameters = np.matmul(LA.inv(M), b);
        
        A = parameters[0];
        B = 1;
        C = parameters[1];
        D = parameters[2];
        
    return A, B, C, D
         
         
            

def vectors2Euler(vec0, vec1):
    # rotate from vec0 to vec1
    assert(vec0.shape[0] == 3)
    assert(vec1.shape[0] == 3)
    axis = np.cross(vec0, vec1);
    angle = np.arccos(np.dot(vec0, vec1)/(LA.norm(vec0) * LA.norm(vec1)))
    euler = trans.axangle2euler(axis, angle)
    
    return (180/np.pi)*np.array(euler)

def vectorPairsToEuler(v1, v2, v3, v4):
    assert(v1.shape[0] == 3)
    assert(v2.shape[0] == 3)
    assert(v3.shape[0] == 3)
    assert(v4.shape[0] == 3)
    
    v1 = v1/LA.norm(v1)
    v2 = v2/LA.norm(v2)
    v3 = v3/LA.norm(v3)
    v4 = v4/LA.norm(v4)
    assert(np.abs(np.dot(v1, v3) - np.dot(v2, v4)) < 1e-6)
    
    # for plane 1
    p1 = np.array([0,0,0])
    p2 = 0.5*(v1 + v2)
    p3 = np.cross(v1, v2)
    
    A1, B1, C1, D1 = getPlane(p1,p2,p3)
    
    # for plane 2
    p1 = np.array([0,0,0])
    p2 = 0.5*(v3 + v4)
    p3 = np.cross(v3, v4)
    
    A2, B2, C2, D2 = getPlane(p1,p2,p3)
    
    # compute axis
    M = [[B1, C1], [B2, C2]]
    b = [-D1-A1, -D2-A2]
    parameters = np.matmul(LA.inv(M), b)
    
    axis = np.array([1, parameters[0], parameters[1]])
    axis = axis/LA.norm(axis)
    
    # compute theta
    t1 = v1 - np.dot(v1, axis)*axis
    t2 = v2 - np.dot(v2, axis)*axis
    
    theta = np.arccos(np.dot(t1, t2)/LA.norm(t1)/LA.norm(t2))
    
    # check direction
    euler = trans.axangle2euler(axis, theta)
    rotMat = trans.euler2mat(euler[0], euler[1], euler[2])
    v2New = np.matmul(rotMat, v1)
    
    if (LA.norm(v2 - v2New) < 1e-6):
        return axis, theta, euler
    
    euler = trans.axangle2euler(axis, -theta)
    return axis, -theta, euler
    


def placePose(fileName, controlList, dofList, attrList, useRootPos=False, rootPos=np.array([0,0,0]), hScale=1, trackFeet=False):
    # first reset pose
    resetGesture(controlList, dofList)
    
    df = pd.read_csv(fileName, delimiter=',', header=None)
    poseList = [np.array([x[0], -x[1], -x[2]]) for x in df.values]
    assert(len(poseList) == 25)
    poseDict = {}
    
    poseDict["head"] = poseList[0];
    poseDict["neck"] = poseList[1];
    poseDict["rightArm"] = poseList[2];
    poseDict["rightForeArm"] = poseList[3];
    poseDict["rightHand"] = poseList[4];
    poseDict["leftArm"] = poseList[5];
    poseDict["leftForeArm"] = poseList[6];
    poseDict["leftHand"] = poseList[7];
    poseDict["hip"] = poseList[8];
    poseDict["rightUpLeg"] = poseList[9];
    poseDict["rightLeg"] = poseList[10];
    poseDict["rightFoot"] = poseList[24];
    poseDict["leftUpLeg"] = poseList[12];
    poseDict["leftLeg"] = poseList[13];
    poseDict["leftFoot"] = poseList[21];
    poseDict["rightToe"] = poseList[22];
    poseDict["leftToe"] = poseList[19];
    
    # spine
    vec0 = poseDict['leftArm'] - poseDict['hip']
    vec1 = poseDict['rightArm'] - poseDict['hip']
    body_norm = np.cross(vec0, vec1)
    body_up = 0.5*(vec0 + vec1)
    vec2 = np.array(mc.getAttr(controlList[13] + '.' + attrList[1])) - np.array(mc.getAttr(controlList[0] + '.' + attrList[1]))
    vec3 = np.array(mc.getAttr(controlList[10] + '.' + attrList[1])) - np.array(mc.getAttr(controlList[0] + '.' + attrList[1]))
    body_up_1 = 0.5*(vec2[0] + vec3[0])
    body_norm_1 = np.cross(vec2[0], vec3[0])
    
    _, _, euler = vectorPairsToEuler(body_up, body_up_1, body_norm, body_norm_1)
    
    mc.setAttr(controlList[1] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
 
    
    # head
    vec = poseDict["head"] - poseDict["neck"]
    ref = poseDict["neck"] - poseDict["hip"]
    euler = vectors2Euler(ref, vec)
    #print("head:")
    #print(euler)
    mc.setAttr(controlList[16] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # right arm
    vec = poseDict["rightForeArm"] - poseDict["rightArm"]
    ref = np.cross(body_norm, body_up)
    euler = vectors2Euler(ref, vec)
    #print("right arm:")
    #print(euler)
    mc.setAttr(controlList[10] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # left arm
    vec = poseDict["leftForeArm"] - poseDict["leftArm"]
    ref = np.cross(body_up, body_norm)
    euler = vectors2Euler(ref, vec)
    #print("left arm:")
    #print(euler)
    mc.setAttr(controlList[13] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # right forearm
    vec = poseDict["rightHand"] - poseDict["rightForeArm"]
    ref = poseDict["rightForeArm"] - poseDict["rightArm"]
    euler = vectors2Euler(ref, vec)
    #print("right forearm:")
    #print(euler)
    mc.setAttr(controlList[11] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # left forearm
    vec = poseDict["leftHand"] - poseDict["leftForeArm"]
    ref = poseDict["leftForeArm"] - poseDict["leftArm"]
    euler = vectors2Euler(ref, vec)
    #print("left forearm:")
    #print(euler)
    mc.setAttr(controlList[14] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # right up leg
    vec = poseDict["rightLeg"] - poseDict["rightUpLeg"]
    ref = -body_up
    euler = vectors2Euler(ref, vec)
    #print("right up leg:")
    #print(euler)
    mc.setAttr(controlList[4] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # left up leg
    vec = poseDict["leftLeg"] - poseDict["leftUpLeg"]
    ref = -body_up
    euler = vectors2Euler(ref, vec)
    #print("left up leg:")
    #print(euler)
    mc.setAttr(controlList[7] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # right leg
    vec = poseDict["rightFoot"] - poseDict["rightLeg"]
    ref = poseDict["rightLeg"] - poseDict["rightUpLeg"]
    euler = vectors2Euler(ref, vec)
    #print("right leg:")
    #print(euler)
    mc.setAttr(controlList[5] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    # left leg
    vec = poseDict["leftFoot"] - poseDict["leftLeg"]
    ref = poseDict["leftLeg"] - poseDict["leftUpLeg"]
    euler = vectors2Euler(ref, vec)
    #print("left leg:")
    #print(euler)
    mc.setAttr(controlList[8] + '.' + attrList[0], euler[0], euler[1], euler[2], type='double3')
    
    
    if trackFeet:
        # right foot
        vec = poseDict["rightToe"] - poseDict["rightFoot"]
        ref = poseDict["rightFoot"] - poseDict["rightLeg"]
        mat = np.array(trans2.euler2mat(0, 0, -np.pi/4))
        ref = mat.dot(ref)
        euler = vectors2Euler(ref, vec)
        #print("right foot:")
        #print(euler)
        mc.setAttr(controlList[6] + '.' + attrList[0], euler[0], euler[1], 0, type='double3')
        
        
        # left foot
        vec = poseDict["leftToe"] - poseDict["leftFoot"]
        ref = poseDict["leftFoot"] - poseDict["leftLeg"]
        mat = np.array(trans2.euler2mat(0, 0, -np.pi/4))
        ref = mat.dot(ref)
        euler = vectors2Euler(ref, vec)
        #print("left foot:")
        #print(euler)
        mc.setAttr(controlList[9] + '.' + attrList[0], euler[0], euler[1], 0, type='double3')
    
    
    # whole body rotation
    vec = poseDict['neck'] - poseDict['hip']
    ref = np.array(mc.getAttr(controlList[17] + '.' + attrList[1])) - np.array(mc.getAttr(controlList[0] + '.' + attrList[1]))
    euler1 = vectors2Euler(ref[0], vec)
    
    mc.setAttr(controlList[0] + '.' + attrList[0], euler1[0], euler1[1], euler1[2], type='double3')

    # whole body translation
    if useRootPos:
        vec = hScale*(poseDict['hip'] - rootPos)
        mc.setAttr(controlList[0] + '.' + attrList[1], vec[0], vec[1], vec[2], type='double3')
    

def selectJoints(controlList):
    for j in controlList:
        mc.select(j, add=True)


def buildFrame(t, controlList):
    mc.select( clear=True )
    selectJoints(controlList)
    mc.setKeyframe(attribute='rotate', time=t)
    mc.setKeyframe(attribute='translate', time=t)
    mc.select( clear=True )


def getRootPose(fileName):
    df = pd.read_csv(fileName, delimiter=',', header=None)
    poseList = [np.array([x[0], -x[1], -x[2]]) for x in df.values]
    assert(len(poseList) == 25)
    
    realH = LA.norm(poseList[0] - poseList[8])
    animiH = 62.354
    
    return poseList[8], animiH/realH

controlList = [u'Boy:Hips', 
               u'Boy:Spine',
               u'Boy:Spine1', 
               u'Boy:Spine2',
               u'Boy:RightUpLeg', 
               u'Boy:RightLeg',  # 5
               u'Boy:RightFoot',
               u'Boy:LeftUpLeg', 
               u'Boy:LeftLeg',
               u'Boy:LeftFoot',
               u'Boy:RightArm', # 10
               u'Boy:RightForeArm',
               u'Boy:RightHand',
               u'Boy:LeftArm',
               u'Boy:LeftForeArm',
               u'Boy:LeftHand', # 15
               u'Boy:Neck',
               u'Boy:Head']
               
dofList = [u'rotateX',
           u'rotateY',
           u'rotateZ',
           u'translateX',
           u'translateY',
           u'translateZ']
           
attrList = [u'rotate', u'translate']

tMin = 0
tMax = 1100

mc.select( clear=True )
selectJoints(controlList)
mc.cutKey(time=(tMin, tMax), attribute='rotate')
mc.cutKey(time=(tMin, tMax), attribute='translate')

folderNum = 2
folderPath = '/Users/flemone/Documents/maya/projects/COMP755/data/pose{}/'.format(folderNum)
pathList = glob.glob(os.path.join(folderPath, '*'))
split_ids = []
for path in pathList:
    split_id = os.path.basename(path)
    split_id = split_id.split('_')[1][:6]
    split_id = split_id if split_id != '' else '0'
    split_ids.append(int(split_id))

order = np.argsort(split_ids)
pathList = [pathList[idx] for idx in order]

p, s = getRootPose(pathList[0])

processingTime = list()
maxFrame = len(pathList)
for i in range(maxFrame):
    start = time.time()
    fileName = pathList[i]
    placePose(fileName, controlList, dofList, attrList, useRootPos=True, rootPos=p, hScale=s)
    buildFrame(i, controlList)
    mc.currentTime(i)
    #img = mc.render()
    #copyfile(img, "/Users/flemone/Documents/maya/projects/COMP755/images/test.{:04d}.jpg".format(i))
    end = time.time()
    processingTime.append(end - start)

print("Finish animation!")
print("{} fps.".format(1/np.mean(processingTime)))