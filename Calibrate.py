import cv2
import numpy as np
import math
import os
from itertools import product
import copy

################################################################################################
# To do :
# - choose resolution (perhaps back to python wrapper from imagingsource)
# - parallelisation of get_frame camera
# - DONE : change objp for circlegrid
# - DONE : change way to get detected area for at least Charuco but would be better for all
# - DONE : add calibration
# - DONE : perhaps calibrate camera one by one
# - DONE : calibration each image ( from 2 to infinite) with error
# - DONE : use as much camera as you want
# - DONE : save calibration parameters
# - mono et stereo : add optimization parameters
# - stereo : calibration
# - compare different kinds of objects, program parameters
# - write everything on images
################################################################################################

######################### DOCUMENTATION #########################
# Get frame by starting the app
# Press q to quit the program OR enter if you want to get a frame and find the chessboard / circlegrid / charuco in it
# Press q to refuse a frame OR enter to save this frame and compute the re projection error
#################################################################

# we create some functions to treat our problem

def get_Contour(size, points):
    """parameter :
     - size : 3-uple with take the shape of the initial picture
     - points : a list of points which represent an object
    return :
     - sortie : a list of tuple which represent the shortest contour to surround our object
    """
    # we create a new image and fill it with all our points
    black_pict = np.zeros((size), np.float32)
    # we modify the representation of our points
    points = np.ndarray.astype(points, np.int32)
    # we draw a polygone which all theses points
    cv2.fillConvexPoly(black_pict, points, (0, 0, 255), lineType=8, shift=0)
    # we create a gray picture to work with
    imgray = cv2.cvtColor(black_pict, cv2.COLOR_BGR2GRAY)
    imgray = cv2.convertScaleAbs(imgray)
    # we compute our contour
    _, contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # we compute contour's convexities
    pointsclef = cv2.convexHull(contours[0])
    # we return our result
    return pointsclef


def assym_objp_Circlegrid(CircleGrids_size, dist):
    """parameter:
     - CircleGrids_size : amount of circle on x, amount of circle on y
     - dist : distance between circles
     return:
     - objp : list of theoric position of each circle in the grid"""
    # We create our object
    objp = np.zeros((CircleGrids_size[0] * CircleGrids_size[1], 3), np.float32)
    # We fill it
    for i in range(0, len(objp)):
        x = i // CircleGrids_size[0] * dist / 2
        y = i % CircleGrids_size[0] * dist + x % 72
        objp[i] = (x, y, 0)
        print(objp[i])
    # we return our result
    return objp


def all_camera_opened(cam_list):
    """ function which surveys state of all cameras
    :param cam_list: list of opencv camera
    :return: true is all cameras are ok / false otherwise
    """
    for cam in cam_list:
        if cam.isOpened() == False:
            return False
    return True


def everything_is_valid(validity_list):
    """ function which surveys state of all frames
    :param cam_list: list of boolean
    :return: true is all frames are ok / false otherwise
    """
    for validity in validity_list:
        if validity == False:
            return False
    return True


def found_object_inframe(validity_list):
    """ function which surveys state of all found objects
    :param cam_list: list of boolean
    :return: true is at least one frames are ok or the list is empty (charuco) / false otherwise
    """
    if validity_list == []:
        return True
    for validity in validity_list:
        if validity == True:
            return True
    return False


def Calibration_multi_cam(list_cam, method, chessboard_size=None, circlegrid_size=None, charuco_size=None,
                          dictionary=None, scale=1 / 2., color=[0, 120, 250], params=0):
    """ This function help you to trigger one or several cameras thanks to opencv
    :param list_cam: a list of cv2 cameras or camera with a function .read() which return a boolean and cv2 image
     and isOpened() which return a boolean
    :param method: allow the user to choose his ways to trigger cameras "chessboard / circlegrid / charuco"
    :param chessboard_size: if the method is "chessboard", you have to give a tuple which
    represent the amount of intersection on the x_axis and y_axis (ex : (9,6))
    :param CircleGrids_size: if the method is "circlegrid", you have to give a tuple which
    represent the amount of circles on the x_axis and y_axis (ex : (4,11))
    :param Charuco_size: if the method is "charuco", you have to give a tuple which
    represent the amount of square on the x_axis and y_axis (ex : (7,10))
    :param dictionary: if the method is "charuco", you have to give
     the dictionnary to create the charuco
    :param scale : allow the user to choose the scale of the cam display
    :param color : choose the color of separators between camera and saved_areas
    :param params : allow the user to choose the opencv parameters to calibrate our cams (ex : CV_CALIB_USE_INTRINSIC_GUESS)
    :return: obj_points,img_points
    """
    # We look if the user forgot or missinformed parameters and we init main parameters
    try:
        nb_camera = len(list_cam)
        if nb_camera == 0:
            print("list_camera is empty.")
            exit(1)
    except:
        print("list_camera is not a list.")
        exit(1)
    try:
        for cam in list_cam:
            if cam.isOpened() == False:
                print("Error opening video stream cam")
                exit(1)
            bool, frame = cam.read()
    except:
        print("at least one camera haven't a function read() which return a boolean and a frame Or a camera isn't "
              "connected.")
        exit(1)
    if method == "chessboard" and chessboard_size != None:
        print("Calibration thanks to a chessboard.")
        try:
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        except:
            print("chessboard_size is supposed to be a tuple.")
            exit(1)
    elif method == "circlegrid" and circlegrid_size != None:
        print("Calibration thanks to a circlegrid.")
        try:
            objp = assym_objp_Circlegrid(circlegrid_size, 72)
        except:
            print("circlegrid is supposed to be a tuple.")
            exit(1)
    elif method == "charuco" and charuco_size != None and dictionary != None:
        print("Calibration thanks to a charuco.")
        try:
            Aruco_size = (math.ceil(charuco_size[0] / 2.), math.floor(charuco_size[1] / 2.))
        except:
            print("charuco_size is supposed to be a tuple.")
            exit(1)
        try:
            board = cv2.aruco.CharucoBoard_create(charuco_size[0], charuco_size[1], .025, .0125, dictionary)
        except:
            print("an error occurred with the dictionnary.")
    else:
        print("The selected method doesn't exist or some parameters are missing"
              " (available methods : chessboard / circlegrid / charuco )")
        exit(1)

    # We create useful parameters and our storage for our points
    if method == "chessboard" or method == "circlegrid":
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.00001)
        # we create our storage for the calibration
        obj_points = []
        img_points = []
        for i in range(0, len(list_cam)):
            obj_points.append([])
            img_points.append([])
    else:
        allcorners = []
        allids = []
        for i in range(0, len(list_cam)):
            allcorners.append([])
            allids.append([])

    # We create several pictures to represente areas already stored
    _, frame = list_cam[0].read()
    size = np.shape(frame)
    saved_areas = []
    for i in range(0, len(list_cam)):
        saved_areas.append(np.zeros((size[0], size[1], 3), np.float32))

    # We fix the display size of all our cams
    try:
        display_size = (int(scale * size[1]), int(scale * size[0]))
    except:
        print("scale must be a float or an int.")
        exit(1)

    # We create a list to store our calibration results
    calibration_result = []
    image_amount = []
    common_image_amount = 0
    for i in range(0, len(list_cam)):
        calibration_result.append("Unknown")
        image_amount.append(0)

    # we prepare our lines to separate our pictures
    V_cut = np.array(int(size[0] * scale) * color).reshape((int(size[0] * scale), 1, 3))
    H_cut = np.array(int(len(list_cam) * (size[1] * scale + 1) - 1) * color).reshape(
        (1, int(len(list_cam) * (size[1] * scale + 1) - 1), 3))

    # We start to get frames
    while (all_camera_opened(list_cam)):
        frame_list = []
        validity_list = []

        # we get a frame per cam
        for i in range(0, len(list_cam)):
            validity, frame = list_cam[i].read()
            validity_list.append(validity)
            frame_list.append(frame)

        # we survey if all cams get a frame
        if everything_is_valid(validity_list):

            # we display all cams and saved_areas
            # for i in range(0, len(list_cam)):
            #     cv2.imshow("Cam" + str(i + 1), cv2.resize(frame_list[i], display_size))
            #     cv2.imshow("registered_area_Cam" + str(i + 1), cv2.resize(saved_areas[i], display_size))

            for i in range(0, len(list_cam)):
                if i == 0:
                    frame_line = np.copy(cv2.resize(frame_list[i], display_size))
                    saved_line = cv2.resize(saved_areas[i], display_size)
                else:
                    frame_line = np.concatenate((frame_line, V_cut, cv2.resize(frame_list[i], display_size)), axis=1)
                    saved_line = np.concatenate((saved_line, V_cut, cv2.resize(saved_areas[i], display_size)), axis=1)

            cv2.imshow("Calibration_screen", np.concatenate((frame_line / 255., H_cut, saved_line), axis=0))

            k = cv2.waitKey(1)
            # If we want to try a detection on current frames
            if k == 13:
                validity_list = []
                points_list = []
                # According to the chosen method, we try to detect the targeted object
                if method == "chessboard":
                    for i in range(0, len(list_cam)):
                        validity, points = cv2.findChessboardCorners(frame_list[i], chessboard_size, None)
                        validity_list.append(validity)
                        points_list.append(points)

                elif method == "circlegrid":
                    for i in range(0, len(list_cam)):
                        validity, points = cv2.findCirclesGrid(frame_list[i], circlegrid_size,
                                                               flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                        validity_list.append(validity)
                        points_list.append(points)
                else:
                    for i in range(0, len(list_cam)):
                        points = cv2.aruco.detectMarkers(frame_list[i], dictionary)
                        if points[1] is None:
                            validity_list.append(False)
                        else:
                            validity_list.append(True)
                        points_list.append(points)

                # If at least one frame as the targeted object, we show the result
                if (found_object_inframe(validity_list)):
                    # We Draw and display the corners for the user
                    if method == "chessboard" or method == "circlegrid":
                        for i in range(0, len(list_cam)):
                            if validity_list[i]:
                                if method == "chessboard":
                                    cv2.drawChessboardCorners(frame_list[i], chessboard_size, points_list[i],
                                                              validity_list[i])
                                else:
                                    cv2.drawChessboardCorners(frame_list[i], circlegrid_size, points_list[i],
                                                              validity_list[i])
                    else:
                        for i in range(0, len(list_cam)):
                            if validity_list[i]:
                                cv2.aruco.drawDetectedMarkers(frame_list[i], points_list[i][0], points_list[i][1])
                    # We wait that the user choose to keep or remove these results
                    while (True):
                        # we display all cams and saved_areas
                        # for i in range(0, len(list_cam)):
                        #     cv2.imshow("Cam" + str(i + 1), cv2.resize(frame_list[i], display_size))
                        #     cv2.imshow("registered_area_Cam" + str(i + 1), cv2.resize(saved_areas[i], display_size))

                        for i in range(0, len(list_cam)):
                            if i == 0:
                                frame_line = np.copy(cv2.resize(frame_list[i], display_size))
                                saved_line = cv2.resize(saved_areas[i], display_size)
                            else:
                                frame_line = np.concatenate(
                                    (frame_line, V_cut, cv2.resize(frame_list[i], display_size)), axis=1)
                                saved_line = np.concatenate(
                                    (saved_line, V_cut, cv2.resize(saved_areas[i], display_size)), axis=1)

                        cv2.imshow("Calibration_screen", np.concatenate((frame_line / 255., H_cut, saved_line), axis=0))

                        k = cv2.waitKey(1)
                        # if we want to keep these datas
                        if k == 13:
                            if method == "chessboard" or method == "circlegrid":
                                for i in range(0, len(list_cam)):
                                    if validity_list[i]:
                                        obj_points[i].append(objp)
                                        cv2.cornerSubPix(cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2GRAY),
                                                         points_list[i], (11, 11), (-1, -1), criteria)
                                        img_points[i].append(points_list[i])
                                    else:
                                        obj_points[i].append([])
                                        img_points[i].append([])
                            else:
                                for i in range(0, len(list_cam)):
                                    if validity_list[i]:
                                        CharucoCorner = cv2.aruco.interpolateCornersCharuco(points_list[i][0],
                                                                                            points_list[i][1],
                                                                                            cv2.cvtColor(frame_list[i],
                                                                                                         cv2.COLOR_BGR2GRAY),
                                                                                            board)
                                        if CharucoCorner[1] is not None and CharucoCorner[2] is not None and len(
                                                CharucoCorner[1]) > 5:
                                            allcorners[i].append(CharucoCorner[1])
                                            allids[i].append(CharucoCorner[2])
                                        else:
                                            validity_list[i] = False
                                            allcorners[i].append([])
                                            allids[i].append([])

                            # we compute the contour around all our points
                            if method == "chessboard" or method == "circlegrid":
                                for i in range(0, len(list_cam)):
                                    if validity_list[i]:
                                        shortest_contour = get_Contour(size, points_list[i])
                                        saved_area = np.zeros((size[0], size[1]), np.uint8)
                                        cv2.fillConvexPoly(saved_area, shortest_contour, (255))
                                        saved_areas[i][saved_area == 255] += (0, 0.2, 0)
                                        saved_areas[i][saved_areas[i] > 1] = 1

                            else:
                                for i in range(0, len(list_cam)):
                                    if validity_list[i]:
                                        shortest_contour = get_Contour(size, allcorners[i][-1])
                                        saved_area = np.zeros((size[0], size[1]), np.uint8)
                                        cv2.fillConvexPoly(saved_area, shortest_contour, (255))
                                        saved_areas[i][saved_area == 255] += (0, 0.2, 0)
                                        saved_areas[i][saved_areas[i] > 1] = 1

                            # we calibrate our cam and store the error in a list
                            for i in range(0, len(list_cam)):
                                if (method == "chessboard" or method == "circlegrid") and validity_list[i]:
                                    # we have to change the way to store or objects for the following computations
                                    obj_points_tempo = []
                                    img_points_tempo = []
                                    for k in range(0, len(obj_points[i])):
                                        if obj_points[i][k] != []:
                                            obj_points_tempo.append(obj_points[i][k])
                                            img_points_tempo.append(img_points[i][k])

                                    # we calcul the calibration matrix and the re projection error
                                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_tempo,
                                                                                       img_points_tempo,
                                                                                       cv2.cvtColor(frame_list[i],
                                                                                                    cv2.COLOR_BGR2GRAY).shape[
                                                                                       ::-1], None, None,flags=params)

                                    calibration_result[i] = ret
                                    image_amount[i] += 1

                                elif method == "charuco" and validity_list[i]:
                                    allcorners_tempo = []
                                    allids_tempo = []
                                    for k in range(0, len(allcorners[i])):
                                        if allcorners[i][k] != []:
                                            allcorners_tempo.append(allcorners[i][k])
                                            allids_tempo.append(allids[i][k])

                                    # we calcul the calibration matrix and the re projection error
                                    ret, cameraMatrix, disCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                                        allcorners_tempo, allids_tempo, board, (size[0], size[1]), None, None,flags=params)

                                    calibration_result[i] = ret
                                    image_amount[i] += 1

                            # we display all available informations
                            cam_img_string = "Nb_Image :"
                            re_projection_string = "Re-projection error :"

                            for i in range(0, len(list_cam)):
                                cam_img_string += "| CAM " + str(i + 1) + " : " + str(image_amount[i]) + " |"
                                if calibration_result[i] == "Unknown":
                                    re_projection_string += "| CAM " + str(i + 1) + " : " + str(
                                        calibration_result[i]) + " |"
                                else:
                                    re_projection_string += "| CAM " + str(i + 1) + " : " + str(
                                        round(calibration_result[i], 2)) + " |"

                            print("\n#########################################\n")
                            print(cam_img_string)
                            print(re_projection_string)

                            # if all frames are finally useful
                            if everything_is_valid(validity_list):
                                common_image_amount += 1

                            print("amount of common images for all cams : " + str(common_image_amount))

                            print("Datas saved. back to video.")
                            break

                        # if we wont keep these data
                        if k == 113:
                            print("#########################################\n")
                            print("Datas not saved. back to video.")
                            break
            # If we want to stop detection and go to the next step
            elif k == 113:
                # if the storage file isn't already created, we create it
                if not os.path.exists("CalibrationParameters"):
                    os.makedirs("CalibrationParameters")
                # we save calibration parameters
                print("\n#########################################\n")
                for i in range(0, len(list_cam)):
                    if method == "chessboard" or method == "circlegrid":
                        # we have to change the way to store or objects for the following computations
                        obj_points_tempo = []
                        img_points_tempo = []
                        for k in range(0, len(obj_points[i])):
                            if obj_points[i][k] != []:
                                obj_points_tempo.append(obj_points[i][k])
                                img_points_tempo.append(img_points[i][k])

                        if obj_points_tempo == []:
                            print("No images available for camera " + str(i + 1) + ", no calibration parameters saved.")
                            continue

                        # we compute the calibration matrix and the re projection error
                        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_tempo,
                                                                           img_points_tempo,
                                                                           cv2.cvtColor(frame_list[i],
                                                                                        cv2.COLOR_BGR2GRAY).shape[::-1],
                                                                           None, None,flags=params)
                        # we save all our parameters
                        print("calibration parameters for camera " + str(
                            i + 1) + " stored in CalibrationParameters/CalibrationParametersCam" + str(i + 1) + ".npz")
                        np.savez("CalibrationParameters/CalibrationParametersCam" + str(i + 1), ret=ret, mtx=mtx,
                                 dist=dist, rvecs=rvecs, tvecs=tvecs)

                    else:
                        allcorners_tempo = []
                        allids_tempo = []
                        for k in range(0, len(allcorners[i])):
                            if allcorners[i][k] != []:
                                allcorners_tempo.append(allcorners[i][k])
                                allids_tempo.append(allids[i][k])

                        if allcorners_tempo == []:
                            print("No images available for camera " + str(i + 1) + ", no calibration parameters saved.")
                            continue

                        # we calcul the calibration matrix and the re projection error
                        ret, cameraMatrix, disCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                            allcorners_tempo, allids_tempo, board, (size[0], size[1]), None, None,flags=params)

                        # we save all our parameters
                        print("calibration parameters for camera " + str(
                            i + 1) + " stored in CalibrationParameters/CalibrationParametersCam" + str(i + 1))
                        np.savez("CalibrationParameters/CalibrationParametersCam" + str(i + 1), ret=ret,
                                 mtx=cameraMatrix, dist=disCoeffs, rvecs=rvecs, tvecs=tvecs)

                # In order to help the user, we explain of to get back saved data
                print("\nCalibration parameters are saved with np.savez with parameters : ret,mtx,dist,rvecs,tvecs.")
                cv2.destroyAllWindows()
                if method == "chessboard" or method == "circlegrid":
                    return (obj_points, img_points)
                else:
                    return (allcorners, allids)


def parameters_test(obj_points, img_points,method,size, board=None):
    """ this function is designed to test all parameters and return the best calibration
     parameters based on re projection error

    :param obj_points:
    :param img_points:
    :param method:
    :param size: (width,height)
    :param board: if you have a charuco, you have to give the used board
    :return:
    """

    best_calibration_tot=[]
    best_re_pro_tot=[]

    for i in range(0,len(obj_points)):
        obj_points_tempo = []
        img_points_tempo = []

        for j in range(0, len(obj_points[i])):
            if obj_points[i][j] != []:
                obj_points_tempo.append(obj_points[i][j])
                img_points_tempo.append(img_points[i][j])

        best_calibration="None"
        best_re_pro=1000
        Values=[cv2.CALIB_USE_INTRINSIC_GUESS,cv2.CALIB_FIX_PRINCIPAL_POINT,cv2.CALIB_FIX_ASPECT_RATIO,
                cv2.CALIB_ZERO_TANGENT_DIST,cv2.CALIB_FIX_K1,cv2.CALIB_FIX_K2,cv2.CALIB_FIX_K3,cv2.CALIB_FIX_K4,
                cv2.CALIB_FIX_K5,cv2.CALIB_FIX_K6,cv2.CALIB_RATIONAL_MODEL]
        for k in product([0,1],repeat=len(Values)):
            flag=0
            for l in range(0,len(k)):
                if k[l]==1:
                    flag+=Values[l]

            if method=="chessboard" or method=="circlegrid":
                ret, _, _, _, _ = cv2.calibrateCamera(obj_points_tempo,img_points_tempo,
                                                      (size[0], size[1]),None, None,flags=flag)
            elif method=="charuco":
                ret, _, _, _, _ = cv2.aruco.calibrateCameraCharuco(obj_points_tempo, img_points_tempo, board,
                                                                   (size[0], size[1]), None, None, flags=flag)
            else:
                print("wrong method")
                exit(1)

            if ret < best_re_pro:
                best_re_pro=ret
                best_calibration=k

        best_re_pro_tot.append(best_re_pro)
        best_calibration_tot.append(best_calibration)

        # if the storage file isn't already created, we create it
        if not os.path.exists("CalibrationParametersOptimized"):
            os.makedirs("CalibrationParametersOptimized")
        # we save calibration parameters
        print("\n#########################################\n")
        if method == "chessboard" or method == "circlegrid":
            # we compute the calibration matrix and the re projection error
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_tempo,img_points_tempo,
                                                      (size[0], size[1]),None, None,flags=best_calibration[i])
            # we save all our parameters
            print("calibration opimized parameters for camera " + str(
                i + 1) + " stored in CalibrationParametersOptimized/CalibrationParametersCam" + str(i + 1) + ".npz")
            np.savez("CalibrationParametersOptimized/CalibrationParametersCam" + str(i + 1), ret=ret, mtx=mtx,
                     dist=dist, rvecs=rvecs, tvecs=tvecs)

        else:
            # we calcul the calibration matrix and the re projection error
            ret, cameraMatrix, disCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(obj_points_tempo, img_points_tempo, board,
                                                                   (size[0], size[1]), None, None, flags=best_calibration[i])

            # we save all our parameters
            print("calibration parameters for camera " + str(
                i + 1) + " stored in CalibrationParametersOptimized/CalibrationParametersCam" + str(i + 1))
            np.savez("CalibrationParametersOptimized/CalibrationParametersCam" + str(i + 1), ret=ret,
                     mtx=cameraMatrix, dist=disCoeffs, rvecs=rvecs, tvecs=tvecs)

    return(best_re_pro_tot,best_calibration_tot)

def Read_Camera(camera,method,obj_points,img_points,charuco_size=None,dictionary=None,flag=None):
    """This function is designed to read the camera with new parameters in order to see the improvement
    :param camera the camera that we want to display
    :param npz_file the parameters file that we have to load
    """
    Values = [cv2.CALIB_USE_INTRINSIC_GUESS, cv2.CALIB_FIX_PRINCIPAL_POINT, cv2.CALIB_FIX_ASPECT_RATIO,
              cv2.CALIB_ZERO_TANGENT_DIST, cv2.CALIB_FIX_K1, cv2.CALIB_FIX_K2, cv2.CALIB_FIX_K3, cv2.CALIB_FIX_K4,
              cv2.CALIB_FIX_K5, cv2.CALIB_FIX_K6, cv2.CALIB_RATIONAL_MODEL]

    value = 0
    if flag!=None:
        for i in range(len(Values)):
            value+=flag[0][i]*Values[i]

    # we get back calibration parameters
    for i in range(0,len(obj_points)):
        obj_points_tempo = []
        img_points_tempo = []

        for j in range(0, len(obj_points[i])):
            if obj_points[i][j] != []:
                obj_points_tempo.append(obj_points[i][j])
                img_points_tempo.append(img_points[i][j])

    # we compute the new optimal matrix
    _, frame = camera.read()
    h, w = frame.shape[:2]

    if method=="chessboard" or method=="circlegrid":
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points_tempo, img_points_tempo,
                                                       (h, w), None, None, flags=value)
    if method=="charuco":
        board = cv2.aruco.CharucoBoard_create(charuco_size[0], charuco_size[1], .025, .0125, dictionary)
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(obj_points_tempo, img_points_tempo, board,
                                                        (h, w), None, None, flags=value)

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    np.savez("CalibrationParametersCam1", mtx=mtx,
             dist=dist, newcameramtx=newcameramtx, roi=roi)

    # we display our frames with the undistortion
    while (1):
        # undistortion
        _, frame = camera.read()
        # cv2.imshow("old_cam", frame)
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # cv2.imshow("new_cam", dst)
        # we crop the image
        x, y, w, h = roi
        dst2 = dst[y:y + h, x:x + w]
        cv2.imshow("new_cam_cut", dst2)
        k = cv2.waitKey(1)
        if k == 113:
            cv2.destroyAllWindows()
            break

def stereo_parameters_test(obj_points, img_points,path_calib1,path_calib2,img_size):
    """ this function is designed to test all available parameters for stereo calibration
    :param obj_points:
    :param img_points:
    :return:
    """

    best_calibration = "None"
    best_re_pro = 1000

    # we get back calibration parameters
    obj_points_tempo = []
    img1_points_tempo = []
    img2_points_tempo = []

    # Values = [cv2.CALIB_FIX_INTRINSIC,
    #           cv2.CALIB_FIX_FOCAL_LENGTH,cv2.CALIB_FIX_ASPECT_RATIO,cv2.CALIB_SAME_FOCAL_LENGTH
    #           ,cv2.CALIB_FIX_K2,cv2.CALIB_FIX_K3,cv2.CALIB_FIX_K4,
    #           cv2.CALIB_FIX_K5,cv2.CALIB_FIX_K6,cv2.CALIB_RATIONAL_MODEL]

    for j in range(0, len(obj_points[0])):
        if obj_points[1][j] != [] and obj_points[0][j]!=[]:
            obj_points_tempo.append(obj_points[0][j])
            img1_points_tempo.append(img_points[0][j])
            img2_points_tempo.append(img_points[1][j])

    file1 = np.load(path_calib1)
    file2 = np.load(path_calib2)
    mtx1=file1['mtx']
    dist1=file1['dist']
    mtx2=file2['mtx']
    dist2=file2['dist']

    # for k in product([0, 1], repeat=len(Values)):
    #     flag = 0
    #     for l in range(0, len(k)):
    #         if k[l] == 1:
    #             flag += Values[l]

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F=cv2.stereoCalibrate(obj_points_tempo, img1_points_tempo, img2_points_tempo,mtx1,dist1,mtx2,dist2,img_size)

        # if retval < best_re_pro:
        #     best_re_pro = retval
        #     best_calibration = k

    print(retval)


cam=cv2.VideoCapture(0)
cam2=cv2.VideoCapture(2)
obj_points, img_points = Calibration_multi_cam([cam,cam2], "circlegrid",
                                               circlegrid_size=(4, 11),
                                               chessboard_size=(9, 6), charuco_size=(7, 10),
                                               dictionary=cv2.aruco.getPredefinedDictionary(
                                                   cv2.aruco.DICT_ARUCO_ORIGINAL),scale=1.2/2)

_,frame=cam.read()
stereo_parameters_test(obj_points,img_points,"cam1_calibration.npz","cam2_calibration.npz",(frame.shape[0],frame.shape[1]))

# Read_Camera(cam, "charuco", obj_points, img_points,charuco_size=(7, 10),dictionary=cv2.aruco.getPredefinedDictionary(
#                                                    cv2.aruco.DICT_ARUCO_ORIGINAL))

# _,frame=cam.read()
# ret,values=parameters_test(obj_points,img_points,"charuco",(frame.shape[0],frame.shape[1]),cv2.aruco.CharucoBoard_create(7, 10, .025, .0125, cv2.aruco.getPredefinedDictionary(
#                                                    cv2.aruco.DICT_ARUCO_ORIGINAL)))
#
# Read_Camera(cam, "charuco", obj_points, img_points,charuco_size=(7, 10),dictionary=cv2.aruco.getPredefinedDictionary(
#                                                    cv2.aruco.DICT_ARUCO_ORIGINAL),flag=values)
# print(values)