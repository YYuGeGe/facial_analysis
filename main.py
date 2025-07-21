# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os

import cv2
import numpy as np
from utils import *
from config import *
from draw import *
from align import *
import mediapipe as mp


face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


def face_mesh_detection(image):
    return face_mesh.process(image)


def cpt_max_rect(points):
    x, y, w, h = None, None, None, None
    # 类型和形状处理
    if not isinstance(points, np.ndarray):
        mask_points = np.array(points)
    mask_points = np.squeeze(mask_points)
    try:
        x, y, w, h = cv2.boundingRect(mask_points)
    except Exception as e:
        print(f"failed to get boundingRect: {e}")

    return [x, y, w, h]


def get_region_std_mask(red_mask_image, region_mask_image, region_std_shape):
    """
    在 mask 图像中找到最大外接矩形，并将该区域裁剪并 resize 到指定尺寸。

    参数:
        region_mask_image (np.ndarray): 输入的单通道二值图像（0 和 255）。
        region_std_shape (tuple): 目标尺寸 (width, height)

    返回:
        resized_roi (np.ndarray or None): 调整尺寸后的 ROI 区域
    """
    if region_mask_image is None or len(region_mask_image.shape) != 2:
        print("输入图像无效或不是单通道图像")
        return None

    # 只保留region_mask_image对应区域的数据信息
    red_mask_image = cv2.bitwise_and(red_mask_image, red_mask_image, mask=region_mask_image)


    # 寻找轮廓
    contours, _ = cv2.findContours(region_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未找到任何轮廓")
        return None

    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 获取最大外接矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 裁剪图像
    roi_mask = region_mask_image[y:y + h, x:x + w]

    roi_points_mask = red_mask_image[y:y + h, x:x + w]
    roi_points_mask = cv2.cvtColor(roi_points_mask, cv2.COLOR_BGR2GRAY)

    # 缩放至目标尺寸
    roi_mask = cv2.resize(roi_mask, (region_std_shape[0], region_std_shape[1]))
    roi_points_mask = cv2.resize(roi_points_mask, (region_std_shape[0], region_std_shape[1]))

    # cv2.namedWindow('roi_mask', 2)
    # cv2.imshow('roi_mask', roi_mask)
    # cv2.namedWindow('roi_points_mask', 2)
    # cv2.imshow('roi_points_mask', roi_points_mask)
    # cv2.waitKey()

    return roi_mask, roi_points_mask


def count_white_pixels(roi_mask, roi_points_mask):
    """
    统计图像中大于指定阈值的“白点”像素数量和占比。

    参数:
        image (np.ndarray): 输入图像（支持单通道或三通道）。
        threshold (int): 白色像素的灰度阈值，默认 200。

    返回:
        dict: 包含 white_count 和 white_ratio 的字典。
    """
    if roi_mask is None or roi_mask.size == 0:
        return {"white_count": 0, "white_ratio": 0.0}
    if roi_points_mask is None or roi_points_mask.size == 0:
        return {"white_count": 0, "white_ratio": 0.0}
    # 确保 roi_points_mask 是单通道图像
    if len(roi_points_mask.shape) == 3 and roi_points_mask.shape[2] == 3:
        roi_points_mask = cv2.cvtColor(roi_points_mask, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('roi_mask', 2)
    cv2.imshow('roi_mask', roi_mask)
    cv2.namedWindow('roi_points_mask', 2)
    cv2.imshow('roi_points_mask', roi_points_mask)
    cv2.waitKey()

    region_total_pixels = cv2.countNonZero(roi_mask)
    region_white_count = cv2.countNonZero(roi_points_mask)

    return [region_total_pixels, region_white_count]


def combine_all_mask(masks, SHOW=False):
    if len(masks) == 0:
        return None
    combined_mask = np.zeros_like(masks[0])

    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    if SHOW:
        cv2.namedWindow("Combined Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Combined Mask", combined_mask)
        cv2.waitKey(0)

    return combined_mask


def get_face_mask_from_rgb_image(rgb_img):

    # step 1: detect landmarks
    results = face_mesh_detection(rgb_img)

    # step 2: get region mask
    lip_mask_image = np.zeros(rgb_img.shape, np.uint8)
    mouse_mask_image = np.zeros(rgb_img.shape, np.uint8)
    nose_mask_image = np.zeros(rgb_img.shape, np.uint8)
    head_mask_image = np.zeros(rgb_img.shape, np.uint8)
    right_face_mask_image = np.zeros(rgb_img.shape, np.uint8)
    left_face_mask_image = np.zeros(rgb_img.shape, np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lip_mask_points = get_ordered_points(rgb_img, face_landmarks, lip_points)
            mouse_mask_points = get_ordered_points(rgb_img, face_landmarks, mouse_points)
            nose_mask_points = get_ordered_points(rgb_img, face_landmarks, nose_points)
            head_mask_points = get_ordered_points(rgb_img, face_landmarks, head_points)
            right_face_mask_points = get_ordered_points(rgb_img, face_landmarks, right_face_points)
            left_face_mask_points = get_ordered_points(rgb_img, face_landmarks, left_face_points)

            cv2.fillPoly(lip_mask_image, lip_mask_points, (255, 255, 255), )
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (lip_dilate_size, lip_dilate_size))
            lip_mask_image = cv2.dilate(lip_mask_image, kernel, iterations=lip_dilate_iterations)
            cv2.fillPoly(mouse_mask_image, mouse_mask_points, (255, 255, 255), )
            mouse_mask_image = mouse_mask_image - lip_mask_image

            cv2.fillPoly(nose_mask_image, nose_mask_points, (255, 255, 255), )
            cv2.fillPoly(head_mask_image, head_mask_points, (255, 255, 255), )
            cv2.fillPoly(right_face_mask_image, right_face_mask_points, (255, 255, 255), )
            cv2.fillPoly(left_face_mask_image, left_face_mask_points, (255, 255, 255), )

    mouse_mask_single = cv2.cvtColor(mouse_mask_image, cv2.COLOR_BGR2GRAY)
    nose_mask_single = cv2.cvtColor(nose_mask_image, cv2.COLOR_BGR2GRAY)
    head_mask_single = cv2.cvtColor(head_mask_image, cv2.COLOR_BGR2GRAY)
    right_face_mask_single = cv2.cvtColor(right_face_mask_image, cv2.COLOR_BGR2GRAY)
    left_face_mask_single = cv2.cvtColor(left_face_mask_image, cv2.COLOR_BGR2GRAY)

    regions_mask = [mouse_mask_single, nose_mask_single, head_mask_single, left_face_mask_single, right_face_mask_single]

    return regions_mask


def get_region_std_red_image(red_img, region_mask_image, region_std_shape):
    """
    在 mask 图像中找到最大外接矩形，并将该区域裁剪并 resize 到指定尺寸。

    参数:
        region_mask_image (np.ndarray): 输入的单通道二值图像（0 和 255）。
        region_std_shape (tuple): 目标尺寸 (width, height)

    返回:
        resized_roi (np.ndarray or None): 调整尺寸后的 ROI 区域
    """
    if red_img is None or len(red_img.shape) != 3:
        print("输入图像无效或不是单通道图像")
        return None

    # 寻找轮廓
    contours, _ = cv2.findContours(region_mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未找到任何轮廓")
        return None

    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 获取最大外接矩形
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 裁剪图像
    roi_red_region_image = red_img[y:y + h, x:x + w, :]
    roi_mask_region_image = region_mask_image[y:y + h, x:x + w]

    # cv2.namedWindow('roi_red_region_image', 2)
    # cv2.imshow('roi_red_region_image', roi_red_region_image)
    # cv2.namedWindow('roi_mask_region_image', 2)
    # cv2.imshow('roi_mask_region_image', roi_mask_region_image)
    # cv2.waitKey()

    # 缩放至目标尺寸
    roi_red_region_image = cv2.resize(roi_red_region_image, (region_std_shape[0], region_std_shape[1]))
    roi_mask_region_image = cv2.resize(roi_mask_region_image, (region_std_shape[0], region_std_shape[1]))

    return roi_mask_region_image, roi_red_region_image


def analysis_face_red_image(face_region_mask, red_img, rgb_img=None):

    # face_mask_only_region roi_mask_region_image, roi_red_image
    mouse_roi_mask, mouse_roi_red_image = get_region_std_red_image(red_img, face_region_mask[0], mouse_shape)
    nose_roi_mask, nose_roi_red_image = get_region_std_red_image(red_img, face_region_mask[1], nose_shape)
    head_roi_mask, head_roi_red_image = get_region_std_red_image(red_img, face_region_mask[2], head_shape)
    right_face_roi_mask, right_face_roi_red_image = get_region_std_red_image(red_img, face_region_mask[3], right_face_shape)
    left_face_roi_mask, left_face_roi_red_image = get_region_std_red_image(red_img, face_region_mask[4], left_face_shape)

    # face_mask_only_region
    mouse_roi_point_mask = analysis_mouse_red_image(mouse_roi_red_image, mouse_roi_mask)
    nose_roi_point_mask = analysis_nose_red_image(nose_roi_red_image, nose_roi_mask)
    head_roi_point_mask = analysis_head_red_image(head_roi_red_image, head_roi_mask)
    right_face_roi_point_mask = analysis_right_face_red_image(right_face_roi_red_image, right_face_roi_mask)
    left_face_roi_point_mask = analysis_left_face_red_image(left_face_roi_red_image, left_face_roi_mask)

    roi_masks = [mouse_roi_mask, nose_roi_mask, head_roi_mask, right_face_roi_mask, left_face_roi_mask]
    roi_points_masks = [mouse_roi_point_mask, nose_roi_point_mask, head_roi_point_mask, right_face_roi_point_mask, left_face_roi_point_mask]

    return roi_masks, roi_points_masks


def cpt_region_ration(roi_masks, roi_points_masks):

    ration_results = dict()

    # count_white_pixels(roi_mask, roi_points_mask)
    mouse_result = count_white_pixels(roi_masks[0], roi_points_masks[0])
    nose_result = count_white_pixels(roi_masks[1], roi_points_masks[1])
    head_result = count_white_pixels(roi_masks[2], roi_points_masks[2])
    left_face_result = count_white_pixels(roi_masks[3], roi_points_masks[3])
    right_face_result = count_white_pixels(roi_masks[4], roi_points_masks[4])

    ration_results['mouse'] = mouse_result
    ration_results['nose'] = nose_result
    ration_results['head'] = head_result
    ration_results['right_face'] = right_face_result
    ration_results['left_face'] = left_face_result

    return ration_results


def facial_analysis(star_rgb_image_path, star_red_image_path, end_rgb_image_path, end_red_image_path):
    # step 1: 对rgb图片进行对齐
    method = 'sift'  # 'mediapipe' or 'sift'
    ref_rgb_img, aligned_rgb_img, M, _, _, _ = align_faces(star_rgb_image_path, end_rgb_image_path, method)

    # step 2: 对red图片进行相对应的投影变换
    ref_red_img = cv2.imread(star_red_image_path)
    end_red_img = cv2.imread(end_red_image_path)
    aligned_red_img = cv2.warpPerspective(end_red_img, M, (ref_rgb_img.shape[1], ref_rgb_img.shape[0]),
                                              flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)

    # 显示对齐效果
    SHOW_ALIGN_RESULT = False
    if SHOW_ALIGN_RESULT:
        aligned_result_image = cv2.addWeighted(ref_rgb_img, 0.5, aligned_rgb_img, 0.5, 0)
        cv2.namedWindow("aligned_result_image", cv2.WINDOW_NORMAL)
        cv2.imshow("aligned_result_image", aligned_result_image)
        cv2.waitKey(0)
        # cv2.imwrite(f"ref_rgb_img.jpg", ref_rgb_img)
        # cv2.imwrite(f"aligned_rgb_img.jpg", aligned_rgb_img)
        # cv2.imwrite(f"ref_red_img.jpg", ref_red_img)
        # cv2.imwrite(f"aligned_red_img.jpg", aligned_red_img)

    # step 3: get region mask
    ref_regions_mask = get_face_mask_from_rgb_image(ref_rgb_img)
    aligned_regions_mask = get_face_mask_from_rgb_image(aligned_rgb_img)

    # step 4: analysis red region image
    ref_roi_masks, ref_roi_points_masks = analysis_face_red_image(ref_regions_mask, ref_red_img, rgb_img=ref_rgb_img)
    aligned_roi_masks, aligned_roi_points_masks = analysis_face_red_image(aligned_regions_mask, aligned_red_img, rgb_img=aligned_rgb_img)

    # step 5: get region ration
    ref_ration_results = cpt_region_ration(ref_roi_masks, ref_roi_points_masks)
    aligned_ration_results = cpt_region_ration(aligned_roi_masks, aligned_roi_points_masks)

    # step 6: show results
    print('==========================================')
    ref_values = list(ref_ration_results.values())
    if ref_values:
        total_pixels = sum(ref_values[0])
    for key, value in ref_ration_results.items():
        print(key, 100 * value[1] / total_pixels)

    print('==========================================')
    aligned_values = list(aligned_ration_results.values())
    if aligned_values:
        total_pixels = sum(aligned_values[0])
    for key, value in aligned_ration_results.items():
        print(key, 100 * value[1] / total_pixels)


if __name__ == '__main__':

    folder_name = '00eafc9b145f40f49ea4c236c8f549a6'

    star_rgb_image_name = '00eafc9b145f40f49ea4c236c8f549a6-20230916-Rgb_Middle_T.jpg'
    star_red_image_name = '00eafc9b145f40f49ea4c236c8f549a6-20230916-Red_Middle_T.jpg'
    star_rgb_image_path = os.path.join(data_root_path, folder_name, star_rgb_image_name)
    star_red_image_path = os.path.join(data_root_path, folder_name, star_red_image_name)

    end_rgb_image_name = '00eafc9b145f40f49ea4c236c8f549a6-20240215-Rgb_Middle_T.jpg'
    end_red_image_name = '00eafc9b145f40f49ea4c236c8f549a6-20240215-Red_Middle_T.jpg'
    end_rgb_image_path = os.path.join(data_root_path, folder_name, end_rgb_image_name)
    end_red_image_path = os.path.join(data_root_path, folder_name, end_red_image_name)

    facial_analysis(star_rgb_image_path, star_red_image_path, end_rgb_image_path, end_red_image_path)
