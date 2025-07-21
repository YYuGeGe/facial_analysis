from pathlib import Path
import cv2
import numpy as np
from config import *

SHOW_SPLIT = False  # False True

def get_folders_in_path01(path):
    return [path / item.name for item in Path(path).iterdir() if item.is_dir()]


def get_folders_in_path(path):
    path_obj = Path(path)
    return [item for item in path_obj.iterdir() if item.is_dir()]


def get_files_by_extensions01(path, extensions):
    """
    获取指定路径下所有指定扩展名的文件

    :param path: 要搜索的目录路径
    :param extensions: 文件扩展名列表（如 ['.txt', '.csv']）
    :return: 符合条件的文件路径列表
    """
    all_files = []
    for ext in extensions:
        all_files.extend(Path(path).rglob(f'*{ext}'))

    return all_files


def get_files_by_extensions(path, extensions):
    all_files = []
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"路径不存在: {path}")
        return []

    for ext in extensions:
        matched_files = list(path_obj.rglob(f'*{ext}'))
        print(f"找到 {len(matched_files)} 个 {ext} 文件")
        all_files.extend(matched_files)

    return all_files


def check_file():
    folder_list = get_folders_in_path(data_root_path)
    print(folder_list)

    for folder in folder_list:
        print(folder)

        files = get_files_by_extensions(folder, extensions=extensions)

        for file in files:
            print(file)


def read_image(file, SHOW=False):
    image = cv2.imread(file)

    if SHOW:
        cv2.namedWindow('image', 2)
        cv2.imshow('image', image)
        cv2.waitKey(0)

    return image


def filter_nevus():
    # 异常点-痣 处理
    pass


def analysis_red_image(red_image):

    # 分离通道
    b, g, r = cv2.split(red_image)

    b = 255 - b
    g = 255 - g
    r = 255 - r

    SHOW_SPLIT = False  # False True
    if SHOW_SPLIT:
        cv2.namedWindow('image', 2)
        cv2.imshow('image', red_image)
        cv2.namedWindow('Blue Channel', 2)
        cv2.imshow('Blue Channel', b)
        cv2.namedWindow('Red Channel', 2)
        cv2.imshow('Red Channel', r)
        cv2.namedWindow('Green Channel', 2)
        cv2.imshow('Green Channel', g)
        cv2.waitKey(0)

    _, binary = cv2.threshold(b, head_intensity_threshold, 255, cv2.THRESH_BINARY)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 创建一个空白图像用于保存结果
    filtered_image = np.zeros_like(binary)

    # 遍历每个连通域（跳过背景 0）设置最小面积阈值
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_red_area:
            filtered_image[labels == i] = 255

    if SHOW_SPLIT:
        # 显示最终清理后的图像
        cv2.namedWindow('red_image', 2)
        cv2.imshow('red_image', red_image)
        cv2.namedWindow('Filtered Image', 2)
        cv2.imshow('Filtered Image', filtered_image)
        cv2.namedWindow('binary Image', 2)
        cv2.imshow('binary Image', binary)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filtered_image, binary


def analysis_mouse_red_image(mouse_roi_red_image, mouse_roi_mask):
    # 分离通道
    b, g, r = cv2.split(mouse_roi_red_image)
    b = 255 - b
    b = cv2.bitwise_and(b, b, mask=mouse_roi_mask)


    if SHOW_SPLIT:
        cv2.namedWindow('image', 2)
        cv2.imshow('image', mouse_roi_red_image)
        cv2.namedWindow('mask', 2)
        cv2.imshow('mask', mouse_roi_mask)
        cv2.namedWindow('Blue Channel', 2)
        cv2.imshow('Blue Channel', b)
        cv2.namedWindow('Red Channel', 2)
        cv2.imshow('Red Channel', r)
        cv2.namedWindow('Green Channel', 2)
        cv2.imshow('Green Channel', g)
        cv2.waitKey(0)

    _, binary = cv2.threshold(b, mouse_intensity_threshold, 255, cv2.THRESH_BINARY)
    non_zero_points = cv2.findNonZero(binary)
    # 排除干扰区域

    if SHOW_SPLIT:
        rgb_show = mouse_roi_red_image.copy()
        rgb_show_blank = np.zeros_like(mouse_roi_red_image)

        if non_zero_points is not None:
            for point in non_zero_points:
                x, y = tuple(point[0])
                cv2.circle(rgb_show_blank, (x, y), 1, (0, 0, 255), -1)  # 在空白图上画点

        rgb_show = cv2.addWeighted(rgb_show, 1.0, rgb_show_blank, 1.5, 0)
        cv2.namedWindow("rgb_show", cv2.WINDOW_NORMAL)
        cv2.imshow("rgb_show", rgb_show)
        cv2.waitKey(0)

    filter_points = []
    if non_zero_points is not None:
        for point in non_zero_points:
            location = tuple(point[0])
            x, y = location
            B = mouse_roi_red_image[y, x, 0]
            G = mouse_roi_red_image[y, x, 1]
            R = mouse_roi_red_image[y, x, 2]
            if R > mouse_red_threshold and G < mouse_green_threshold and B < mouse_blue_threshold:
                if y > 30:
                    filter_points.append(location)
                elif R > mouse_red_threshold + 30 and G < mouse_green_threshold - 30 and B < mouse_blue_threshold - 30:
                    filter_points.append(location)

    filter_show = mouse_roi_red_image.copy()
    filter_show_blank = np.zeros_like(mouse_roi_red_image)
    for point in filter_points:
        cv2.circle(filter_show_blank, point, 1, (0, 255, 0), -1)  # 在空白图上画点

    filter_show = cv2.addWeighted(filter_show, 1.0, filter_show_blank, 1.5, 0)

    if SHOW_SPLIT:
        cv2.namedWindow("filter_show", cv2.WINDOW_NORMAL)
        cv2.imshow("filter_show", filter_show)
        cv2.waitKey(0)

    region_mask = np.zeros_like(mouse_roi_red_image)
    for point in filter_points:
        cv2.circle(region_mask, point, 1, (255, 255, 255), -1)

    if SHOW_SPLIT:
        cv2.namedWindow("region_mask", cv2.WINDOW_NORMAL)
        cv2.imshow("region_mask", region_mask)
        cv2.waitKey(0)

    return region_mask


def analysis_nose_red_image(nose_roi_red_image, nose_roi_mask):
    # 分离通道
    b, g, r = cv2.split(nose_roi_red_image)
    b = 255 - b
    b = cv2.bitwise_and(b, b, mask=nose_roi_mask)

    if SHOW_SPLIT:
        cv2.namedWindow('image', 2)
        cv2.imshow('image', nose_roi_red_image)
        cv2.namedWindow('mask', 2)
        cv2.imshow('mask', nose_roi_mask)
        cv2.namedWindow('Blue Channel', 2)
        cv2.imshow('Blue Channel', b)
        cv2.waitKey(0)

    _, binary = cv2.threshold(b, nose_intensity_threshold, 255, cv2.THRESH_BINARY)
    non_zero_points = cv2.findNonZero(binary)

    if SHOW_SPLIT:
        rgb_show = nose_roi_red_image.copy()
        rgb_show_blank = np.zeros_like(nose_roi_red_image)

        if non_zero_points is not None:
            for point in non_zero_points:
                x, y = tuple(point[0])
                cv2.circle(rgb_show_blank, (x, y), 1, (0, 0, 255), -1)  # 在空白图上画点

        rgb_show = cv2.addWeighted(rgb_show, 1.0, rgb_show_blank, 1.5, 0)
        cv2.namedWindow("rgb_show", cv2.WINDOW_NORMAL)
        cv2.imshow("rgb_show", rgb_show)
        cv2.waitKey(0)

    filter_points = []  # 排除干扰区域
    if non_zero_points is not None:
        for point in non_zero_points:
            location = tuple(point[0])
            x, y = location
            B = nose_roi_red_image[y, x, 0]
            G = nose_roi_red_image[y, x, 1]
            R = nose_roi_red_image[y, x, 2]
            if R > nose_red_threshold and G < nose_green_threshold and B < nose_blue_threshold:
                if (x <30 and y > 216) or (x > 170 and y > 216) or (y > 268):
                    if R > nose_red_threshold + 30 and G < nose_green_threshold - 50 and B < nose_blue_threshold - 50:
                        filter_points.append(location)
                else:
                    filter_points.append(location)

    if SHOW_SPLIT:
        filter_show = nose_roi_red_image.copy()
        filter_show_blank = np.zeros_like(nose_roi_red_image)
        for point in filter_points:
            cv2.circle(filter_show_blank, point, 1, (0, 255, 0), -1)  # 在空白图上画点

        filter_show = cv2.addWeighted(filter_show, 1.0, filter_show_blank, 1.5, 0)

        cv2.namedWindow("filter_show", cv2.WINDOW_NORMAL)
        cv2.imshow("filter_show", filter_show)
        cv2.waitKey(0)

    region_mask = np.zeros_like(nose_roi_red_image)
    for point in filter_points:
        cv2.circle(region_mask, point, 1, (255, 255, 255), -1)

    if SHOW_SPLIT:
        cv2.namedWindow("region_mask", cv2.WINDOW_NORMAL)
        cv2.imshow("region_mask", region_mask)
        cv2.waitKey(0)

    return region_mask

def analysis_head_red_image(head_roi_red_image, head_roi_mask):

    # 分离通道
    b, g, r = cv2.split(head_roi_red_image)
    b = 255 - b
    b = cv2.bitwise_and(b, b, mask=head_roi_mask)

    if SHOW_SPLIT:
        cv2.namedWindow('image', 2)
        cv2.imshow('image', head_roi_red_image)
        cv2.namedWindow('mask', 2)
        cv2.imshow('mask', head_roi_mask)
        cv2.namedWindow('Blue Channel', 2)
        cv2.imshow('Blue Channel', b)
        cv2.waitKey(0)

    _, binary = cv2.threshold(b, head_intensity_threshold, 255, cv2.THRESH_BINARY)
    non_zero_points = cv2.findNonZero(binary)

    if SHOW_SPLIT:
        rgb_show = head_roi_red_image.copy()
        rgb_show_blank = np.zeros_like(head_roi_red_image)

        if non_zero_points is not None:
            for point in non_zero_points:
                x, y = tuple(point[0])
                cv2.circle(rgb_show_blank, (x, y), 1, (0, 0, 255), -1)  # 在空白图上画点

        rgb_show = cv2.addWeighted(rgb_show, 1.0, rgb_show_blank, 1.5, 0)
        cv2.namedWindow("rgb_show", cv2.WINDOW_NORMAL)
        cv2.imshow("rgb_show", rgb_show)
        cv2.waitKey(0)

    filter_points = []  # 排除干扰区域
    if non_zero_points is not None:
        for point in non_zero_points:
            location = tuple(point[0])
            filter_points.append(location)

    if SHOW_SPLIT:
        filter_show = head_roi_red_image.copy()
        filter_show_blank = np.zeros_like(head_roi_red_image)
        for point in filter_points:
            cv2.circle(filter_show_blank, point, 1, (0, 255, 0), -1)  # 在空白图上画点

        filter_show = cv2.addWeighted(filter_show, 1.0, filter_show_blank, 1.5, 0)

        cv2.namedWindow("filter_show", cv2.WINDOW_NORMAL)
        cv2.imshow("filter_show", filter_show)
        cv2.waitKey(0)

    region_mask = np.zeros_like(head_roi_red_image)
    for point in filter_points:
        cv2.circle(region_mask, point, 1, (255, 255, 255), -1)

    if SHOW_SPLIT:
        cv2.namedWindow("region_mask", cv2.WINDOW_NORMAL)
        cv2.imshow("region_mask", region_mask)
        cv2.waitKey(0)

    return region_mask

def analysis_right_face_red_image(right_face_roi_red_image, right_face_roi_mask):
    # 分离通道
    b, g, r = cv2.split(right_face_roi_red_image)
    b = 255 - b
    b = cv2.bitwise_and(b, b, mask=right_face_roi_mask)

    if SHOW_SPLIT:
        cv2.namedWindow('image', 2)
        cv2.imshow('image', right_face_roi_red_image)
        cv2.namedWindow('mask', 2)
        cv2.imshow('mask', right_face_roi_mask)
        cv2.namedWindow('Blue Channel', 2)
        cv2.imshow('Blue Channel', b)
        cv2.waitKey(0)

    _, binary = cv2.threshold(b, right_face_intensity_threshold, 255, cv2.THRESH_BINARY)
    non_zero_points = cv2.findNonZero(binary)

    if SHOW_SPLIT:
        rgb_show = right_face_roi_red_image.copy()
        rgb_show_blank = np.zeros_like(right_face_roi_red_image)

        if non_zero_points is not None:
            for point in non_zero_points:
                x, y = tuple(point[0])
                cv2.circle(rgb_show_blank, (x, y), 1, (0, 0, 255), -1)  # 在空白图上画点

        rgb_show = cv2.addWeighted(rgb_show, 1.0, rgb_show_blank, 1.5, 0)
        cv2.namedWindow("rgb_show", cv2.WINDOW_NORMAL)
        cv2.imshow("rgb_show", rgb_show)
        cv2.waitKey(0)

    filter_points = []  # 排除干扰区域
    if non_zero_points is not None:
        for point in non_zero_points:
            location = tuple(point[0])
            x, y = location
            B = right_face_roi_red_image[y, x, 0]
            G = right_face_roi_red_image[y, x, 1]
            R = right_face_roi_red_image[y, x, 2]
            if x < 60 and 220 < y < 330:
                if R > right_face_red_threshold and G < right_face_green_threshold and B < right_face_blue_threshold:
                    filter_points.append(location)
            else:
                filter_points.append(location)


    if SHOW_SPLIT:
        filter_show = right_face_roi_red_image.copy()
        filter_show_blank = np.zeros_like(right_face_roi_red_image)
        for point in filter_points:
            cv2.circle(filter_show_blank, point, 1, (0, 255, 0), -1)  # 在空白图上画点

        filter_show = cv2.addWeighted(filter_show, 1.0, filter_show_blank, 1.5, 0)

        cv2.namedWindow("filter_show", cv2.WINDOW_NORMAL)
        cv2.imshow("filter_show", filter_show)
        cv2.waitKey(0)

    region_mask = np.zeros_like(right_face_roi_red_image)
    for point in filter_points:
        cv2.circle(region_mask, point, 1, (255, 255, 255), -1)

    if SHOW_SPLIT:
        cv2.namedWindow("region_mask", cv2.WINDOW_NORMAL)
        cv2.imshow("region_mask", region_mask)
        cv2.waitKey(0)

    return region_mask


def analysis_left_face_red_image(left_face_roi_red_image, left_face_roi_mask):
    # 分离通道
    b, g, r = cv2.split(left_face_roi_red_image)
    b = 255 - b
    b = cv2.bitwise_and(b, b, mask=left_face_roi_mask)

    if SHOW_SPLIT:
        cv2.namedWindow('left_face_image', 2)
        cv2.imshow('left_face_image', left_face_roi_red_image)
        cv2.namedWindow('left_face_mask', 2)
        cv2.imshow('left_face_mask', left_face_roi_mask)
        cv2.namedWindow('left_face_Blue Channel', 2)
        cv2.imshow('left_face_Blue Channel', b)
        cv2.waitKey(0)

    _, binary = cv2.threshold(b, left_face_intensity_threshold, 255, cv2.THRESH_BINARY)
    non_zero_points = cv2.findNonZero(binary)

    if SHOW_SPLIT:
        rgb_show = left_face_roi_red_image.copy()
        rgb_show_blank = np.zeros_like(left_face_roi_red_image)

        if non_zero_points is not None:
            for point in non_zero_points:
                x, y = tuple(point[0])
                cv2.circle(rgb_show_blank, (x, y), 1, (0, 0, 255), -1)  # 在空白图上画点

        rgb_show = cv2.addWeighted(rgb_show, 1.0, rgb_show_blank, 1.5, 0)
        cv2.namedWindow("left_face_rgb_show", cv2.WINDOW_NORMAL)
        cv2.imshow("left_face_rgb_show", rgb_show)
        cv2.waitKey(0)

    filter_points = []  # 排除干扰区域
    if non_zero_points is not None:
        for point in non_zero_points:
            location = tuple(point[0])
            x, y = location
            B = left_face_roi_red_image[y, x, 0]
            G = left_face_roi_red_image[y, x, 1]
            R = left_face_roi_red_image[y, x, 2]
            if x > 226 and 220 < y < 330:
                if R > right_face_red_threshold and G < right_face_green_threshold and B < right_face_blue_threshold:
                    filter_points.append(location)
            else:
                filter_points.append(location)

    if SHOW_SPLIT:
        filter_show = left_face_roi_red_image.copy()
        filter_show_blank = np.zeros_like(left_face_roi_red_image)
        for point in filter_points:
            cv2.circle(filter_show_blank, point, 1, (0, 255, 0), -1)  # 在空白图上画点

        filter_show = cv2.addWeighted(filter_show, 1.0, filter_show_blank, 1.5, 0)

        cv2.namedWindow("left_face_filter_show", cv2.WINDOW_NORMAL)
        cv2.imshow("left_face_filter_show", filter_show)
        cv2.waitKey(0)

    region_mask = np.zeros_like(left_face_roi_red_image)
    for point in filter_points:
        cv2.circle(region_mask, point, 1, (255, 255, 255), -1)

    if SHOW_SPLIT:
        cv2.namedWindow("left_face_region_mask", cv2.WINDOW_NORMAL)
        cv2.imshow("left_face_region_mask", region_mask)
        cv2.waitKey(0)

    return region_mask



