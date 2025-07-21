import os
import cv2
import numpy as np
import mediapipe as mp

# 创建SIFT检测器
sift = cv2.SIFT_create()


def get_all_facial_landmarks(image, method='mediapipe'):
    """
    获取图像中关键点坐标
    :param image: 输入图像
    :param method: 关键点检测方法 ('mediapipe' 或 'sift')
    :return: 关键点坐标数组 (N, 2)
    """
    if method == 'mediapipe':
        points = get_mediapipe_keypoints(image)
    elif method == 'sift':
        points = get_sift_keypoints(image)
    else:
        raise ValueError(f"不支持的检测方法: {method}")

    # 确保点阵是二维数组
    if len(points) > 0:
        if len(points.shape) == 1:
            points = points.reshape(-1, 2)
        return points
    else:
        return np.empty((0, 2), dtype=np.float32)


def get_mediapipe_keypoints(image):
    """
    使用MediaPipe获取人脸关键点
    """
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = []

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        for landmark in face_landmarks.landmark:
            landmarks.append((int(landmark.x * w), int(landmark.y * h)))

    return np.array(landmarks, dtype=np.float32)


def get_sift_keypoints(image):
    """
    使用SIFT算法获取关键点
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测关键点和计算描述符
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # 提取关键点坐标
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return points


def match_keypoints(desc1, desc2):
    """
    匹配两个图像的SIFT特征点
    """
    # 创建FLANN匹配器
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行KNN匹配
    matches = flann.knnMatch(desc1, desc2, k=2)

    # 筛选良好匹配点 (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches


def align_faces_v1(reference_path, target_path, method='mediapipe'):
    """
    使用选择的特征点方法进行人脸对齐
    :param reference_path: 基准图像路径
    :param target_path: 待对齐图像路径
    :param method: 关键点检测方法 ('mediapipe' 或 'sift')
    :return: 对齐后的图像和变换矩阵
    """
    # 读取图像
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise ValueError(f"无法读取图像文件: {reference_path}")

    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError(f"无法读取图像文件: {target_path}")

    # 添加调试信息
    # print(f"参考图像关键点形状: {ref_points.shape}")
    # print(f"目标图像关键点形状: {target_points.shape}")
    # print(f"参考图像关键点数量: {len(ref_points)}")
    # print(f"目标图像关键点数量: {len(target_points)}")

    # 对于SIFT方法，需要额外的匹配步骤
    if method == 'sift':
        # 获取所有关键点
        ref_points = get_all_facial_landmarks(ref_img, 'sift')
        target_points = get_all_facial_landmarks(target_img, 'sift')

        # 检查关键点数量
        if len(ref_points) < 10 or len(target_points) < 10:
            raise RuntimeError(f"未检测到足够的关键点: ref={len(ref_points)}, target={len(target_points)}")
        # 转换为灰度图
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

        # 检测关键点和描述符
        ref_kps, ref_desc = sift.detectAndCompute(ref_gray, None)
        target_kps, target_desc = sift.detectAndCompute(target_gray, None)

        # 匹配关键点
        matches = match_keypoints(ref_desc, target_desc)

        if len(matches) < 10:
            raise RuntimeError(f"未找到足够的匹配点: {len(matches)}")

        # 获取匹配点的坐标
        ref_points = np.float32([ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        target_points = np.float32([target_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    else:
        # 获取所有关键点
        ref_points = get_all_facial_landmarks(ref_img, 'mediapipe')
        target_points = get_all_facial_landmarks(target_img, 'mediapipe')

        # 检查关键点数量
        if len(ref_points) < 10 or len(target_points) < 10:
            raise RuntimeError(f"未检测到足够的关键点: ref={len(ref_points)}, target={len(target_points)}")
        # 对于MediaPipe，直接使用所有点
        ref_points = ref_points.reshape(-1, 1, 2)
        target_points = target_points.reshape(-1, 1, 2)

    # 使用RANSAC计算最佳变换矩阵
    M, mask = cv2.findHomography(
        target_points,
        ref_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        confidence=0.99
    )

    if M is None:
        raise RuntimeError("无法计算有效的变换矩阵")

    # 应用投影变换
    aligned_img = cv2.warpPerspective(
        target_img,
        M,
        (ref_img.shape[1], ref_img.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )

    return ref_img, aligned_img,  M, mask, ref_points, target_points


def align_faces(reference_path, target_path, method='mediapipe'):
    """
    使用选择的特征点方法进行人脸对齐，自动回退机制

    :param reference_path: 基准图像路径
    :param target_path: 待对齐图像路径
    :param method: 首选关键点检测方法 ('mediapipe' 或 'sift')
    :return: 对齐后的图像和变换矩阵
    """
    # 读取图像
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise ValueError(f"无法读取图像文件: {reference_path}")

    target_img = cv2.imread(target_path)
    if target_img is None:
        raise ValueError(f"无法读取图像文件: {target_path}")

    # 尝试首选方法
    fallback_used = False
    fallback_method = 'sift' if method == 'mediapipe' else 'mediapipe'

    try:
        # SIFT方法处理流程
        if method == 'sift':
            ref_points, target_points, success = process_sift_method(ref_img, target_img)
            if not success:
                raise RuntimeError(f"SIFT方法关键点不足")

        # MediaPipe方法处理流程
        else:
            ref_points, target_points, success = process_mediapipe_method(ref_img, target_img)
            if not success:
                raise RuntimeError(f"MediaPipe方法关键点不足")

    except Exception as e:
        print(f"⚠️ {method}方法失败: {str(e)}，尝试回退到{fallback_method}方法")
        fallback_used = True

        # 回退到备选方法
        if fallback_method == 'sift':
            ref_points, target_points, success = process_sift_method(ref_img, target_img)
            if not success:
                raise RuntimeError(f"回退SIFT方法仍关键点不足")
        else:
            ref_points, target_points, success = process_mediapipe_method(ref_img, target_img)
            if not success:
                raise RuntimeError(f"回退MediaPipe方法仍关键点不足")

    # 使用RANSAC计算最佳变换矩阵
    M, mask = cv2.findHomography(
        target_points,
        ref_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        confidence=0.99
    )

    if M is None:
        raise RuntimeError("无法计算有效的变换矩阵")

    # 应用投影变换
    aligned_img = cv2.warpPerspective(
        target_img,
        M,
        (ref_img.shape[1], ref_img.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # 记录实际使用的方法
    actual_method = fallback_method if fallback_used else method

    return ref_img, aligned_img, M, mask, ref_points, target_points
    # return ref_img, aligned_img, M, mask, ref_points, target_points, actual_method


# 辅助函数：处理SIFT方法
def process_sift_method(ref_img, target_img):
    """SIFT方法处理流程"""
    # 获取所有关键点
    ref_points = get_all_facial_landmarks(ref_img, 'sift')
    target_points = get_all_facial_landmarks(target_img, 'sift')

    # 检查关键点数量
    if len(ref_points) < 10 or len(target_points) < 10:
        return None, None, False

    # 转换为灰度图
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # 检测关键点和描述符
    ref_kps, ref_desc = sift.detectAndCompute(ref_gray, None)
    target_kps, target_desc = sift.detectAndCompute(target_gray, None)

    # 匹配关键点
    matches = match_keypoints(ref_desc, target_desc)

    if len(matches) < 10:
        return None, None, False

    # 获取匹配点的坐标
    ref_points = np.float32([ref_kps[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target_points = np.float32([target_kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    return ref_points, target_points, True


# 辅助函数：处理MediaPipe方法
def process_mediapipe_method(ref_img, target_img):
    """MediaPipe方法处理流程"""
    # 获取所有关键点
    ref_points = get_all_facial_landmarks(ref_img, 'mediapipe')
    target_points = get_all_facial_landmarks(target_img, 'mediapipe')

    # 检查关键点数量
    if len(ref_points) < 10 or len(target_points) < 10:
        return None, None, False

    # 对于MediaPipe，直接使用所有点
    ref_points = ref_points.reshape(-1, 1, 2)
    target_points = target_points.reshape(-1, 1, 2)

    return ref_points, target_points, True


def visualize_alignment(ref_img, aligned_img, ref_points, target_points, M, mask, method='mediapipe'):
    """可视化对齐结果和关键点匹配"""
    # 创建可视化图像
    vis_img = np.hstack((ref_img, aligned_img))

    # 统一处理点阵形状
    if ref_points.ndim == 3:  # 形状为 (N, 1, 2)
        ref_points = ref_points.reshape(-1, 2)
    if target_points.ndim == 3:  # 形状为 (N, 1, 2)
        target_points = target_points.reshape(-1, 2)

    # 对于SIFT方法
    if method == 'sift':
        # 绘制基准图像的关键点
        for pt in ref_points:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

        # 变换目标关键点
        transformed_points = cv2.perspectiveTransform(
            target_points.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        # 绘制变换后的目标点
        for pt in transformed_points:
            cv2.circle(vis_img, (int(pt[0] + ref_img.shape[1]), int(pt[1])), 5, (0, 0, 255), -1)

        # 绘制匹配线
        for i in range(len(ref_points)):
            if mask[i] > 0:  # 只绘制内点
                ref_pt = (int(ref_points[i][0]), int(ref_points[i][1]))
                target_pt = (int(transformed_points[i][0] + ref_img.shape[1]), int(transformed_points[i][1]))
                cv2.line(vis_img, ref_pt, target_pt, (255, 0, 0), 2)

    # 对于MediaPipe方法
    else:
        # 绘制基准图像的关键点
        for pt in ref_points:
            if isinstance(pt, np.ndarray) and pt.size == 2:
                cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
            else:
                print(f"警告: 无效的点格式: {pt}")

        # 变换目标关键点
        transformed_points = cv2.perspectiveTransform(
            target_points.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        # 绘制变换后的目标点
        for pt in transformed_points:
            cv2.circle(vis_img, (int(pt[0] + ref_img.shape[1]), int(pt[1])), 2, (0, 0, 255), -1)

        # 绘制匹配线 (只绘制前50个点避免混乱)
        for i in range(min(50, len(ref_points), len(transformed_points))):
            if mask[i] > 0:  # 只绘制内点
                cv2.line(
                    vis_img,
                    (int(ref_points[i][0]), int(ref_points[i][1])),
                    (int(transformed_points[i][0] + ref_img.shape[1]), int(transformed_points[i][1])),
                    (255, 0, 0), 1
                )

    # 添加方法标签
    cv2.putText(vis_img, f"Method: {method}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return vis_img


if __name__ == "__main__":
    reference_path = r'../images2025-07-08\00eafc9b145f40f49ea4c236c8f549a6\00eafc9b145f40f49ea4c236c8f549a6-20230916-Rgb_Middle_T.jpg'
    target_path = r'../images2025-07-08\00eafc9b145f40f49ea4c236c8f549a6\00eafc9b145f40f49ea4c236c8f549a6-20231022-Rgb_Middle_T.jpg'

    # 选择关键点检测方法: 'mediapipe' 或 'sift'
    method = 'mediapipe'  # 可以更改为 'mediapipe' 进行对比

    try:
        # 获取对齐结果和点阵
        ref_img, aligned_img, M, mask, ref_points, target_points = align_faces(reference_path, target_path, method)

        # 可视化
        vis_img = visualize_alignment(ref_img, aligned_img, ref_points, target_points, M, mask, method)

        # 显示结果
        cv2.namedWindow("Alignment Visualization", cv2.WINDOW_NORMAL)
        cv2.imshow("Alignment Visualization", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite(f"ref_img.jpg", ref_img)
        cv2.imwrite(f"aligned_result_{method}.jpg", aligned_img)
        cv2.imwrite(f"alignment_visualization_{method}.jpg", vis_img)

        print(f"使用 {method} 方法对齐完成，结果已保存")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        import traceback

        traceback.print_exc()  # 打印完整错误堆栈
