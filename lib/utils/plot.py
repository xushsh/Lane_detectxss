import matplotlib.pyplot as plt
import cv2, os
import numpy as np
import random


def plot_img_and_mask(img, mask, index,epoch,save_dir):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig(save_dir+"/batch_{}_{}_seg.png".format(epoch,index))

def sort_contours_by_left(contours):
    # 创建一个列表来存储轮廓和其对应的左侧边界
    contours_with_left = [(contour, cv2.boundingRect(contour)[0]) for contour in contours]

    # 按照左侧边界排序
    contours_with_left.sort(key=lambda x: x[1])

    # 提取排序后的轮廓
    sorted_contours = [contour for contour, _ in contours_with_left]

    return sorted_contours

def distance_between_contours(contour1, contour2):
    # 计算两个轮廓质心的距离
    M1 = cv2.moments(contour1)
    M2 = cv2.moments(contour2)
    if M1["m00"] == 0 or M2["m00"] == 0:
        return float('inf')  # 如果轮廓面积为0，返回无穷大
    cX1 = int(M1["m10"] / M1["m00"])
    cY1 = int(M1["m01"] / M1["m00"])
    cX2 = int(M2["m10"] / M2["m00"])
    cY2 = int(M2["m01"] / M2["m00"])
    return np.sqrt((cX1 - cX2)**2 + (cY1 - cY2)**2)
# 相邻的分组
def group_close_contours(sort_contours, kept_indices):
    # 创建一个空字典来存储分组
    contour_groups = {}
    threshold = 200
    # 计算每个轮廓的最小外接矩形并获取中心点
    for i in kept_indices:
        # 尝试将轮廓加入到最近的组中
        min_distance = float('inf')
        closest_group = None
        # 遍历现有组
        for j, contour in enumerate(sort_contours):
                distance = distance_between_contours(sort_contours[i], sort_contours[j])
                if distance < threshold:
                    closest_group = i
                    if  closest_group is not None:
                        if i in contour_groups:
                            contour_groups[i].append(j)
                        else:
                            contour_groups[i] = [j]
    return contour_groups
#处理可行驶区域
def process_drivable_area(img, result, index, epoch, save_dir=None, is_ll=False, palette=None, is_demo=False, is_gt=False):

    if result[0].any():
        # 找到连续的True区域
        binary_image = np.uint8(result[0]) * 255
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            # 排除左边车道的影响
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                cX = int(M['m10'] / M['m00'])  # 质心的x坐标（即中心点离图像左边的距离）
                if cX < 200:
                    continue
            # 计算边界矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [center], -1, 255, -1)
            cropped_image = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow('Cropped Image', cropped_image)
            x1, y1, _ = cropped_image.shape

        cv2.imwrite(f"../crop_img/{key}.jpg", cropped_image)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
def show_seg_result(img, result, index, epoch, save_dir=None, is_ll=False, palette=None, is_demo=False, is_gt=False,img_name=None):
    # cv2.imwrite(f"../old_img/{img_name}.jpg", img)

    path = img_name.split("-")[0]
    crop_img_path = f"../crop_img/{path}"
    old_img_path =  f"../old_img/{path}"
    if not os.path.exists(crop_img_path):
        os.makedirs(crop_img_path)
    if not os.path.exists(old_img_path):
        os.makedirs(old_img_path)
    cv2.imencode('.jpg', img)[1].tofile(f"{old_img_path}/{img_name}.jpg")
    if palette is None:  # 如果调色板为空，则随机生成一个3x3的调色板
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]  # 将调色板的第一个颜色设置为黑色
    palette[1] = [0, 255, 0]  # 将调色板的第二个颜色设置为绿色
    palette[2] = [255, 0, 0]  # 将调色板的第三个颜色设置为红色
    palette = np.array(palette)  # 将调色板转换为NumPy数组
    assert palette.shape[0] == 3  # len(classes)   # 确保调色板有3种颜色（类别）
    assert palette.shape[1] == 3  # 确保每种颜色有3个通道（RGB）
    assert len(palette.shape) == 2  # 确保调色板是一个二维数组

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)  # 创建一个与result相同形状的三通道零矩阵
        for label, color in enumerate(palette):  # 根据结果的标签使用对应的调色板颜色填充color_seg
            color_seg[result == label, :] = color
    else:
        # 原图画线
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[1] == 1] = [0, 0, 255]
        color_seg = color_area
        color_seg = color_seg.astype(np.uint8)  # 将图像数据类型转换为无符号8位整型
        if result[1].any():
            # 找到连续的True区域
            binary_image = np.uint8( result[1]) * 255
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sort_contours = sort_contours_by_left(contours)
            # 设定阈值，用于判断矩形中心点是否相近
            threshold_distance = 200
            # 创建一个列表来存储要保留的轮廓索引
            kept_indices = []
            # 计算每个轮廓的面积和中心点
            for i, contour in enumerate(sort_contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                # 排除左边车道的影响
                M = cv2.moments(contour)
                if M['m00'] != 0.0:
                    cX = int(M['m10'] / M['m00'])  # 质心的x坐标（即中心点离图像左边的距离）
                    if cX < 50:
                        continue
                # 检查当前轮廓是否应该被保留
                is_kept = True
                for kept_idx in kept_indices:
                    kept_contour = sort_contours[kept_idx]
                    kept_area = cv2.contourArea(kept_contour)
                    # kept_x, kept_y, kept_w, kept_h = cv2.boundingRect(kept_contour)
                    # kept_center = (kept_x + kept_w // 2, kept_y + kept_h // 2)
                    # distance = distance_between_contours(kept_contour, contour)
                    M_kept = cv2.moments(kept_contour)
                    if M_kept['m00'] != 0.0:
                        cX_kept = int(M_kept['m10'] / M_kept['m00'])
                    # distance = np.linalg.norm(np.array(center) - np.array(kept_center))
                    # 检查两个矩形中心点的距离
                    if abs(cX_kept - cX) <= threshold_distance:
                        # 如果当前轮廓面积小于已保留的轮廓面积，则不保留当前轮廓
                        if area < kept_area :
                            is_kept = False
                            break
                        else:
                            # 如果当前轮廓面积大于已保留的轮廓面积，则删除已保留的轮廓
                            kept_indices.remove(kept_idx)
                            # 如果轮廓应该被保留，则添加到列表中
                if is_kept:
                    kept_indices.append(i)
                    # 遍历每个轮廓
            print("kept_indices:", kept_indices)
            # 合并相邻的轮廓
            contour_groups = group_close_contours(sort_contours, kept_indices, )
            print("contour_groups:",contour_groups)
            for key, value in contour_groups.items():
                if len(value) > 1:

                    mers = [cv2.minAreaRect(sort_contours[k]) for k in value]
                    minX, minY = float('inf'), float('inf')
                    maxX, maxY = -float('inf'), -float('inf')
                    for mer in mers:
                        box = cv2.boxPoints(mer)
                        box = np.int0(box)
                        x_coords, y_coords = box[:, 0], box[:, 1]
                        minX = min(minX, np.min(x_coords))
                        minY = min(minY, np.min(y_coords))
                        maxX = max(maxX, np.max(x_coords))
                        maxY = max(maxY, np.max(y_coords))
                        minX = max(minX, 0)
                        minY = max(minY, 0)
                        maxX = min(maxX, img.shape[1])
                        maxY = min(maxY, img.shape[0])
                    final_mer_tl = (minX, minY)  # 左上角
                    final_mer_br = (maxX, maxY)  # 右下角
                    # 从原图切出矩形区域,并切除矩形右上角和左下角三分之一的区域
                    crop_img = img[int(final_mer_tl[1]):int(final_mer_br[1]), int(final_mer_tl[0]):int(final_mer_br[0])]
                    # cv2.imshow("img", crop_img)
                    # cv2.waitKey(0)
                    width = final_mer_br[0] - final_mer_tl[0]
                    height = final_mer_br[1] - final_mer_tl[1]
                    print("crop_minx:", minX)
                    if minX >800:
                        pts = np.array(
                            [[0, 0], [width//5 * 1, 0], [width, height // 5 * 4], [width , height],
                             [width// 5 * 4, height], [0, height // 5 * 1]])
                        # pts = np.array([[cut_width // 5 * 3, 0], [cut_width , 0],[cut_width , cut_height//5 * 3],[cut_width // 5 * 3, cut_height], [0, cut_height],[0, cut_height//5 * 3]])
                        pts = np.array([pts])
                        # 和原始图像一样大小的0矩阵，作为mask
                        mask = np.zeros(crop_img.shape[:2], np.uint8)
                        # 在mask上将多边形区域填充为白色
                        cv2.polylines(mask, pts, 1, 255)  # 描绘边缘
                        cv2.fillPoly(mask, pts, 255)  # 填充
                        # 逐位与，得到裁剪后图像，此时是黑色背景
                        dst = cv2.bitwise_and(crop_img, crop_img, mask=mask)
                        # 添加白色背景
                        bg = np.ones_like(crop_img, np.uint8) * 0
                        cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
                        cropped_image = bg + dst
                        cv2.imencode('.jpg', cropped_image)[1].tofile(f"{crop_img_path}/{img_name}_right.jpg")
                    else:
                        # 创建一个掩码来切除这个区域
                        pts = np.array([[width // 5 * 4, 0], [width , 0],[width , height//5 * 1],[width // 5 * 1, height], [0, height],[0, height//3 * 2]])
                        # pts = np.array([[cut_width // 5 * 3, 0], [cut_width , 0],[cut_width , cut_height//5 * 3],[cut_width // 5 * 3, cut_height], [0, cut_height],[0, cut_height//5 * 3]])
                        pts = np.array([pts])
                        # 和原始图像一样大小的0矩阵，作为mask
                        mask = np.zeros(crop_img.shape[:2], np.uint8)
                        # 在mask上将多边形区域填充为白色
                        cv2.polylines(mask, pts, 1, 255)  # 描绘边缘
                        cv2.fillPoly(mask, pts, 255)  # 填充
                        # 逐位与，得到裁剪后图像，此时是黑色背景
                        dst = cv2.bitwise_and(crop_img, crop_img, mask=mask)
                        # 添加白色背景
                        bg = np.ones_like(crop_img, np.uint8) * 0
                        cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
                        cropped_image = bg + dst
                        cv2.imencode('.jpg', cropped_image)[1].tofile(f"{crop_img_path}/{img_name}_left.jpg")

                else:
                    contour = sort_contours[key]
                    # 计算边界矩形
                    rect  = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    center = (x1 + w1 // 2, y1 + h1 // 2)
                    x_coords, y_coords = box[:, 0], box[:, 1]
                    minX, minY = float('inf'), float('inf')
                    minX = min(minX, np.min(x_coords))
                    minX = max(minX, 0)


                    # ------------------------------------------------
                    # 合并相近的轮廓
                    # 每个方向上扩展10个像素
                    if minX < 1100:
                        horizontal_extension = 1  # 水平方向拓展10个像素
                        vertical_extension = 40  # 垂直方向拓展5个像素（可以根据需要调整）
                    else:
                        horizontal_extension = 40  # 水平方向拓展10个像素
                        vertical_extension = 1  # 垂直方向拓展5个像素（可以根据需要调整）
                    # 拓展旋转矩形的四个角点
                    # 首先，计算矩形宽度和高度的方向向量
                    # 这里我们使用box的第一个和第二个点来计算宽度方向向量
                    width_vector = box[1] - box[0]
                    height_vector = box[2] - box[1]
                    # width_vector = final_mer_br[0] - final_mer_tl[0]
                    # height_vector = final_mer_br[1] - final_mer_tl[1]  # 或者使用其他非相邻的点来计算高度方向向量

                    # 归一化向量以得到单位向量
                    width_vector_normalized = width_vector / np.linalg.norm(width_vector)
                    height_vector_normalized = height_vector / np.linalg.norm(height_vector)

                    # 计算拓展后的角点
                    # 拓展水平方向
                    box_expanded = box.astype(np.float32)
                    box_expanded[0] += -horizontal_extension * width_vector_normalized
                    box_expanded[1] += horizontal_extension * width_vector_normalized
                    box_expanded[2] += horizontal_extension * width_vector_normalized
                    box_expanded[3] += -horizontal_extension * width_vector_normalized

                    # 拓展垂直方向（如果需要）
                    box_expanded[0] += -vertical_extension * height_vector_normalized
                    box_expanded[1] += -vertical_extension * height_vector_normalized
                    box_expanded[2] += vertical_extension * height_vector_normalized
                    box_expanded[3] += vertical_extension * height_vector_normalized

                    # 确保扩展后的角点仍然在原图像的边界内
                    box_expanded = np.clip(box_expanded, (0, 0), (img.shape[1] - 1, img.shape[0] - 1)).astype(np.int32)
                    mask = np.zeros(img.shape[:2], dtype="uint8")
                    cv2.drawContours(mask, [box_expanded], -1, 255, -1)
                    cropped_image = cv2.bitwise_and(img, img, mask=mask)
                    # cv2.imshow('Cropped Image', cropped_image)
                    x1, y1, _ = cropped_image.shape
                    print("minx:",minX)
                    if minX > 1000:
                        cv2.imencode('.jpg', cropped_image)[1].tofile(f"{crop_img_path}/{img_name}_right.jpg")
                    else:
                        cv2.imencode('.jpg', cropped_image)[1].tofile(f"{crop_img_path}/{img_name}_left.jpg")
        color_mask = np.mean(color_seg, 2)  # 计算颜色分割的均值
        img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5  # 根据颜色分割结果修改图像
        img = img.astype(np.uint8)  # 将图像数据类型转换为无符号8位整型
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR) # 调整图像大小为 (1280, 720)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_segresult.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_segresult.png".format(epoch, index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_seg_gt.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_seg_gt.png".format(epoch, index), img)
    return img
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
def show_seg_result1(img, result, index, epoch, save_dir=None, is_ll=False, palette=None, is_demo=False, is_gt=False):
    if palette is None:  # 如果调色板为空，则随机生成一个3x3的调色板
        palette = np.random.randint(
            0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]  # 将调色板的第一个颜色设置为黑色
    palette[1] = [0, 255, 0]  # 将调色板的第二个颜色设置为绿色
    palette[2] = [255, 0, 0]  # 将调色板的第三个颜色设置为红色
    palette = np.array(palette)  # 将调色板转换为NumPy数组
    assert palette.shape[0] == 3  # len(classes)   # 确保调色板有3种颜色（类别）
    assert palette.shape[1] == 3  # 确保每种颜色有3个通道（RGB）
    assert len(palette.shape) == 2  # 确保调色板是一个二维数组

    if not is_demo:
        color_seg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)  # 创建一个与result相同形状的三通道零矩阵
        for label, color in enumerate(palette):  # 根据结果的标签使用对应的调色板颜色填充color_seg
            color_seg[result == label, :] = color
    else:
        # 原图画线
        color_area = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
        color_area[result[1] == 1] = [0, 0, 255]
        color_seg = color_area
        color_seg = color_seg.astype(np.uint8)  # 将图像数据类型转换为无符号8位整型
        if result[1].any():
            # 找到连续的True区域
            binary_image = np.uint8( result[1]) * 255
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sort_contours = sort_contours_by_left(contours)
            # 设定阈值，用于判断矩形中心点是否相近
            threshold_distance = 800
            # 创建一个列表来存储要保留的轮廓索引
            kept_indices = []
            # 计算每个轮廓的面积和中心点
            for i, contour in enumerate(sort_contours):
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                # 排除左边车道的影响
                M = cv2.moments(contour)
                if M['m00'] != 0.0:
                    cX = int(M['m10'] / M['m00'])  # 质心的x坐标（即中心点离图像左边的距离）
                    if cX < 200:
                        continue
                # 检查当前轮廓是否应该被保留
                is_kept = True
                for kept_idx in kept_indices:
                    kept_contour = sort_contours[kept_idx]
                    kept_area = cv2.contourArea(kept_contour)
                    kept_x, kept_y, kept_w, kept_h = cv2.boundingRect(kept_contour)
                    kept_center = (kept_x + kept_w // 2, kept_y + kept_h // 2)
                    # distance = distance_between_contours(kept_contour, contour)
                    distance = np.linalg.norm(np.array(center) - np.array(kept_center))
                    # 检查两个矩形中心点的距离
                    if distance <= threshold_distance:
                        # 如果当前轮廓面积小于已保留的轮廓面积，则不保留当前轮廓
                        if area < kept_area :
                            is_kept = False
                            break
                        else:
                            # 如果当前轮廓面积大于已保留的轮廓面积，则删除已保留的轮廓
                            kept_indices.remove(kept_idx)
                            # 如果轮廓应该被保留，则添加到列表中
                if is_kept:
                    kept_indices.append(i)
                    # 遍历每个轮廓
            print("kept_indices:", kept_indices)
            # 合并相邻的轮廓
            contour_groups = group_close_contours(sort_contours, kept_indices, )
            print("contour_groups:",contour_groups)
            for key, value in contour_groups.items():
                if len(value) > 1:
                    mers = [cv2.minAreaRect(sort_contours[k]) for k in value]
                    minX, minY = float('inf'), float('inf')
                    maxX, maxY = -float('inf'), -float('inf')
                    for mer in mers:
                        box = cv2.boxPoints(mer)
                        box = np.int0(box)
                        x_coords, y_coords = box[:, 0], box[:, 1]
                        minX = min(minX, np.min(x_coords))
                        minY = min(minY, np.min(y_coords))
                        maxX = max(maxX, np.max(x_coords))
                        maxY = max(maxY, np.max(y_coords))
                    final_mer_tl = (minX, minY)  # 左上角
                    final_mer_br = (maxX, maxY)  # 右下角
                    # 从原图切出矩形区域
                    crop_img = img[int(final_mer_tl[1]):int(final_mer_br[1]), int(final_mer_tl[0]):int(final_mer_br[0])]
                    cv2.imshow('cropmage', crop_img)
                    cv2.waitKey(0)
            for idx in kept_indices:

                contour = sort_contours[idx]
                # 计算边界矩形
                rect  = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                center = (x1 + w1 // 2, y1 + h1 // 2)


                # ------------------------------------------------
                # 合并相近的轮廓
                # 每个方向上扩展10个像素
                if idx // 2 == 0:
                    horizontal_extension = 5  # 水平方向拓展10个像素
                    vertical_extension = 30  # 垂直方向拓展5个像素（可以根据需要调整）
                else:
                    horizontal_extension = 30  # 水平方向拓展10个像素
                    vertical_extension = 5  # 垂直方向拓展5个像素（可以根据需要调整）
                # 拓展旋转矩形的四个角点
                # 首先，计算矩形宽度和高度的方向向量
                # 这里我们使用box的第一个和第二个点来计算宽度方向向量
                width_vector = box[1] - box[0]
                height_vector = box[2] - box[1]
                # width_vector = final_mer_br[0] - final_mer_tl[0]
                # height_vector = final_mer_br[1] - final_mer_tl[1]  # 或者使用其他非相邻的点来计算高度方向向量

                # 归一化向量以得到单位向量
                width_vector_normalized = width_vector / np.linalg.norm(width_vector)
                height_vector_normalized = height_vector / np.linalg.norm(height_vector)

                # 计算拓展后的角点
                # 拓展水平方向
                box_expanded = box.astype(np.float32)
                box_expanded[0] += -horizontal_extension * width_vector_normalized
                box_expanded[1] += horizontal_extension * width_vector_normalized
                box_expanded[2] += horizontal_extension * width_vector_normalized
                box_expanded[3] += -horizontal_extension * width_vector_normalized

                # 拓展垂直方向（如果需要）
                box_expanded[0] += -vertical_extension * height_vector_normalized
                box_expanded[1] += -vertical_extension * height_vector_normalized
                box_expanded[2] += vertical_extension * height_vector_normalized
                box_expanded[3] += vertical_extension * height_vector_normalized

                # 确保扩展后的角点仍然在原图像的边界内
                box_expanded = np.clip(box_expanded, (0, 0), (img.shape[1] - 1, img.shape[0] - 1)).astype(np.int32)
                mask = np.zeros(img.shape[:2], dtype="uint8")
                cv2.drawContours(mask, [box_expanded], -1, 255, -1)
                cropped_image = cv2.bitwise_and(img, img, mask=mask)
                cv2.imshow('Cropped Image', cropped_image)
                x1, y1, _ = cropped_image.shape

                cv2.imwrite(f"../crop_img/{idx}.jpg", cropped_image)

                # # ------------------------------------------------
                # (x, y), (w, h), angle = cv2.minAreaRect(box.astype(np.int32))
                # # 计算边界矩形的中心点
                # center = (int(x + w / 2), int(y + h / 2))
                #
                # # 将边界矩形的角点坐标计算出来
                # box_rect = cv2.boxPoints(cv2.minAreaRect(box.astype(np.int32)))
                # box_rect = np.int0(box_rect)
                #
                # # 获取旋转矩形的最小和最大坐标
                # min_x = np.min(box_rect[:, 0])
                # min_y = np.min(box_rect[:, 1])
                # max_x = np.max(box_rect[:, 0])
                # max_y = np.max(box_rect[:, 1])
                #
                # # 创建一个新的空白图像，大小足够容纳旋转矩形
                # height = int(max_y - min_y)
                # width = int(max_x - min_x)
                # rotated_img = np.zeros((height, width, img.shape[2]), dtype=img.dtype)
                #
                # # 计算旋转矩阵
                # M = cv2.getRotationMatrix2D(center, angle, 1)
                #
                # # 将原图像的旋转矩形部分映射到新的图像上
                # rotated_img = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR,
                #                              borderMode=cv2.BORDER_TRANSPARENT)
                # min_y_new = 0
                # min_x_new = 0
                # # 旋转后的图像可能包含旋转矩形之外的区域，所以我们需要裁剪它
                # cropped_img = rotated_img[int(min_y - min_y_new):int(max_y - min_y_new),
                #               int(min_x - min_x_new):int(max_x - min_x_new)]
                #
                # # 注意：这里 min_y_new 和 min_x_new 是用来调整裁剪位置的（如果必要的话），但在大多数情况下它们应该是0
                # # 因为我们已经创建了足够大的新图像来容纳整个旋转矩形
                #
                # # 显示或保存裁剪后的图像
                # x1, y1, _ = cropped_img.shape
                # if x1 and y1 > 0:
                #     # cv2.imwrite(f"../crop_img/{i}.jpg", cropped_rotated)
                #     cv2.imshow('Cropped Image', cropped_img)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                # ------------------------------------------------



                # 计算旋转矩形的中心点、宽度、高度和角度
                # (x, y), (width, height), angle = rect
                # # 获取四个顶点坐标
                # left_point_x = np.min(box[:, 0])
                # right_point_x = np.max(box[:, 0])
                # top_point_y = np.min(box[:, 1])
                # bottom_point_y = np.max(box[:, 1])
                #
                # left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
                # right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
                # top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
                # bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
                # # 上下左右四个点坐标
                # vertices = np.array(
                #     [[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],
                #      [right_point_x, right_point_y]])
                # cv2.drawContours(img, vertices, 0, (0, 255, 0), 2)
                #
                # # 将矩形和编号添加到列表中
                # rectangles_with_labels.append((i, rect))
                # cv2.imwrite("../seg_result/result.jpg", img)
                # 创建一个旋转矩阵，用于将矩形旋转到水平位置
                # 计算旋转矩形的中心点、宽度、高度和角度
                # (x, y), (width, height), angle = rect
                # width = width + 60
                # height = height + 60
                # # 创建一个旋转矩阵，用于将矩形旋转到水平位置
                # rotation_matrix = cv2.getRotationMatrix2D((x, y), angle, 1.0)
                #
                # # 计算仿射变换后的图像大小，确保旋转后的矩形完全可见
                # cos = np.abs(rotation_matrix[0, 0])
                # sin = np.abs(rotation_matrix[0, 1])
                #
                # # 计算新的图像边界
                # nW = int((height * sin) + (width * cos))
                # nH = int((height * cos) + (width * sin))
                #
                # # 调整旋转中心，使其适应新的图像尺寸
                # rotation_matrix[0, 2] += (nW / 2) - x
                # rotation_matrix[1, 2] += (nH / 2) - y
                #
                # # 执行仿射变换（旋转）
                # rotated_image = cv2.warpAffine(img, rotation_matrix, (nW, nH))
                #
                # # 在旋转后的图像上裁剪出矩形区域
                # # 由于旋转矩阵已经将其移动到了中心，所以我们可以直接计算新的位置
                # x, y = int(nW / 2 - width / 2), int(nH / 2 - height / 2)
                # cropped_rotated = rotated_image[y:int(y + height), x:int(x + width)]
                # x1, y1, _ = cropped_rotated.shape
                # if x1 and y1 > 0:
                #
                #     cv2.imwrite(f"../crop_img/{i}.jpg", cropped_rotated)
                # # 显示结果

        color_mask = np.mean(color_seg, 2)  # 计算颜色分割的均值
        img[color_mask != 0] = img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5  # 根据颜色分割结果修改图像
        img = img.astype(np.uint8)  # 将图像数据类型转换为无符号8位整型
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR) # 调整图像大小为 (1280, 720)

    if not is_demo:
        if not is_gt:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_segresult.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_segresult.png".format(epoch, index), img)
        else:
            if not is_ll:
                cv2.imwrite(save_dir + "/batch_{}_{}_da_seg_gt.png".format(epoch, index), img)
            else:
                cv2.imwrite(save_dir + "/batch_{}_{}_ll_seg_gt.png".format(epoch, index), img)
    return img

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.0001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=1)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


if __name__ == "__main__":
    pass