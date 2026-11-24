import argparse
import cv2
import numpy as np


def make_detector(name):
    name = name.lower()
    if name == 'sift':
        return cv2.SIFT_create()
    if name == 'surf':
        return cv2.xfeatures2d.SURF_create()


def quad_is_sane(corners, scene_shape, min_area=0.01, max_area=0.95, max_edge_ratio=4.0):
    pts = corners.reshape(-1, 2).astype(np.float64)

    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    H, W = scene_shape[:2]
    if not (min_area * W * H <= area <= max_area * W * H):
        return False

    signs = []
    for i in range(4):
        v1 = pts[(i + 1) % 4] - pts[i]
        v2 = pts[(i + 2) % 4] - pts[(i + 1) % 4]
        signs.append(v1[0] * v2[1] - v1[1] * v2[0])
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return False

    edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    if min(edges) < 10 or max(edges) / min(edges) > max_edge_ratio:
        return False

    return True


def recognize(obj_img, scene_img, name='sift', ratio=0.75, min_matches=8, ransac=5.0):
    detector = make_detector(name)
    norm = cv2.NORM_L2         

    obj_gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
    kp_obj, des_obj = detector.detectAndCompute(obj_gray, None)

    gray = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)
    kp_scene, des_scene = detector.detectAndCompute(gray, None)

    res = {'found': False, 'n_good': 0, 'n_inliers': 0, 'corners': None, 'H': None,
           'kp_obj': kp_obj or [], 'kp_scene': kp_scene or [],
           'good': [], 'inliers': None}
    if des_obj is None or des_scene is None:
        return res

    matcher = cv2.BFMatcher(norm)
    good = []
    for pair in matcher.knnMatch(des_obj, des_scene, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    res['good'], res['n_good'] = good, len(good)
    if len(good) < max(4, min_matches):
        return res

    src = np.float32([kp_obj[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac)
    if H is None:
        return res
    inliers = mask.ravel().tolist()
    res['inliers'], res['n_inliers'], res['H'] = inliers, sum(inliers), H
    if res['n_inliers'] < 4:
        return res

    h, w = obj_gray.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(corners, H)
    if not quad_is_sane(corners, scene_img.shape):
        return res              

    res['corners'] = corners
    res['found'] = True
    return res


def extract_found(scene_img, H, obj_shape):
    h, w = obj_shape[:2]
    return cv2.warpPerspective(scene_img, np.linalg.inv(H), (w, h))


def draw(scene_img, obj_img, res):
    scene_draw = scene_img.copy()
    if res['found']:
        pts = np.int32(res['corners']).reshape(-1, 2)
        cv2.polylines(scene_draw, [pts], True, (0, 0, 255), 3)
    if res['good']:
        swapped = [cv2.DMatch(m.trainIdx, m.queryIdx, m.distance) for m in res['good']]
        return cv2.drawMatches(
            scene_draw, res['kp_scene'], obj_img, res['kp_obj'], swapped, None,
            matchColor=(180, 160, 255), singlePointColor=(255, 0, 0),
            matchesMask=res['inliers'],
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return scene_draw


def main():
    p = argparse.ArgumentParser(
        description='Поиск объекта по образцу'
    )

    p.add_argument(
        'scene',
        nargs='?',
        default='scene.jpg',
        help='файл изображения сцены'
    )

    p.add_argument(
        'object',
        nargs='?',
        default='object.jpg',
        help='файл изображения объекта'
    )

    p.add_argument(
        '--detector',
        default='sift',
        choices=['sift', 'surf']
    )

    p.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='показать результат'
    )

    a = p.parse_args()

    scene = cv2.imread(a.scene)
    obj = cv2.imread(a.object)
    if scene is None:
        raise SystemExit(f'Не удалось открыть сцену: {a.scene}')
    if obj is None:
        raise SystemExit(f'Не удалось открыть образец: {a.object}')

    res = recognize(obj, scene, a.detector)

    print(f'Детектор:         {a.detector}')
    print(f'Точек на образце: {len(res["kp_obj"])}')
    print(f'Точек на сцене:   {len(res["kp_scene"])}')
    print(f'Совпадений (Лоу): {res["n_good"]}')
    print(f'Инлайеров RANSAC: {res["n_inliers"]}')

    if res['found']:
        print('Объект НАЙДЕН')
        found = extract_found(scene, res['H'], obj.shape)
        canvas = draw(scene, obj, res)
        cv2.imwrite('found_object.jpg', found)
        cv2.imwrite('result.jpg', canvas)
        print('Сохранено: found_object.jpg (найденный объект), result.jpg (сцена с рамкой)')
        if a.show:
            cv2.imshow('found object', found)

            h, w = canvas.shape[:2]
            scale = min(1.0, 1400 / w, 800 / h)
            small = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imshow('result', small)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print('Объект НЕ найден — изображение не сохранено')


if __name__ == '__main__':
    main()