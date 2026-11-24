import argparse
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


LANG = 'rus+eng'        
PSM = 6                 
MIN_CONF = 30
SHOW_WINDOWS = True
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    'C:/Windows/Fonts/arial.ttf',
]

FONT_SCORE = ['dejavu', 'liberation', 'freesans', 'noto', 'roboto', 'arial',
              'pt-sans', 'ptsans', 'jetbrains', 'opensans', 'open-sans']
FONT_DIRS = ['/usr/share/fonts', '/usr/local/share/fonts',
             os.path.expanduser('~/.fonts'),
             os.path.expanduser('~/.local/share/fonts')]


def find_font():
    here = Path(__file__).resolve().parent
    local = sorted(here.glob('*.ttf')) + sorted(here.glob('*.otf'))
    if local:
        return str(local[0])

    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p

    found = []
    for d in FONT_DIRS:
        for ext in ('**/*.ttf', '**/*.otf'):
            found.extend(Path(d).glob(ext)) if os.path.isdir(d) else None
    scored = []
    for f in found:
        name = f.name.lower()
        score = next((i for i, key in enumerate(FONT_SCORE) if key in name), 99)
        scored.append((score, str(f)))
    scored.sort()
    return scored[0][1] if scored else None

def ocr(image, lang, psm, min_conf):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 127:                
        gray = 255 - gray
    cfg = f'--psm {psm} --oem 3'
    d = pytesseract.image_to_data(gray, lang=lang, config=cfg,
                                  output_type=pytesseract.Output.DICT)
    items = []
    for i in range(len(d['text'])):
        txt = d['text'][i].strip()
        conf = int(d['conf'][i])
        if not txt or conf < min_conf:
            continue
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        if w < 2 or h < 2:
            continue
        items.append({'text': txt, 'conf': conf, 'bbox': (x, y, w, h),
                      'cy': y + h / 2.0, 'h': h})
    return items


def build_text(items):
    if not items:
        return ''
    lines = []
    for it in sorted(items, key=lambda i: i['cy']):
        for line in lines:
            if abs(it['cy'] - line['cy']) < 0.5 * (line['h'] + it['h']) / 2 + 2:
                n = line['n']
                line['cy'] = (line['cy'] * n + it['cy']) / (n + 1)
                line['h'] = (line['h'] * n + it['h']) / (n + 1)
                line['n'] += 1
                line['items'].append(it)
                break
        else:
            lines.append({'cy': it['cy'], 'h': it['h'], 'n': 1, 'items': [it]})
    lines.sort(key=lambda l: l['cy'])
    out = []
    for line in lines:
        words = sorted(line['items'], key=lambda i: i['bbox'][0])
        out.append(' '.join(w['text'] for w in words))
    return '\n'.join(out)


def draw_labels(img, items, font_path):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, 18)
    for it in items:
        x, y, w, h = it['bbox']
        dr.rectangle([x, y, x + w, y + h], outline=(0, 180, 0), width=2)
        dr.text((x, max(0, y - 22)), it['text'], font=font, fill=(255, 0, 0))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser(description='OCR документов (Tesseract)')
    p.add_argument('--image', default='text.jpg', help='путь к изображению')
    p.add_argument('--lang', default=LANG, help="языки: rus | eng | rus+eng")
    p.add_argument('--psm', type=int, default=PSM)
    p.add_argument('--save', metavar='FILE')
    a = p.parse_args()

    try:
        print(f'Tesseract: {pytesseract.get_tesseract_version()}, языки: {a.lang}')
    except pytesseract.TesseractNotFoundError:
        raise SystemExit('Нет tesseract: sudo apt install tesseract-ocr tesseract-ocr-rus')

    image = cv2.imread(a.image)
    if image is None:
        raise SystemExit(f'Не удалось открыть {a.image}')

    font_path = find_font()
    if font_path is None:
        raise SystemExit('Не найден TTF-шрифт для подписей (DejaVu/Arial).')

    try:
        items = ocr(image, a.lang, a.psm, MIN_CONF)
    except pytesseract.TesseractError as e:
        raise SystemExit(f'Ошибка Tesseract (проверь языки: tesseract --list-langs):\n{e}')

    text = build_text(items)
    print('--- Распознано ---')
    print(text if text else '(пусто)')
    print('------------------')
    print(f'Слов: {len(items)}')

    vis = draw_labels(image, items, font_path)
    if a.save:
        cv2.imwrite(a.save, vis)
        print(f'Сохранено: {a.save}')
    if SHOW_WINDOWS:
        h, w = vis.shape[:2]
        scale = min(1.0, 1400 / w, 800 / h)
        small = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow('recognized', small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

